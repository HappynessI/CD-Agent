"""Shared Qwen input, answer-only loss and label-free scoring for V2/V3."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration, StopStringCriteria

from common import WORKSPACE
from exp3_mllm import GENERAL_PROMPT
from exp6b_common import binary_metrics, verifier_split, write_json

MODEL = WORKSPACE / "models/Qwen3.5-4B"
MARKER = "答案："
PROMPT = GENERAL_PROMPT.split("Answer with JSON only:")[0].replace("red box", "red contour")
INSTRUCTIONS = {
    "a": "\n只判断红色轮廓内的地物变化。最后一行只写“答案：是”或“答案：否”。",
    "b": ("\n只判断红色轮廓内的地物变化。先用一句中文说明两期地物及变化类型；"
          "无变化时说明仅色调或成像差异。最后另起一行写“答案：是”或“答案：否”。"),
}


def read_rows(path, split):
    path = Path(path)
    report = json.loads((path.parent / "data_report.json").read_text())
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle]
    train, val = verifier_split()
    if (report["train_images"] != train or report["val_images"] != val
            or len(rows) != report["splits"][split]["sampled"]):
        raise ValueError(f"{path}: incomplete data or changed verifier split")
    allowed = set(train if split == "train" else val) if split != "test" else None
    dataset = "second" if split == "test" else "second_cv"
    keys = set()
    for row in rows:
        key = (row["image"], row["piece_id"])
        if (row["dataset"] != dataset or row["split"] != split or row["kind"] != "recall"
                or row["label"] not in (0, 1) or key in keys
                or (allowed is not None and row["image"] not in allowed)):
            raise ValueError(f"{path}: inconsistent split or duplicate piece {key}")
        keys.add(key)
    if not rows:
        raise ValueError(f"{path}: empty sample")
    return rows


def load_model(model_path=MODEL, adapter=None, trainable=False, rank=16, alpha=32):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa")
    base.requires_grad_(False)
    if adapter is not None:
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable, local_files_only=True)
    else:
        # Include dense MLPs and both full-attention and gated-delta projections.
        targets = [name for name, module in base.named_modules()
                   if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)]
        if not targets:
            raise ValueError("Qwen language projections were not found")
        model = get_peft_model(base, LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM"))
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ("lora_" not in name or ".visual." in name):
            raise ValueError(f"Unexpected trainable parameter: {name}")
    return processor, model.eval()


def input_batch(processor, row, variant, images=None):
    if images is None:
        images = []
        for view in ("t1", "t2"):
            with Image.open(row["inputs"][view]) as image:
                images.append(image.convert("RGB"))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 (T1, earlier):"},
        {"type": "image", "image": images[0]},
        {"type": "text", "text": "Image 2 (T2, later):"},
        {"type": "image", "image": images[1]},
        {"type": "text", "text": PROMPT + INSTRUCTIONS[variant]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    batch = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
    return append_ids(batch, encode(processor, MARKER)) if variant == "a" else batch


def encode(processor, text):
    return processor.tokenizer.encode(text, add_special_tokens=False)


def append_ids(batch, ids):
    suffix = torch.as_tensor(ids, device=batch["input_ids"].device, dtype=torch.long).view(1, -1)
    result = dict(batch)
    result["input_ids"] = torch.cat((batch["input_ids"], suffix), dim=1)
    result["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(suffix)), dim=1)
    # Processor-provided type IDs describe image spans; appended tokens are text.
    if "mm_token_type_ids" in batch:
        result["mm_token_type_ids"] = torch.cat((batch["mm_token_type_ids"], torch.zeros_like(suffix)), dim=1)
    return result


def target_ids(processor, row, variant):
    answer = "是" if row["label"] else "否"
    prefix = row["target_b"].rsplit(MARKER, 1)[0] + MARKER if variant == "b" else ""
    # Explicit boundary keeps the supervised answer tokens identical to the scoring tokens.
    return encode(processor, prefix) + encode(processor, answer) + [processor.tokenizer.eos_token_id]


def response_logps(model, batch, response, temperature=1.0):
    """Only project response positions to the large vocabulary; no prompt loss."""
    full = append_ids(batch, response)
    n = len(response)
    logits = model(**full, use_cache=False, logits_to_keep=n + 1).logits[0, :-1].float() / temperature
    tokens = torch.as_tensor(response, device=logits.device, dtype=torch.long)
    return -F.cross_entropy(logits, tokens, reduction="none")


@lru_cache(maxsize=2)
def answer_stop(tokenizer):
    return StopStringCriteria(tokenizer, [MARKER])


@torch.no_grad()
def score_one(processor, model, row, variant, max_new_tokens=96, images=None):
    batch = input_batch(processor, row, variant, images)
    raw, status = "", "answer_only"
    if variant == "b":
        stop = answer_stop(processor.tokenizer)
        generated = model.generate(
            **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            stopping_criteria=[stop], pad_token_id=processor.tokenizer.eos_token_id)
        raw = processor.tokenizer.decode(generated[0, batch["input_ids"].shape[1]:],
                                         skip_special_tokens=True)
        if MARKER in raw:
            reason = raw.split(MARKER, 1)[0]
            status = "generated_marker"
        else:
            reason = raw.rstrip() + "\n"
            status = "forced_marker_after_generation"
        batch = append_ids(batch, encode(processor, reason + MARKER))
    ids = [encode(processor, answer)[0] for answer in ("是", "否")]
    if ids[0] == ids[1]:
        raise ValueError("Yes/no first tokens are identical")
    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    delta = float(logits[ids[0]] - logits[ids[1]])
    if not np.isfinite(delta):
        raise ValueError("Non-finite answer score")
    return dict(logit_diff=delta, score=float(torch.sigmoid(torch.tensor(delta, dtype=torch.float64))),
                raw=raw, status=status)


def score_rows(processor, model, rows, variant, max_new_tokens=96, journal=None):
    model.eval()
    torch.cuda.synchronize()
    start = time.monotonic()
    results = []
    for i, row in enumerate(rows):
        result = score_one(processor, model, row, variant, max_new_tokens)
        result.update(image=row["image"], piece_id=row["piece_id"])
        results.append(result)
        if journal is not None:
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
        if (i + 1) % 100 == 0:
            print(f"score {i + 1}/{len(rows)}", flush=True)
    torch.cuda.synchronize()
    return results, time.monotonic() - start


def validation(processor, model, rows, variant, max_new_tokens):
    results, seconds = score_rows(processor, model, rows, variant, max_new_tokens)
    metrics = binary_metrics([r["label"] for r in rows], [r["logit_diff"] for r in results])
    if metrics["auc"] is None:
        raise ValueError("Validation needs both classes")
    metrics.update(seconds=seconds, forced_marker_count=sum(
        r["status"] == "forced_marker_after_generation" for r in results))
    return metrics, results


def save_adapter(model, processor, out, metadata):
    folder = Path(out) / "adapter"
    model.save_pretrained(folder, safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(folder)
    write_json(Path(out) / "best.json", metadata)


def read_adapter(adapter, variant):
    metadata = json.loads((Path(adapter).parent / "best.json").read_text())
    train, val = verifier_split()
    if (metadata["variant"] != variant or metadata["train_images"] != train
            or metadata["val_images"] != val):
        raise ValueError("Adapter variant or verifier image split differs")
    return metadata
