"""Experiment 3, step 2: ask a local MLLM whether each candidate region is a real change.

Input: the t1 and t2 crops (red box marks the region). Two readouts per candidate:
  p_yes      P("yes") / (P("yes") + P("no")) at the answer slot after the forced prefix
             '{"change": "', i.e. the model's own token probability;
  generated  greedy JSON answer {"change": "yes|no", "confidence": 0-100}.
Thinking is disabled. Run in conda_envs/vlm-vs-onejev-env on one GPU.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import transformers
from PIL import Image

from common import DATASETS, OUT

PROMPT = (
    "Image 1 and Image 2 are co-registered remote-sensing images of the same area: Image 1 is the "
    "earlier date (T1) and Image 2 the later date (T2). The red box marks the same location in both.\n"
    "Question: inside the red box, is there a real building change between T1 and T2, i.e. a building "
    "newly constructed or demolished (including clearly started construction)? Ignore differences "
    "caused only by illumination, season, shadows, vegetation, vehicles, or slight misregistration.\n"
    'Answer with JSON only: {"change": "yes" or "no", "confidence": integer 0-100 for your answer}'
)
GENERAL_PROMPT = (
    "Image 1 and Image 2 are co-registered remote-sensing images of the same area: Image 1 is the "
    "earlier date (T1) and Image 2 the later date (T2). The red box marks the same location in both.\n"
    "Question: inside the red box, is there a real land-cover change between T1 and T2, for example a "
    "building constructed or demolished, a road or other hard surface built, vegetation cleared or "
    "planted, water area changed, or bare land converted? Ignore differences caused only by "
    "illumination, seasonal colour of the same vegetation, shadows, vehicles, or slight misregistration.\n"
    'Answer with JSON only: {"change": "yes" or "no", "confidence": integer 0-100 for your answer}'
)
PREFIX = '{"change": "'


def prompt_for(dataset):
    """Building-change datasets keep the original prompt; SYSU/SECOND use the general one."""
    return PROMPT if DATASETS[dataset]["task"] == "building" else GENERAL_PROMPT


def load(model_path):
    common = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(model_path, **common)
    processor = transformers.AutoProcessor.from_pretrained(model_path, **common)
    cls = transformers.Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" \
        else transformers.AutoModelForImageTextToText
    model = cls.from_pretrained(model_path, config=config, dtype=torch.bfloat16, device_map={"": "cuda:0"}, **common)
    return processor, model.eval()


def inputs_for(processor, record, prompt, prefix=""):
    images = [Image.open(record["inputs"][v]).convert("RGB") for v in ("t1", "t2")]
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 (T1, earlier):"}, {"type": "image", "image": images[0]},
        {"type": "text", "text": "Image 2 (T2, later):"}, {"type": "image", "image": images[1]},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) + prefix
    return processor(text=[text], images=images, return_tensors="pt").to("cuda:0"), text


def parse(text):
    match = re.search(r'"change"\s*:\s*"(yes|no)"', text)
    conf = re.search(r'"confidence"\s*:\s*(\d+)', text)
    return (match.group(1) if match else None), (int(conf.group(1)) if conf else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    processor, model = load(args.model)
    tok = processor.tokenizer
    yes_no = [tok.encode(w, add_special_tokens=False) for w in ("yes", "no")]
    assert all(len(ids) == 1 for ids in yes_no), yes_no
    yes_id, no_id = yes_no[0][0], yes_no[1][0]
    for dataset in args.datasets:
        prompt_text = prompt_for(dataset)
        rows = [json.loads(line) for line in open(OUT / "exp3" / dataset / "candidates.jsonl")][: args.limit or None]
        out_path = OUT / "exp3" / dataset / f"mllm_{args.tag}.jsonl"
        start = time.time()
        with open(out_path, "w") as f:
            for i, r in enumerate(rows):
                with torch.inference_mode():
                    batch, prompt = inputs_for(processor, r, prompt_text, PREFIX)
                    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                    pair = logits[[yes_id, no_id]]
                    p_yes = float(pair.softmax(0)[0])
                    mass = float((pair - logits.logsumexp(0)).exp().sum())
                    batch, _ = inputs_for(processor, r, prompt_text)
                    gen = model.generate(**batch, max_new_tokens=40, do_sample=False)
                text = tok.decode(gen[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                answer, conf = parse(text)
                f.write(json.dumps({"cid": r["cid"], "p_yes": p_yes, "yes_no_mass": mass, "raw": text,
                                    "answer": answer, "confidence": conf, "prompt": prompt if i == 0 else None}) + "\n")
                if (i + 1) % 50 == 0:
                    print(dataset, f"{i + 1}/{len(rows)} {time.time() - start:.0f}s", flush=True)
        print(dataset, "done", out_path, flush=True)


if __name__ == "__main__":
    main()
