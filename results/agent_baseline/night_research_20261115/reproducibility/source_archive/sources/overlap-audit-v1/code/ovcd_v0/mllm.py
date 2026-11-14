"""Local Qwen3.5-4B for T3 (verify) and T5 (box).

T3: P(yes) is read from the first generated token after the forced prefix '{"change": "'
(same readout as exp6j/exp6l: softmax over the yes/no logits), and the rest of the JSON (type,
reason) is generated greedily in the same call.  T5: greedy generation after '{"bbox_2d": ['.
"""
from __future__ import annotations
import json
import math
import re
import time

import prompts as P


class Qwen:
    def __init__(self, path):
        import torch
        import transformers
        self.torch = torch
        self.processor = transformers.AutoProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'}).eval()
        tok = self.processor.tokenizer
        base = tok.encode(P.PREFIX, add_special_tokens=False)
        ids = {}
        for w in ('yes', 'no'):
            full = tok.encode(P.PREFIX + w, add_special_tokens=False)
            if full[:len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(f'yes/no is not a single token after the prefix: {w}')
            ids[w] = full[len(base)]
        self.yes_no = [ids['yes'], ids['no']]
        self.meta = dict(model=str(path), transformers=transformers.__version__, torch=torch.__version__,
                         dtype='bfloat16', thinking=False, yes_no_ids=self.yes_no)

    def _generate(self, pairs, question, prefix, max_new_tokens):
        """Batched greedy generation; every request has the same template and two 384x384
        crops, so all rows have the same length and no padding is involved."""
        msgs = [P.messages(t1, t2, question) for t1, t2 in pairs]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) + prefix for m in msgs]
        images = [im for t1, t2 in pairs for im in (t1, t2)]
        batch = self.processor(text=texts, images=images, return_tensors='pt', padding=True).to('cuda:0')
        if 'attention_mask' in batch and not bool(batch['attention_mask'].all()):
            raise ValueError('rows differ in length; batching would need padding')
        with self.torch.inference_mode():
            out = self.model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                      output_logits=True, return_dict_in_generate=True)
        n = batch['input_ids'].shape[1]
        gens = self.processor.tokenizer.batch_decode(out.sequences[:, n:], skip_special_tokens=True)
        return gens, out.logits[0].float(), int(n)

    def verify(self, pairs):
        """T3 for a batch of (T1 crop, T2 crop); seconds = batch wall time / batch size."""
        t0 = time.time()
        gens, logits, n = self._generate(pairs, P.VERIFY, P.PREFIX, 96)
        sec = (time.time() - t0) / len(pairs)
        out = []
        for gen, lg in zip(gens, logits):
            pair = lg[self.yes_no]
            raw = P.PREFIX + gen
            parsed = parse_json(raw)
            out.append(dict(p_yes=float(pair.softmax(0)[0]), logit_diff=float(pair[0] - pair[1]),
                            yes_no_mass=float((pair - lg.logsumexp(0)).exp().sum()), raw=raw,
                            answer=parsed.get('change') if parsed else None,
                            type=parsed.get('type') if parsed else None,
                            reason=parsed.get('reason') if parsed else None,
                            input_tokens=n, seconds=sec, batch=len(pairs)))
        return out

    def box(self, pairs):
        t0 = time.time()
        gens, _, n = self._generate(pairs, P.BOX, P.BOX_PREFIX, 40)
        sec = (time.time() - t0) / len(pairs)
        return [dict(raw=P.BOX_PREFIX + g, box=parse_box(P.BOX_PREFIX + g), input_tokens=n, seconds=sec,
                     batch=len(pairs)) for g in gens]


def parse_json(raw):
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_box(raw):
    """The bbox_2d list of a complete JSON answer as a valid 0-1000 box, else None (-> fallback)."""
    obj = parse_json(raw)
    box = obj.get('bbox_2d') if isinstance(obj, dict) else None
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 2 or y1 - y0 < 2 or x1 > 1000 or y1 > 1000 or x0 < 0 or y0 < 0:
        return None
    return [x0, y0, x1, y1]
