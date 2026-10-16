"""Experiment 3 with a hosted Bailian model: same candidates, crops and prompt as exp3_mllm.py.

Launch through tools/run_with_bailian_csv.py (DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL in the
environment; never logged). Only datasets covered by the user's Bailian authorization may be
passed (LEVIR, and WHU since the 2026-10-16 authorization). Writes exp3/<dataset>/mllm_<tag>.jsonl in the exp3_mllm
format plus usage. p_yes comes from token logprobs when the endpoint returns them, else None.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import OUT
from exp3_mllm import parse, prompt_for

# WHU, SYSU-CD and SECOND added by user authorizations on 2026-10-16 (see AGENTS.md).
AUTHORIZED = {"levir", "whu", "sysu", "second"}


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def request(body, retries=4):
    url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + os.environ["DASHSCOPE_API_KEY"],
                                          "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as error:
            detail = error.read(2000).decode("utf-8", "replace")
            if error.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {error.code}: {detail[:500]}") from None
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            raise


def yes_probability(choice):
    """P(yes)/(P(yes)+P(no)) at the first answer token, from top_logprobs."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    for entry in content:
        if entry["token"].strip().strip('"').lower() in ("yes", "no"):
            mass = {"yes": 0.0, "no": 0.0}
            for alt in entry.get("top_logprobs") or [entry]:
                key = alt["token"].strip().strip('"').lower()
                if key in mass:
                    mass[key] += math.exp(alt["logprob"])
            total = mass["yes"] + mass["no"]
            return (mass["yes"] / total if total else None), total
    return None, None


def judge(record, model, logprobs, prompt):
    content = [{"type": "text", "text": "Image 1 (T1, earlier):"},
               {"type": "image_url", "image_url": {"url": data_url(record["inputs"]["t1"])}},
               {"type": "text", "text": "Image 2 (T2, later):"},
               {"type": "image_url", "image_url": {"url": data_url(record["inputs"]["t2"])}},
               {"type": "text", "text": prompt}]
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, "max_completion_tokens": 64,
            "temperature": 0, "seed": 42, "enable_thinking": False, "stream": False}
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    start = time.time()
    response = request(body)
    choice = response["choices"][0]
    text = choice["message"]["content"]
    answer, conf = parse(text)
    p_yes, mass = yes_probability(choice) if logprobs else (None, None)
    return {"cid": record["cid"], "p_yes": p_yes, "yes_no_mass": mass, "raw": text, "answer": answer,
            "confidence": conf, "usage": response.get("usage"), "seconds": round(time.time() - start, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="qwen3.7-plus-2026-05-26")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.dataset not in AUTHORIZED:
        raise SystemExit(f"{args.dataset} is not covered by the Bailian authorization")
    base = OUT / "exp3" / args.dataset
    rows = [json.loads(line) for line in open(base / "candidates.jsonl")][: args.limit or None]
    prompt = prompt_for(args.dataset)
    # One probe call decides whether the endpoint accepts logprobs for this model.
    try:
        judge(rows[0], args.model, True, prompt)
        logprobs = True
    except RuntimeError as error:
        print("logprobs not accepted, using verbalized confidence only:", str(error)[:200], flush=True)
        logprobs = False
    out_path = base / f"mllm_{args.tag}.jsonl"
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool, open(out_path, "w") as f:
        def safe(record):
            try:
                return judge(record, args.model, logprobs, prompt)
            except Exception as error:  # keep the run; the failure is recorded and counted as unparsed
                return {"cid": record["cid"], "p_yes": None, "yes_no_mass": None, "raw": "", "answer": None,
                        "confidence": None, "error": str(error)[:300]}

        for i, result in enumerate(pool.map(safe, rows)):
            result.update(model=args.model, logprobs_requested=logprobs, prompt=prompt if i == 0 else None)
            f.write(json.dumps(result) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} {time.time() - start:.0f}s", flush=True)
    print("done", out_path, flush=True)


if __name__ == "__main__":
    main()
