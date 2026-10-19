"""Build results/pipeline.html from probe/exp6_pipeline_template.html (no network, all inline).

Placeholders: {{img:<path under figures/pipeline>}} -> JPEG data URI; {{json:<key>}} -> a pretty
JSON block from the example image's example.json (keys: pos, neg, sources) or the fold-0 train
config (train_config); {{sek_svg}} -> validation SeK curves of the three cross-fitting folds.
Usage: python exp6_pipeline_page.py --example test_123
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re

from common import OUT

FIG = OUT / "figures/pipeline"
TEMPLATE = OUT.parents[1] / "probe/exp6_pipeline_template.html"
CV = OUT / "exp6/second_cv"
TRAIN_KEYS = ("model_type", "encoder_pretrained_path", "max_iters", "global_batch_size", "batch_size",
              "world_size", "learning_rate", "weight_decay", "lr_step_size", "crop_size", "fold")


def sek_svg():
    width, height, pad = 640, 220, 36
    colours = ("#2d6cdf", "#d0632a", "#1f9a5c")
    curves = []
    for k in range(3):
        rows = [json.loads(l) for l in open(CV / f"fold{k}/training.jsonl")]
        curves.append([(r["iteration"], r["results"]["Validation"]["sek"]) for r in rows])
    lo, hi = 0.15, 0.26
    x = lambda it: pad + (width - 2 * pad) * it / 50000
    y = lambda v: height - pad - (height - 2 * pad) * (v - lo) / (hi - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="验证 SeK 曲线">']
    for v in (0.16, 0.18, 0.20, 0.22, 0.24, 0.26):
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>'
                     f'<text x="{pad - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for it in range(0, 50001, 10000):
        parts.append(f'<text x="{x(it):.1f}" y="{height - pad + 16}" text-anchor="middle">{it // 1000}k</text>')
    for k, (pts, c) in enumerate(zip(curves, colours)):
        path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in pts)
        best = max(pts, key=lambda p: p[1])
        parts.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="1.6"/>'
                     f'<circle cx="{x(best[0]):.1f}" cy="{y(best[1]):.1f}" r="4" fill="{c}"/>'
                     f'<text x="{width - pad + 4}" y="{pad + 14 * k}" fill="{c}">fold{k}</text>')
    parts.append(f'<text x="{width / 2}" y="{height - 4}" text-anchor="middle">训练步数（圆点 = 选中的最优 checkpoint）</text></svg>')
    last = max(c[-1][0] for c in curves)
    return "\n".join(parts), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", required=True)
    parser.add_argument("--out", default=str(OUT / "results/pipeline.html"))
    args = parser.parse_args()
    info = json.loads((FIG / args.example / "example.json").read_text())
    config = json.loads((CV / "fold0/train_config.json").read_text())["args"]
    blocks = {"pos": info["verifier_examples"]["pos"], "neg": info["verifier_examples"]["neg"],
              "sources": info["per_source"], "f1": info["f1_image"],
              "train_config": {k: config[k] for k in TRAIN_KEYS}}
    svg, last_iter = sek_svg()
    text = TEMPLATE.read_text(encoding="utf-8").replace("{{EX}}", args.example)

    def image(match):
        data = base64.b64encode((FIG / match.group(1)).read_bytes()).decode()
        return f"data:image/jpeg;base64,{data}"

    text = re.sub(r"\{\{img:([^}]+)\}\}", image, text)
    text = re.sub(r"\{\{json:(\w+)\}\}",
                  lambda m: html.escape(json.dumps(blocks[m.group(1)], indent=2, ensure_ascii=False)), text)
    text = text.replace("{{sek_svg}}", svg).replace("{{sek_iter}}", f"{last_iter // 1000}k")
    left = re.findall(r"\{\{[^}]+\}\}", text)
    if left:
        raise SystemExit(f"unfilled placeholders: {sorted(set(left))}")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(args.out, len(text) // 1024, "KiB")


if __name__ == "__main__":
    main()
