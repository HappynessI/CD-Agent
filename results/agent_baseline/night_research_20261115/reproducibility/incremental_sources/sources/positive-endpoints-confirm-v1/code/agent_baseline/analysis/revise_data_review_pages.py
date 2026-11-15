"""Render a separate 19-image review from completed audits; no scoring/construction.

The original 17-page review, all candidate manifests and all GT stay unchanged.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from revise_t5_review import page
from revise_whu_official import write


def build(a):
    raise SystemExit('Historical WHU comparison renderer retired after XOR GT removal; retain the published embedded review pages. Use finalize_whu_gt for current references.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--whu',type=Path,required=True);build(p.parse_args())
