"""Export a demo pool after review, for the ICL evaluation scripts.

  python export_demos.py --db verifier.db --set exp6k_demos_second --out demos_second.json
      drops rejected demos, keeps passed and not-yet-reviewed ones
  python export_demos.py ... --only-passed
      keeps passed demos only

Each entry carries the sample fields (label, rationale, pseudo_type, change_kind, area, source,
det_mean, inputs, ...), its stratum and the review status and note. Reviews are per sample,
so a demo reviewed in the 64 pool counts as reviewed in the 256 pool too. A summary goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import store


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--set', required=True, help='demo pool name')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--only-passed', action='store_true')
    args = p.parse_args()
    db = store.connect(args.db)
    set_id = store.set_id_of(db, args.set)
    rows = [dict(r) for r in db.execute(
        'SELECT s.*, i.stratum, i.weight, v.status AS review_status, v.note AS review_note, v.reviewed_at '
        'FROM sample_set_item i JOIN sample s ON s.id = i.sample_id LEFT JOIN sample_review v ON v.sample_id = s.id '
        'WHERE i.set_id = ? ORDER BY s.id', (set_id,))]
    keep = [r for r in rows if (r['review_status'] == 'pass' if args.only_passed else r['review_status'] != 'reject')]
    for r in keep:
        store.with_semantic(r)
        r['inputs'] = json.loads(r['inputs'])
        r['bbox'] = json.loads(r['bbox']) if r['bbox'] else None
        r.update(json.loads(r.pop('meta') or '{}'))
        r.pop('id')
    args.out.write_text(json.dumps(keep, ensure_ascii=False, indent=1) + '\n')
    status = Counter(r['review_status'] or 'todo' for r in rows)
    print(f"{args.set}: {len(rows)} demos (pass {status['pass']}, reject {status['reject']}, todo {status['todo']}) "
          f"-> {len(keep)} written to {args.out}; positives {sum(r['label'] == 1 for r in keep)}", file=sys.stderr)


if __name__ == '__main__':
    main()
