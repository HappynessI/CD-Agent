"""Render anonymous review contacts from already archived inputs; Slurm only."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pack', type=Path)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Render on a Slurm compute node')
    for case in sorted(args.pack.glob('case_*')):
        # The extra-context condition supplies original RGB for an audit;
        # do not read task identities, GT labels, predictions or mapping key.
        context = next(v for v in ('A', 'B') if len(json.loads(
            (case/v/'t1/input.json').read_text())['images']) > 2)
        files = [
            ('support close', case/'support_close.png'),
            ('T1 support close', case/context/'t1/input_00.png'),
            ('T2 support close', case/context/'t2/input_00.png'),
            ('support wide', case/'support_wide.png'),
            ('T1 support wide', case/context/'t1/input_01.png'),
            ('T2 support wide', case/context/'t2/input_01.png'),
            ('white included / black excluded', None),
            ('T1 original close', case/context/'t1/input_02.png'),
            ('T2 original close', case/context/'t1/input_04.png'),
            (case.name, None),
            ('T1 original wide', case/context/'t1/input_03.png'),
            ('T2 original wide', case/context/'t1/input_05.png'),
        ]
        panel = Image.new('RGB', (1536, 2160), 'white')
        draw = ImageDraw.Draw(panel)
        for i, (label, path) in enumerate(files):
            x, y = (i % 3)*512, (i // 3)*540
            draw.text((x+6, y+6), label, fill='black')
            if path:
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    image.thumbnail((512, 512))
                    panel.paste(image, (x, y+28))
        panel.save(case/'review_contact.jpg', quality=95)
        print(case.name, flush=True)


if __name__ == '__main__':
    main()
