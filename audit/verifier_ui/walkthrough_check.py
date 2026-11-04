"""Exercise real offline assets, read-only routes and the embedded JavaScript."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import os
import socket
from pathlib import Path
import subprocess
import tempfile

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.active = not dict(attrs).get('src')

    def handle_endtag(self, tag):
        if tag == 'script':
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--http', action='store_true', help='Also exercise ASGI through TestClient; requires socketpair permission')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='walkthrough-check-') as temp:
        absent_db = Path(temp) / 'must-not-be-created.db'
        os.environ['VERIFIER_DB'] = str(absent_db)
        os.environ.pop('WALKTHROUGH_DIR', None)
        import app
        empty = app.walkthrough_catalog()
        assert not empty['configured'] and not any(s['ready'] for s in empty['scopes'])
        assert Path(app.index().path).is_file()
        try:
            app.walkthrough_image('second', 'val', 'missing')
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError('Unset directory must fail with 404')
        app.WALKTHROUGH_DIR = args.assets.resolve()
        catalog = app.walkthrough_catalog()
        ready = [s for s in catalog['scopes'] if s['ready']]
        assert ready, 'Need actual built images'
        images = assets = units = 0
        for scope in ready:
            ds, split, index = scope['dataset'], scope['split'], scope['index']
            assert len(index['images']) == index['generated']
            assert set(index['typical']) <= set(index['images'])
            statistics = json.loads((args.assets/ds/split/'stats.json').read_text())
            assert index['statistics'] == {k:statistics[k] for k in index['statistics']}
            assert bool(statistics['averages'] is not None) == statistics['complete']
            assert index['pool'] == statistics['typical'] + statistics['random']
            assert len(set(index['pool'])) == index['pool_size'] <= 10
            assert set(index['images']) <= set(index['pool'])
            assert not set(statistics['typical']) & set(statistics['random'])
            assert index['png_bytes'] <= 1_000_000_000 // 7
            assert statistics['analyzed'] == len(statistics['records'])
            for step in range(7):
                records = statistics['records']
                aggregate = statistics['aggregates'][step]
                assert aggregate['mean_count'] == sum(r['steps'][step]['count'] for r in records)/len(records)
                assert aggregate['hits2'] == sum(r['steps'][step]['hits2'] for r in records)
                assert aggregate['objects2'] == sum(r['steps'][step]['objects2'] for r in records)
            stats_by_name = {r['image']:r for r in statistics['records']}
            for name in index['images']:
                stem = Path(name).stem
                d = app.walkthrough_image(ds, split, stem)
                assert d['image'] == name and len(d['steps']) == 7
                assert [step['stats'] for step in d['steps']] == stats_by_name[name]['steps']
                assert d['features'] == stats_by_name[name]['features']
                for file, background in d['backgrounds'].items():
                    assert background['file'] in ('t1.png','t2.png')
                    x0,y0,x1,y1 = background['box']
                    assert 0<=x0<x1<=d['shape'][1] and 0<=y0<y1<=d['shape'][0]
                    with Image.open(args.assets/ds/split/stem/file) as overlay:
                        assert overlay.mode=='RGBA' and overlay.size==(x1-x0,y1-y0)
                        assert set(np.unique(np.array(overlay)[...,3])) <= {0,255}
                assert d['verification']['status'] == 'exact'
                assert len(d['steps'][6]['candidates']) == d['verification']['units']
                files = set(d['assets'])
                references = []
                for step in d['steps']:
                    references += [f['file'] for f in step['frames']]
                    references += [s['file'] for s in step.get('sources', [])]
                    for example in step['examples'] + step.get('candidates', []):
                        references += [f['file'] for f in example['frames']]
                assert set(references) <= files
                for file in files:
                    relative = f'{ds}/{split}/{stem}/{file}'
                    path = Path(app.walkthrough_asset(relative).path)
                    assert path.stat().st_size == d['asset_sizes'][file]
                    with Image.open(path) as im:
                        assert im.format == 'PNG' and min(im.size) > 0
                        im.load()
                    assets += 1
                # Reconstruct the click map independently from the frozen packed masks.
                with np.load(d['provenance']['frozen'], allow_pickle=False) as frozen:
                    image_id = frozen['images'].tolist().index(name)
                    rows = np.flatnonzero(frozen['image_id'] == image_id)
                    expected = np.zeros(d['shape'], np.uint32)
                    assert frozen['id'][rows].tolist() == [u['id'] for u in d['steps'][6]['candidates']]
                    boxes, offsets, packed = frozen['bbox'], frozen['mask_offsets'], frozen['mask_data']
                    for ordinal, j in enumerate(rows, 1):
                        x0, y0, x1, y1 = boxes[j]
                        mask = np.unpackbits(packed[offsets[j]:offsets[j+1]], count=int((x1-x0)*(y1-y0)))
                        expected[y0:y1, x0:x1][mask.reshape(y1-y0, x1-x0).astype(bool)] = ordinal
                with Image.open(args.assets / ds / split / stem / 'pick.png') as im:
                    pixels = np.array(im, dtype=np.uint32)
                actual = pixels[..., 0] + 256*pixels[..., 1] + 65536*pixels[..., 2]
                assert np.array_equal(actual, expected), 'Click map differs from frozen candidate pixels'
                images += 1
                units += d['verification']['units']
        for relative in ('../batch54_ui/TASK_candidates_ui.md', '../../etc/passwd', 'missing.png'):
            try:
                app.walkthrough_asset(relative)
            except HTTPException as exc:
                assert exc.status_code == 404
            else:
                raise AssertionError(f'Invalid asset was accepted: {relative}')
        if args.http:
            # Fail at the actual restriction instead of hanging the asyncio wakeup thread.
            pair = socket.socketpair()
            for sock in pair:
                sock.close()
            with TestClient(app.app) as client:
                assert client.get('/').status_code == 200
                assert client.get('/api/walkthrough').json() == catalog
                for scope in ready:
                    ds, split = scope['dataset'], scope['split']
                    for name in scope['index']['images']:
                        stem = Path(name).stem
                        assert client.get(f'/api/walkthrough/{ds}/{split}/{stem}').json() == app.walkthrough_image(ds, split, stem)
                        response = client.get(f'/walkthrough-assets/{ds}/{split}/{stem}/final.png')
                        assert response.status_code == 200 and response.headers['content-type'] == 'image/png'
                        assert response.content.startswith(b'\x89PNG')
                assert client.get('/api/walkthrough/whu/val/missing').status_code == 404
        assert not absent_db.exists(), 'Candidate routes opened the verifier database'
        scripts = Scripts()
        scripts.feed((Path(__file__).parent / 'static/index.html').read_text())
        path = Path(temp) / 'index.js'
        path.write_text('\n'.join(scripts.parts))
        subprocess.run(['node', '--check', str(path)], check=True)
        for name in ('app.py', 'walkthrough_build.py', 'walkthrough_check.py'):
            path = Path(__file__).parent / name
            compile(path.read_text(), str(path), 'exec')
        print(json.dumps(dict(status='passed', scopes=len(ready), images=images, pngs=assets,
                              verified_units=units, database_created=False, http='passed' if args.http else 'not requested', javascript='node --check passed'), ensure_ascii=False))


if __name__ == '__main__':
    main()
