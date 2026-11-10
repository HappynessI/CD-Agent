"""Task data API and image rendering. No changes to the existing API contracts."""
from __future__ import annotations

import io
import json
from contextlib import closing
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from scipy import ndimage as ndi

import task_data_store as store
from task_data_import import read_image

router = APIRouter(prefix='/api/task-data')


class ReviewInput(BaseModel):
    verdict: Literal['reasonable', 'unreasonable'] | None = None
    note: str = ''
    annotator: str = Field(min_length=1)


class LabelInput(BaseModel):
    pass_number: Literal[1, 2] = 1
    color: str
    subtype: str
    annotator: str = Field(min_length=1)


class MappingInput(BaseModel):
    t1: list[int]
    t2: list[int]
    annotator: str = Field(min_length=1)


class UndoInput(BaseModel):
    kind: Literal['review', 'component']
    annotator: str = Field(min_length=1)
    pass_number: Literal[1, 2] = 1
    target: str | None = None


def actor(value):
    if not value.strip():
        raise ValueError('请填写标注人')
    return value.strip()


def write(action):
    try:
        with closing(store.connect()) as db, db:
            return action(db)
    except KeyError:
        raise HTTPException(404, '记录不存在')
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get('/catalog')
def catalog():
    with closing(store.connect()) as db:
        return dict(metadata=store.metadata(db), examples=store.examples(db),
                    mappings=store.mappings(db), colors=store.COLORS, subtypes=store.SUBTYPES,
                    legend=[dict(code=i, name=name, rgb=store.PALETTE[i])
                            for i, name in enumerate(store.CLASS_NAMES)])


@router.get('/components')
def components(pass_number: Literal[1, 2] = 1, image: str = '', direction: str = ''):
    with closing(store.connect()) as db:
        records = store.components(db, pass_number, image, direction)
        return dict(items=records, total=len(records), labeled=sum(bool(r['labeled_at']) for r in records))


@router.post('/reviews/{ident}')
def review(ident: str, body: ReviewInput):
    return write(lambda db: (store.save_review(db, ident, body.verdict, body.note, actor(body.annotator)),
                             dict(ok=True))[1])


@router.post('/labels/{ident}')
def label(ident: str, body: LabelInput):
    return write(lambda db: (store.save_label(db, ident, body.pass_number, body.color, body.subtype,
                                             actor(body.annotator)), dict(ok=True))[1])


@router.post('/mappings/{purpose}')
def mapping(purpose: str, body: MappingInput):
    return write(lambda db: (store.save_mapping(db, purpose, dict(t1=body.t1, t2=body.t2),
                                               actor(body.annotator)), dict(ok=True))[1])


@router.post('/undo')
def undo(body: UndoInput):
    return write(lambda db: dict(target=store.undo(db, body.kind, actor(body.annotator),
                                                  body.pass_number, body.target)))


@router.get('/export/{filename}')
def export(filename: Literal['task_review.json', 'second_component_labels.json']):
    with closing(store.connect()) as db:
        documents = store.exports(db)
        value = documents[0 if filename == 'task_review.json' else 1]
        return Response(json.dumps(value, ensure_ascii=False, indent=2), media_type='application/json',
                        headers={'Content-Disposition': f'attachment; filename="{filename}"'})


def outline(mask):
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)


def paint(image, mask, fill=False):
    rgb = np.array(image.convert('RGB'))
    if mask.shape != rgb.shape[:2]:
        raise ValueError('图像与真值尺寸不一致')
    if fill:
        rgb[mask] = (rgb[mask] * .58 + np.array([255, 190, 0]) * .42).astype(np.uint8)
    rgb[outline(mask)] = [255, 36, 36]
    return Image.fromarray(rgb)


def rendered(db, kind, ident, key):
    table = 'task_example' if kind == 'example' else 'task_component'
    row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (ident,)).fetchone()
    if row is None:
        raise KeyError(ident)
    p = json.loads(row['payload'])
    assets = p['assets']
    if kind == 'example':
        if key in ('t1', 't2'):
            image = read_image(assets[key], 'RGB')
        elif key in ('s1', 's2') and p['semantic_available']:
            label = np.asarray(read_image(assets[key]))
            image = Image.fromarray(np.array(store.PALETTE, dtype=np.uint8)[label])
        elif key == 'overlay':
            mask = np.asarray(read_image(assets['gt'], 'L')) > 0
            image = paint(read_image(assets['t2'], 'RGB'), mask, fill=True)
        else:
            raise KeyError(key)
    else:
        if key not in ('t1', 't2', 'thumb'):
            raise KeyError(key)
        mask = np.zeros((assets['size'][1], assets['size'][0]), bool)
        for y, x0, x1 in assets['runs']:
            mask[y, x0:x1] = True
        # Both periods have the exact same component contour and crop.
        source = 't1' if key == 't1' or (key == 'thumb' and p['direction'] == 'demolished') else 't2'
        image = paint(read_image(assets[source], 'RGB'), mask)
        if key == 'thumb':
            ImageDraw.Draw(image).rectangle(p['bbox'], outline='#ff2424', width=3)
            image.thumbnail((256, 256))
        else:
            image = image.crop(assets['crop'])
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


@router.get('/image/{kind}/{ident}/{key}')
def image(kind: Literal['example', 'component'], ident: str, key: str):
    try:
        with closing(store.connect()) as db:
            return Response(rendered(db, kind, ident, key), media_type='image/png')
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, '图片不存在或没有该期语义标签')
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f'图片读取失败（已重试）：{type(exc).__name__}')
