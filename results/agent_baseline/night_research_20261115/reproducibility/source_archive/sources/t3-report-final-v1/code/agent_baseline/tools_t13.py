"""Opt-in proposal, feature and semantic tools. No model imports at module load."""
from dataclasses import dataclass
import numpy as np
from .state import Instances, Mask, ToolError
from . import instances


@dataclass(frozen=True)
class Features:
    data: np.ndarray  # patch grid H,W,C; bilinear interpolation uses align_corners=True
    shape: tuple
    time: str


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.clip(np.divide(np.sum(a*b, axis=-1), norm, out=np.zeros_like(norm), where=norm > 0), -1, 1)


def pool_features(feature, indices):
    """Exact mean of an align_corners=True bilinearly resized patch grid.

    Accumulate interpolation weights instead of materialising a full H,W,768 map.
    """
    h, w = feature.shape
    gh, gw, dim = feature.data.shape
    idx = np.asarray(indices, np.int64)
    if not len(idx):
        raise ToolError('bad_args', 'Feature regions must be nonempty.')
    y = (idx // w) * ((gh-1) / max(h-1, 1))
    x = (idx % w) * ((gw-1) / max(w-1, 1))
    y0, x0 = y.astype(int), x.astype(int)
    y1, x1 = np.minimum(y0+1, gh-1), np.minimum(x0+1, gw-1)
    dy, dx = y-y0, x-x0
    weights = np.zeros(gh*gw)
    for yy, xx, value in ((y0,x0,(1-dy)*(1-dx)), (y0,x1,(1-dy)*dx),
                          (y1,x0,dy*(1-dx)), (y1,x1,dy*dx)):
        weights += np.bincount(yy*gw+xx, weights=value, minlength=gh*gw)
    return (weights / len(idx)).astype(np.float32) @ feature.data.reshape(-1, dim).astype(np.float32)


def pooled(feature, rows):
    return np.asarray([pool_features(feature, r['idx']) for r in rows], np.float32).reshape(len(rows), feature.data.shape[-1])


def tool_schemas(tool_hints='off'):
    from .tools import schema, arg, TIME, S
    props = arg('string', description='Existing props#k proposal set; IDs persist through select.')
    feat = arg('string', description='Existing feat#k DINOv2 feature handle.')
    schemas = [
        schema('segment_everything', 'Propose class-agnostic SAM masks at one date. Returns props handle, count and area statistics.', dict(time=TIME), ['time']),
        schema('extract_features', 'Extract a cached DINOv2 patch feature grid at one date. Returns feat handle and grid size.', dict(time=TIME), ['time']),
        schema('compare_features', 'Attach score = 1 - cosine of region-mean T1 and T2 DINOv2 features to each proposal. Returns new props handle and score distribution.', dict(props=props, feat_t1=feat, feat_t2=feat), ['props','feat_t1','feat_t2']),
        schema('classify', 'Mean SegEarth-OV class probabilities in each proposal at one date. Softmax vocabulary includes background and the supplied classes. Returns new props handle and per-ID probabilities.', dict(props=props, time=TIME, classes=arg('array', items=S, minItems=1, uniqueItems=True)), ['props','time','classes']),
        schema('segment_semantic', 'Dense SegEarth-OV segmentation for one class at one date, against background. Returns a mask; does not change current answer.', dict(time=TIME, text=S), ['time','text']),
        schema('match_exemplar', 'Rank proposals by cosine similarity to a mask handle or [x,y] point. change uses T2 minus T1 region-mean DINOv2 features; t1/t2 uses appearance. Returns new props handle and per-ID similarities.', dict(props=props, exemplar=arg('object', properties=dict(mask=S, point=arg('array', items=arg('number'), minItems=2, maxItems=2)), additionalProperties=False), mode=arg('string', enum=['change','t1','t2'], default='change')), ['props','exemplar'])]
    if tool_hints != 'off':
        from .gate4 import TOOL_HINTS
        for entry in schemas:
            hint = TOOL_HINTS[tool_hints].get(entry['function']['name'])
            if hint:
                entry['function']['description'] += ' ' + hint
    return schemas


NAMES = ('segment_everything','extract_features','compare_features','classify','segment_semantic','match_exemplar')


def enabled_names(value):
    if value is True:
        return NAMES
    if not value:
        return ()
    names = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if not set(names) <= set(NAMES):
        raise ValueError('Unknown T1.3 tool name')
    return names


def extend_schemas(schemas, enabled, tool_hints='off'):
    import copy
    names = enabled_names(enabled)
    if not names:
        return schemas
    for entry in schemas:
        function = entry['function']
        if function['name'] in ('select','to_mask','inspect','view','add_region'):
            function['description'] += ' Proposal handles props#k use the same operations as inst#k; select by ids or where="score >= threshold", then to_mask(sets=[handle]).'
    return schemas + [copy.deepcopy(s) for s in tool_schemas(tool_hints) if s['function']['name'] in names]


class ProposalTools:
    def _props(self, rows, time, parents=()):
        key = self.session.add_instances(rows, time, parents)
        item = self.session.handles.pop(key)
        new = 'props#' + key.split('#')[1]
        self.session.handles[new] = item
        scores = [float(r['score']) for r in rows if r.get('score') is not None]
        return dict(props=new, n=len(rows), area=int(instances.to_mask(rows, self.session.shape).sum()),
                    score_range=[min(scores), max(scores)] if scores else None)

    def _vision(self, operation, **kwargs):
        return self._backend(self.vision_backend, operation, **kwargs)

    def segment_everything(self, time, region=None):
        from .gate5 import resolve_region, clip_instances, clip_result
        region_data = resolve_region(self, region)
        masks = self._vision('segment_everything', time=time)
        rows = instances.from_dense(masks, np.ones(len(masks)))
        if region is not None:
            rows, stats = clip_instances(rows, region, region_data)
        result = self._props(rows, time, (region,) if region is not None else ())
        return clip_result(self, result, stats) if region is not None else result

    def extract_features(self, time):
        data = np.array(self._vision('extract_features', time=time), copy=True)
        data.flags.writeable = False
        key = self.session._put('feat', Features(data, self.session.shape, time))
        return dict(feat=key, time=time, grid_shape=list(data.shape))

    def _feature(self, time):
        for key, value in self.session.handles.items():
            if isinstance(value, Features) and value.time == time and (self.allowed is None or key in self.allowed):
                return key, value
        key = self.extract_features(time)['feat']
        return key, self.session.handles[key]

    def compare_features(self, props, feat_t1, feat_t2):
        item = self.get(props, Instances)
        a, b = self.get(feat_t1, Features), self.get(feat_t2, Features)
        if (a.time, b.time) != ('t1','t2'):
            raise ToolError('bad_args', 'Features must be ordered T1 then T2.')
        scores = 1 - cosine(pooled(a,item.items), pooled(b,item.items))
        result = self._props([dict(r, score=float(s)) for r,s in zip(item.items,scores)], item.time, (props,feat_t1,feat_t2))
        if self.gate6:
            levels = (0, 10, 25, 50, 75, 90, 100)
            result["score_quantiles"] = dict(zip(("min", "p10", "p25", "p50", "p75", "p90", "max"),
                np.percentile(scores, levels).tolist() if len(scores) else [None] * 7))
        return result

    def classify(self, props, time, classes):
        if any(not c.strip() or c.strip().lower() == 'background' for c in classes):
            raise ToolError('bad_args', 'Supply nonempty class names; background is included automatically.')
        item = self.get(props, Instances)
        probs = self._vision('classify', time=time, classes=classes)
        flat = probs.reshape(len(classes)+1, -1)
        values = [flat[1:,r['idx']].mean(axis=1) for r in item.items]
        rows = [dict(r, probabilities={c:float(p) for c,p in zip(classes,v)}) for r,v in zip(item.items,values)]
        result = self._props(rows, item.time, (props,))
        result.update(classes=classes, probabilities=[dict(id=r['id'], probabilities=r['probabilities']) for r in rows])
        return result

    def segment_semantic(self, time, text):
        if not text.strip() or text.strip().lower() == 'background':
            raise ToolError('bad_args', 'Supply a foreground class name.')
        probs = self._vision('classify', time=time, classes=[text])
        return self._mask(probs.argmax(axis=0) == 1, set_current=False)

    def match_exemplar(self, props, exemplar, mode='change', negative=None):
        if (getattr(self.session, 'external_examples', ()) and
                ('example' in exemplar or str(exemplar.get('mask', '')).startswith('exmask#') or negative is not None)):
            from .examples_t3x import match_external
            return match_external(self, props, exemplar, mode, negative)
        item = self.get(props, Instances)
        if set(exemplar) == {'mask'}:
            idx = np.flatnonzero(self.get(exemplar['mask'], Mask).data)
            parents = (props, exemplar['mask'])
        elif set(exemplar) == {'point'}:
            x,y = exemplar['point']
            h,w = self.session.shape
            if not (0 <= x < w and 0 <= y < h):
                raise ToolError('bad_args', 'Point must be inside the image.')
            idx = np.asarray([int(y)*w+int(x)])
            parents = (props,)
        else:
            raise ToolError('bad_args', 'Exemplar needs exactly one mask or point.')
        def vectors(time):
            key, feature = self._feature(time)
            return key, pooled(feature,item.items), pool_features(feature,idx)
        if mode == 'change':
            k1,a,ea = vectors('t1')
            k2,b,eb = vectors('t2')
            candidate, reference = b-a, eb-ea
            parents += (k1,k2)
        else:
            key,candidate,reference = vectors(mode)
            parents += (key,)
        scores = cosine(candidate, reference)
        rows = [dict(r, score=float(s)) for r,s in zip(item.items,scores)]
        result = self._props(rows,item.time,parents)
        result['similarities'] = [dict(id=r['id'], score=r['score']) for r in sorted(rows,key=lambda r:(-r['score'],r['id']))]
        result['mode'] = mode
        return result
