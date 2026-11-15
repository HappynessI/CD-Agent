"""Opt-in T3 semantic primitives; planner owns proposal choice and submission."""
import numpy as np
from .state import Mask, Instances, ToolError

NAMES = ('interpret_exemplar', 'rank_semantic_transition')


def schemas():
    from .tools import schema, arg
    from .exemplar_semantics import VOCAB
    label = arg('string', enum=list(VOCAB))
    return [schema('interpret_exemplar',
        'Independently inspect the visual example at each date. Returns before/after land-cover classes, possibly unknown, and visual evidence. Does not find changes or set the answer.',
        dict(mask=arg('string', description='Existing visual-example mask handle.')), ['mask']),
        schema('rank_semantic_transition',
        'Score an existing proposal set by mean two-date semantic probability for a directed before-to-after transition. Returns a new props handle, per-ID scores and ranked_ids in descending score order (ties by ID). Choose your own IDs from ranked_ids with select(set=returned props, ids=[...]); there is no automatic top-k, threshold, verification or answer selection. Semantic predictions may be wrong; unknown/same classes yield zero scores.',
        dict(props=arg('string'), before=label, after=label), ['props','before','after'])]


class SemanticBridgeTools:
    def interpret_exemplar(self, mask):
        from .exemplar_semantics import interpret
        from .analysis.io_retry import read_text
        import json
        if getattr(self, '_bridge_failed', False):
            raise ToolError('backend_error', 'Earlier semantic API request failed; inspect archived failure before resuming.')
        exemplar = self.get(mask, Mask).data
        if not exemplar.any(): raise ToolError('bad_args', 'Example mask is empty.')
        folder = self.directory / 'semantic_bridge' / mask.replace('#','_')
        cached = all((folder / time / 'answer.json').exists() for time in ('t1','t2'))
        prior = set(folder.glob('*/answer.json')) | set(folder.glob('*/failure_*.json'))
        prior_attempts = set(folder.glob('*/attempt_*.json'))
        try:
            result = interpret(self.rgb, exemplar, folder, self.bridge_seed)
        except Exception:
            self._bridge_failed = True
            raise
        finally:
            new = (set(folder.glob('*/answer.json')) | set(folder.glob('*/failure_*.json'))) - prior
            records = [json.loads(read_text(path)) for path in new]
            self.meta.update(cost_cny=sum(r['cost_cny'] for r in records),
                tokens={key:sum(r['tokens'].get(key,0) for r in records) for key in ('input','cached','output')},
                attempts=[json.loads(read_text(path)) for path in sorted(set(folder.glob('*/attempt_*.json')) - prior_attempts)])
        return dict(before=result['classes'][0], after=result['classes'][1],
            visual_evidence=result['evidence'], cached=cached)

    def rank_semantic_transition(self, props, before, after):
        from .exemplar_semantics import CLASSES
        item = self.get(props, Instances)
        reverse = {name:i for i,name in CLASSES.items()}
        scores = np.zeros(len(item.items))
        if before in reverse and after in reverse and before != after:
            probabilities = [self._vision('classify', time=t, classes=list(CLASSES.values())) for t in ('t1','t2')]
            # DynamicBackend.semantic prepends background to the requested six
            # classes; this is also the existing classify tool's public contract.
            if any(p.shape != (len(CLASSES)+1, *self.session.shape) for p in probabilities):
                raise ToolError('backend_error', 'Expected background plus six aligned semantic channels.')
            product = (probabilities[0][reverse[before]] * probabilities[1][reverse[after]]).ravel()
            scores = [float(product[r['idx']].mean()) for r in item.items]
        rows = [dict(row, score=float(score)) for row,score in zip(item.items,scores)]
        result = self._props(rows, item.time, (props,))
        result.update(before=before, after=after,
            ranked_ids=[row['id'] for row in sorted(rows,key=lambda row:(-row['score'],row['id']))],
            scores=[dict(id=row['id'], score=row['score']) for row in rows])
        return result
