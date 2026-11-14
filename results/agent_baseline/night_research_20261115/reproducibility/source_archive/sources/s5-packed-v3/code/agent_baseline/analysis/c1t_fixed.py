"""Appeared SECOND SAM3 IMC intersected with T1 vegetation/bare ground."""
import numpy as np
from common import components
from .t12_fixed import semantic


def source_mask(backend):
    # Separate binary segment_semantic calls, then union; identical to T1.2 probe vocabulary.
    return semantic(backend, 't1', 'low vegetation') | semantic(backend, 't1', 'non-vegetated ground')


def intersect(directed, source, mode='pixel', overlap=.5):
    directed, source = np.asarray(directed, bool), np.asarray(source, bool)
    if directed.shape != source.shape:
        raise ValueError('Directed and source masks must have matching shapes')
    if mode == 'none':
        return directed.copy()
    if mode == 'pixel':
        return directed & source
    if mode != 'component' or not 0 < overlap <= 1:
        raise ValueError('Expected pixel/none/component and overlap in (0,1]')
    labels, comps = components(directed)
    ids = [c['id'] for c in comps if (source & (labels == c['id'])).sum()/c['area'] >= overlap]
    return directed & np.isin(labels, ids)


if __name__ == '__main__':
    from .t12_fixed import main
    main('c1t')
