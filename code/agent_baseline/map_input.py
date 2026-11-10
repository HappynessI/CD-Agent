"""Opt-in T5.4 map/imagery input. Only the supplied old map enters Session."""
import json
import numpy as np
from .state import Mask, ToolError

INSTRUCTION = ('Update the old building map using the new image. '
               'Output a binary change mask of newly built and demolished buildings.')
MAP_NOTE = ('T1 is the old building map: white pixels are buildings and black pixels are background. '
            'T2 is the new optical image of the same place. map#0 contains the old building mask '
            'and works wherever a mask handle is accepted. It is an input map, not the current answer. '
            'Optical segmentation and feature tools accept T2 only.')
COUNTS = ['building presence in the new image differs from the old map: newly built or demolished buildings']
NOT_COUNTS = ['building presence agrees between the old map and the new image',
              'appearance changes with unchanged building presence']
PSEUDO_TYPES = ['new_building', 'demolished', 'no_change', 'uncertain']


def render_map(buildings):
    """Full building footprints, white on black, with three identical RGB channels."""
    buildings = np.asarray(buildings, bool)
    if buildings.ndim != 2:
        raise ValueError('Old building map must be a 2D mask')
    return np.repeat((buildings.astype(np.uint8) * 255)[..., None], 3, axis=2)


def map_mask(raster):
    raster = np.asarray(raster)
    if raster.ndim == 3 and raster.shape[2] == 3:
        if not np.all(raster == raster[..., :1]):
            raise ValueError('Old map must have identical RGB channels')
        raster = raster[..., 0]
    if raster.ndim != 2 or not np.isin(raster, [0, 255]).all():
        raise ValueError('Old map raster must use black=0 and white=255')
    return raster == 255


def attach_map(session, raster):
    if (session.target, session.change_direction, session.scope) != ('building', 'any', None):
        raise ValueError('T5.4 requires building, both directions and the map scope')
    if session.initial_mask is not None:
        raise ValueError('T5.4 takes an old map, without an initial change mask')
    session.old_map = session.add_mask(map_mask(raster), handle='map#0')


def check_task(row):
    if (row.get('input_type') != 'map_image' or row['target'] != 'building'
            or row['change_direction'] != 'any' or row.get('scope') is not None
            or row.get('init_mask') or row.get('region_mask') or row.get('exemplar_mask')):
        raise ValueError('T5.4 requires input_type=map_image, building, any direction and an old map as image_t1')


def check_tool(name, args):
    optical = {'segment_text', 'segment_prompt', 'refine_region', 'segment_everything',
               'extract_features', 'classify', 'segment_semantic'}
    if (name in optical and args.get('time') == 't1') or name in {'compare_features', 'match_exemplar'}:
        raise ToolError('unsupported_input', 'T1 is a building map. Use map#0 for its footprints; '
                        'optical segmentation and feature tools accept T2 only.')


def describe(session):
    item = session.get(session.old_map, Mask)
    return MAP_NOTE + f' map#0: area={int(item.data.sum())} pixels; component_count={len(item.components)}.'


def schemas_for_map(schemas):
    """Modify only this session's schema copy, never the frozen global schemas."""
    for schema in schemas:
        fun = schema['function']
        fun['description'] += ' map#0 is the old building map and is usable as a mask handle.'
        if fun['name'] in {'segment_text', 'segment_prompt', 'refine_region', 'segment_everything',
                           'extract_features', 'classify', 'segment_semantic'}:
            fun['description'] += ' T1 is a map; optical tools accept T2 only.'
        if fun['name'] == 'ask':
            fun['description'] = ('Check whether building presence in T2 differs from the old map T1 '
                                  'inside each mask component. ') + fun['description']
    return schemas


def verifier_settings():
    question = ('T1 is an old building map (white buildings, black background); T2 is a new optical image. '
                'The red contour marks the same region in both views. Does building presence inside '
                'the contour differ between the old map and the new image, through construction or demolition? '
                'Reply with JSON only: {"change": "yes" or "no", "type": one of '
                + json.dumps(PSEUDO_TYPES) + ', "reason": one short sentence}')
    return dict(SCOPE_CARD=dict(counts=COUNTS[:], not_counts=NOT_COUNTS[:]),
                PSEUDO_TYPES=PSEUDO_TYPES[:],
                SYSTEM=('You verify building map updates. T1 is an old binary building map, '
                        'and T2 is new optical imagery of the same place. Judge footprint presence '
                        'and geometry, including newly built and demolished buildings. '
                        'An unchanged footprint counts as no_change. The map contains no roof appearance.'),
                VERIFY=question,
                CORRESPONDENCE=('The first two images are close-ups and the last two show the same '
                                'place with more surroundings. ' + question))
