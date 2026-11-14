"""Area constraint from the task list (T4.1 v2): reported at submission, never removed; absent field = no text."""
from pathlib import Path
import unittest
import numpy as np
from agent_baseline.state import Session
from agent_baseline.tools import Toolbox
from .helpers import tempdir, ask


def rectangle(shape, box):
    out = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = True
    return out


class AreaConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempdir()
        self.addCleanup(self.tmp.cleanup)

    def box(self, initial, **kwargs):
        shape = initial.shape
        rgb = [np.zeros((*shape, 3), np.uint8)] * 2
        box = Toolbox(rgb, Path(self.tmp.name), ask=ask, session=Session(shape, initial, **kwargs), gate3=True)
        box.ask_backend = lambda pairs, question: ([dict(answer='yes', type='new_building', p_yes=.9) for _ in pairs], {})
        return box

    def test_small_regions_reported_not_removed(self):
        shape = (60, 60)
        # 16 px region and 400 px region; limit 25 m^2 at 0.5 m/px = 100 px.
        initial = rectangle(shape, (1, 1, 5, 5)) | rectangle(shape, (30, 30, 50, 50))
        box = self.box(initial, area_min_m2=25, gsd_m=0.5)
        self.assertEqual(box.session.area_min_px, 100)
        report = box.pre_submit('mask#0', 2, stage=True).public['report']
        self.assertIn('Area constraint (regions must be larger than 25 m^2): 1 connected regions are not larger, 16 pixels in total', report)
        np.testing.assert_array_equal(box.session.get('mask#0').data, initial)

    def test_no_area_field_no_text(self):
        shape = (60, 60)
        initial = rectangle(shape, (1, 1, 5, 5))
        box = self.box(initial)
        self.assertIsNone(box.session.area_min_px)
        self.assertEqual(box.constraint_report(box.session.get('mask#0')), [])
        self.assertNotIn('Area constraint', box.pre_submit('mask#0', 2, stage=True).public['report'])


if __name__ == '__main__':
    unittest.main()
