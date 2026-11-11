"""The ported system figure must reproduce the hand-edited G1 figures byte for byte."""
from pathlib import Path
import re
import unittest

from agent_baseline.site_svg import agent_figure, route_figure

G1 = Path(__file__).resolve().parents[3] / 'results' / 'agent_baseline' / 'demo_g1'


def figure_of(name):
    return re.search(r'<svg class="sys".*?</svg>', (G1 / name).read_text(encoding='utf-8'), re.S).group(0)


@unittest.skipUnless((G1 / 'index.html').exists(), 'G1 pages are not in this checkout')
class G1FigureTest(unittest.TestCase):
    def test_entry_figure(self):
        self.assertEqual(agent_figure(), figure_of('index.html'))

    def test_session_figures(self):
        for name in ('T1.1xC0_val_27_showcase.html', 'T5.1_val_64_showcase.html'):
            ref = figure_of(name)
            counts = {m.group(1): int(m.group(2) or 0) for m in re.finditer(
                r'<text class="mono" x="\d+" y="\d+" text-anchor="start">([a-z_]+)</text>(?:<text class="mono" x="\d+" y="\d+" text-anchor="end">×(\d+)</text>)?', ref)}
            grab = lambda pattern: re.search(pattern, ref).group(1)
            out = agent_figure(inputs=re.findall(r'class="chiptext"[^>]*>([^<]*)', ref), input_label=grab(r'x="240" y="420" text-anchor="middle">([^<]*)'),
                               counts=counts, answers=grab(r'答案数：(\d+)'), holds=grab(r'暂停提交 (\d+) 次'), ending=grab(r'结束方式：([^<]*)'))
            self.assertEqual(out, ref, name)


class LaterFigureTest(unittest.TestCase):
    def test_switch_group_and_notes_grow_the_frame(self):
        base = agent_figure()
        more = agent_figure('G3', switch_tools={'segment_everything'}, harness_notes=['a', 'b', 'c'])
        height = lambda svg: int(re.search(r'viewBox="0 0 1540 (\d+)"', svg).group(1))
        self.assertGreater(height(more), height(base))
        self.assertIn('segment_everything', more)
        self.assertNotIn('segment_semantic', more)

    def test_route_figure_with_and_without_rule(self):
        fixed = route_figure('t', ['T1', 'T2'], [('a', ['segment_text'], ['x'])])
        rule = route_figure('t', ['T1', 'T2'], [('a', ['segment_text'], ['x'])], rule=dict(decide=['no']))
        self.assertNotIn('验证器 Verifier', fixed)
        self.assertIn('验证器 Verifier', rule)


if __name__ == '__main__':
    unittest.main()
