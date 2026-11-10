import sys
import threading
import types
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock
import numpy as np
from agent_baseline.backends import SharedPrompter, PromptWorker
from agent_baseline.workers.sam_prompt_worker import PromptServer, handler


class FakePredictor:
    """Deterministic stand-in for SamPredictor: masks depend on the image content and the prompt."""
    def __init__(self):
        self.encodes = 0
        self.reset_image()

    def reset_image(self):
        self.features = self.original_size = self.input_size = None
        self.is_image_set = False

    def set_image(self, rgb):
        self.encodes += 1
        self.features = float(rgb.astype(np.int64).sum() % 997)
        self.original_size = self.input_size = rgb.shape[:2]
        self.is_image_set = True

    def predict(self, box=None, point_coords=None, point_labels=None, multimask_output=False):
        h, w = self.original_size
        out = np.zeros((3, h, w), bool)
        if box is not None:
            x0, y0, x1, y1 = (int(v) for v in box)
            out[0, y0:y1, x0:x1] = True
            quality = np.array([0.5 + self.features / 4000, 0, 0])
            return out[:1], quality[:1], None
        for k, (x, y) in enumerate(np.asarray(point_coords, int)):
            out[:, max(0, y - 1 - k):y + 2, max(0, x - 1):x + 2 + k] = True
        quality = np.array([0.2, 0.7 + self.features / 5000, 0.4])
        return out, quality, None


class FakeCached:
    def box(self, split, name, t, box):
        m = np.zeros((8, 8), bool); m[t:t + 3, :len(name)] = True
        return m, 0.25 + t

    def points(self, split, name, t, pts, labels):
        m = np.zeros((8, 8), bool); m[:, t] = True
        return m, 0.75


class PromptWorkerTest(unittest.TestCase):
    def setUp(self):
        self.pred = FakePredictor()
        self.server = PromptServer(image_factory=lambda: self.pred, cached_factory=FakeCached, cache_size=2)
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.server))
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.url = f'http://127.0.0.1:{self.http.server_address[1]}'
        rng = np.random.default_rng(0)
        self.rgb = [rng.integers(0, 255, (16, 20, 3), dtype=np.uint8) for _ in range(2)]

    def local(self):
        fake = types.SimpleNamespace(build=lambda device: FakePredictor(), Prompter=FakeCached)
        return mock.patch.dict(sys.modules, {'sam_tools': fake})

    def test_image_path_matches_in_process(self):
        remote = PromptWorker(self.url).for_image('val', 'a.png', rgb=self.rgb)
        with self.local():
            local = SharedPrompter().for_image('val', 'a.png', rgb=self.rgb)
            for time in ('t1', 't2'):
                for kwargs in (dict(box=[2, 3, 9, 11]), dict(points=[[4, 5], [10, 7]], point_labels=[1, 0])):
                    m1, q1, meta1 = local(time, **kwargs)
                    m2, q2, meta2 = remote(time, **kwargs)
                    self.assertTrue(np.array_equal(m1, m2))
                    self.assertEqual(q1, q2)
                    self.assertEqual(meta2['backend'], 'sam_prompt')

    def test_cached_path_matches_in_process(self):
        remote = PromptWorker(self.url).for_image('val', 'val_7.png')
        local = SharedPrompter(factory=FakeCached).for_image('val', 'val_7.png')
        for time in ('t1', 't2'):
            for kwargs in (dict(box=[1, 1, 4, 4]), dict(points=[[2, 2]], point_labels=[1])):
                m1, q1, _ = local(time, **kwargs)
                m2, q2, _ = remote(time, **kwargs)
                self.assertTrue(np.array_equal(m1, m2))
                self.assertEqual(q1, q2)

    def test_embeddings_reused_across_calls_and_evicted(self):
        call = PromptWorker(self.url).for_image('val', 'a.png', rgb=self.rgb)
        for _ in range(3):
            call('t1', box=[0, 0, 5, 5])
        self.assertEqual(self.pred.encodes, 1)
        call('t2', box=[0, 0, 5, 5])
        call('t1', box=[0, 0, 5, 5])  # restored from the LRU, not re-encoded
        self.assertEqual(self.pred.encodes, 2)
        other = PromptWorker(self.url).for_image('val', 'b.png', rgb=[self.rgb[0][::-1].copy(), self.rgb[1]])
        other('t1', box=[0, 0, 5, 5])  # third image evicts the oldest (cache_size=2)
        call('t2', box=[0, 0, 5, 5])
        self.assertEqual(self.pred.encodes, 4)


if __name__ == '__main__':
    unittest.main()
