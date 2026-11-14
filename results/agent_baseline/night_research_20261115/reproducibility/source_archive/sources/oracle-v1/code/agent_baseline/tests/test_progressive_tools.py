import tempfile
import unittest
from pathlib import Path
from agent_baseline.tools import Toolbox, session_schemas, disclosed_schemas, SKILL_SCHEMA
from agent_baseline.planner import Planner
from agent_baseline.recorder import Recorder
from agent_baseline.loop_json import run_session
from agent_baseline.tests.helpers import rgb, mask, prompt, call, response, ScriptClient


def names(schemas):
    return {s['function']['name'] for s in schemas}


class ProgressiveToolsTests(unittest.TestCase):
    def test_core_and_skill_retained_then_edits_appear(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, segment_pair=True, answer_state=True)
            schemas = [*session_schemas(box.session, segment_pair=True, answer_state=True), SKILL_SCHEMA]
            visible = names(disclosed_schemas(schemas, box.session))
            self.assertTrue({'view', 'inspect', 'segment_text', 'segment_text_pair', 'segment_prompt',
                             'ask', 'submit', 'set_answer', 'to_mask', 'match_instances', 'select', 'load_skill'} <= visible)
            self.assertEqual(names(schemas) - visible, {'combine', 'morph', 'remove_region', 'add_region', 'refine_region'})
            box.session.add_mask(mask())
            self.assertEqual(names(disclosed_schemas(schemas, box.session)), names(schemas))

    def test_initial_mask_keeps_all_tools(self):
        with tempfile.TemporaryDirectory() as d:
            box = Toolbox(rgb(), d, init_mask=mask())
            schemas = session_schemas(box.session)
            self.assertIs(disclosed_schemas(schemas, box.session), schemas)

    def test_live_loop_updates_schema_after_candidate_and_default_is_full(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as d:
                box = Toolbox(rgb(), d, prompt=prompt)
                client = ScriptClient([response([call('segment_prompt', {'time': 't2', 'box': [3,4,9,10]})]),
                                       response([call('submit', {'mask': 'mask#0', 'summary': 'Detected change.'})])])
                recorder = Recorder(Path(d), 'synthetic', 'image.png', gt=mask())
                final = run_session(box, Planner(client=client, sleep=lambda _: None), recorder,
                                    progressive_tools=enabled)
                self.assertEqual('combine' in names(client.requests[0]['tools']), not enabled)
                self.assertIn('combine', names(client.requests[1]['tools']))
                self.assertEqual(final['submitted_by'], 'model')
