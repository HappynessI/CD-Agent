import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from agent_baseline.backends import encode_image
from agent_baseline.analysis.n7_t3_verify import exact_reuse, SYSTEM, T3_QUESTION, MODEL, API_SEED


class ExactReuseTests(unittest.TestCase):
    def test_identical_render_reuses_changed_image_or_prompt_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            view=Image.new('RGB',(8,8),(100,50,20))
            with Image.open(BytesIO(base64.b64decode(encode_image(view)))) as encoded:
                encoded.save(folder/'00.png')
            metadata=dict(system=SYSTEM,question=T3_QUESTION,model=MODEL,seed=API_SEED,
                          max_tokens=768,temperature=0,images=[dict(label='test',path=str(folder/'00.png'))])
            (folder/'input.json').write_text(json.dumps(metadata))
            answer={'parsed':{'change':'yes'},'cost_cny':1}
            (folder/'answer.json').write_text(json.dumps(answer))
            self.assertEqual(exact_reuse(folder,[('test',view)]),answer)
            self.assertIsNone(exact_reuse(folder,[('test',Image.new('RGB',(8,8),(0,0,0)))]))
            metadata['system']='different'
            (folder/'input.json').write_text(json.dumps(metadata))
            self.assertIsNone(exact_reuse(folder,[('test',view)]))

    def test_absent_answer_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(exact_reuse(Path(directory),[]))


if __name__=='__main__':unittest.main()
