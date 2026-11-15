import inspect
import os
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline import tools
from agent_baseline.state import Session
from agent_baseline.analysis import n8_t3_online_provider as provider


class ProviderTests(unittest.TestCase):
    def test_optin_keeps_match_default_and_does_not_set_current(self):
        match=tools.Toolbox.match_exemplar;previous_schemas=tools.session_schemas
        old_schema=tools.SCHEMA_BY_NAME.get('retrieve_example');old_method=getattr(tools.Toolbox,'retrieve_example',None)
        try:
            provider.install();self.assertIs(tools.Toolbox.match_exemplar,match)
            self.assertIn('retrieve_example',[s['function']['name'] for s in tools.session_schemas(Session((4,4)))])
            with tempfile.TemporaryDirectory() as d:
                cfg=Path(d)/'config.json';cfg.write_text(json.dumps(dict(rank='joint',k=5,system_method='route')))
                session=Session((4,4));ex=np.zeros((4,4),bool);ex[0,0]=True;handle=session.add_mask(ex)
                before=session.current_mask;box=tools.Toolbox([np.zeros((4,4,3),np.uint8)]*2,Path(d),session=session)
                mask=np.zeros((1,4,4),bool);mask[0,2,2]=True
                with patch.dict(os.environ,dict(N8_ONLINE_ARM='n8',N8_INFERENCE_CONFIG=str(cfg))),patch.object(provider,'proposals',return_value=(mask,[.7],dict(arm='n8'))):
                    result=box.execute('retrieve_example',dict(mask=handle))
                self.assertTrue(result.public['ok']);self.assertEqual(session.current_mask,before)
                self.assertEqual(result.public['ranked_ids'],[1])
        finally:
            tools.session_schemas=previous_schemas
            if old_schema is None:tools.SCHEMA_BY_NAME.pop('retrieve_example',None)
            else:tools.SCHEMA_BY_NAME['retrieve_example']=old_schema
            if old_method is None:delattr(tools.Toolbox,'retrieve_example')
            else:tools.Toolbox.retrieve_example=old_method

    def test_inference_signature_is_whitelist_only(self):
        self.assertEqual(list(inspect.signature(provider.proposals).parameters),['images','example','backend','arm','config'])


if __name__=='__main__':unittest.main()
