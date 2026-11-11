"""Catalog contracts against actual planner registration, with no backends."""
import json
from pathlib import Path
import unittest
from agent_baseline import tools
from agent_baseline.state import Session
from agent_baseline.make_tools_page import build_catalog, render_catalog
from agent_baseline.make_g3_pages import audit_html


class ToolsPageTest(unittest.TestCase):
    def test_names_and_parameters_match_registration(self):
        session=Session((1,1))
        schemas=tools.session_schemas(session,python=True,tools_t13=True)
        expected={s['function']['name'] for s in schemas+[tools.PLAN_SCHEMA,tools.SKILL_SCHEMA]}
        session.external_examples=(('example#0','exmask#1'),)
        expanded=tools.session_schemas(session,python=True,tools_t13=True)+[tools.PLAN_SCHEMA,tools.SKILL_SCHEMA]
        specs={s['function']['name']:s['function']['parameters'] for s in expanded}
        catalog=build_catalog()
        self.assertEqual({t['name'] for t in catalog},expected)
        self.assertEqual(len(catalog),len(expected))
        for tool in catalog:
            spec=specs[tool['name']]
            self.assertEqual(tool['schema'],spec)
            self.assertEqual({p['name'] for p in tool['parameters']},set(spec['properties']))
            for p in tool['parameters']:
                self.assertEqual(p['schema'],spec['properties'][p['name']])
                self.assertEqual(p['type'],spec['properties'][p['name']]['type'])
                self.assertEqual(p['required'],p['name'] in spec.get('required',[]))
        stored=Path(__file__).resolve().parents[3]/'results/tools/tools_catalog.json'
        if stored.exists(): self.assertEqual(json.loads(stored.read_text()),catalog)

    def test_conditional_extensions_and_switches(self):
        catalog={t['name']:t for t in build_catalog()}
        self.assertEqual(catalog['view']['switch']['label'],'默认开')
        for name in tools.T13_NAMES:
            self.assertEqual(catalog[name]['switch']['flag'],'--tools-t13')
        exemplar=catalog['match_exemplar']['schema']['properties']
        self.assertIn('negative',exemplar)
        self.assertIn('example',exemplar['exemplar']['properties'])
        self.assertNotIn('negative',next(s['function']['parameters']['properties'] for s in tools.tool_schemas() if s['function']['name']=='match_exemplar'))

    def test_html_self_contained(self):
        document=render_catalog(build_catalog())
        audit_html(document)
        for term in ('没有一键输出变化掩码','map#0','region#0','example#k','exmask#k','feat#k','props#k','提交门','未定义','prefers-color-scheme:dark'):
            self.assertIn(term,document)


if __name__=='__main__': unittest.main()
