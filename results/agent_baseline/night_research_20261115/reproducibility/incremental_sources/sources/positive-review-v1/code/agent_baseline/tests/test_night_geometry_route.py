import unittest
from agent_baseline.analysis.night_geometry_route import route,freeze


class RouteTest(unittest.TestCase):
    def test_fixed_geometry_only_routes_new_buildings(self):
        for kind in ('internal_hole','adjacent_edge_fragment'):
            self.assertEqual(route('second_appeared',kind,'V4'),'V0')
            self.assertEqual(route('levir_any',kind,'V2'),'V2')
            self.assertEqual(route('second_disappeared',kind,'V0'),'V0')
        self.assertEqual(route('second_appeared','adjacent_region','V4'),'V4')

    def test_only_positive_complete_primary_tune_can_freeze(self):
        tune=dict(task_selection={'fixed':True},groups={'second_appeared':{'10':dict(complete=True,delta_vs_task=0.)}})
        with self.assertRaises(ValueError):freeze(tune)
        tune['groups']['second_appeared']['10']['delta_vs_task']=.01
        self.assertFalse(freeze(tune)['confirmation_used'])
        tune['groups']['second_appeared']['10']['complete']=False
        with self.assertRaises(ValueError):freeze(tune)


if __name__=='__main__':unittest.main()
