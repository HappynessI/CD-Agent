import unittest
import numpy as np
from agent_baseline.analysis.n8_t3_route import routed_predictions
from agent_baseline.analysis.n8_t3_common import observable_example


class RoutingTests(unittest.TestCase):
    def test_predicted_endpoint_controls_route_and_fusion(self):
        ex=np.zeros((3,3),bool);ex[0,0]=True
        p1=np.zeros((7,3,3));p2=np.zeros_like(p1);p1[1]=.9;p1[0]=.1;p2[5]=.8;p2[0]=.2
        observable,dense=observable_example([p1,p2],ex)
        self.assertEqual(observable['predicted_pair'],[1,5]);self.assertTrue(observable['structural'])
        inst=np.zeros_like(ex);inst[2,2]=True
        pred=routed_predictions(inst,dense,[p1,p2],observable)
        np.testing.assert_array_equal(pred['route'],inst)
        self.assertEqual(int(pred['hybrid_union_0.7'].sum()),8)
        self.assertEqual(int(pred['hybrid_intersection_0.85'].sum()),0)
        p2[:]=0;p2[3]=1
        obs,dense=observable_example([p1,p2],ex)
        self.assertFalse(obs['structural'])
        pred=routed_predictions(inst,dense,[p1,p2],obs)
        np.testing.assert_array_equal(pred['route'],dense)


if __name__=='__main__':unittest.main()
