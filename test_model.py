import unittest
import numpy as np
from model import DecisionModel
from weights import authored_weights,forward,ATTRIBUTES,CONTEXTS
from evaluate import benchmark

class Tests(unittest.TestCase):
    def test_all_dense_parameters_counted(self):
        self.assertEqual(sum(a.size for a in authored_weights()),2499)
        self.assertLessEqual(DecisionModel().parameter_count,10000)
    def test_context_gates(self):
        x=np.zeros(48,dtype=np.float32)
        x[ATTRIBUTES.index('single_page')]=1
        self.assertEqual(forward(x)[0],0)
        x[32+CONTEXTS.index('cv')]=1
        self.assertEqual(forward(x)[0],3)
    def test_veto_overrides_all_positive_attributes(self):
        m=DecisionModel()
        attrs={name:1 for name in ATTRIBUTES}
        out=m.decide('Which?',CONTEXTS,[dict(id='a',attributes=attrs),dict(id='b',attributes={'explicitly_preferred':1})],force=True)
        self.assertEqual(out['decision'],'b')
        self.assertTrue(out['vetoed'][0]['veto'])
    def test_all_vetoed_never_forced(self):
        opts=[dict(id=x,attributes={'explicitly_disallowed':1}) for x in ['a','b']]
        self.assertIsNone(DecisionModel().decide('Which?',['general'],opts,True)['decision'])
    def test_missing_preference_abstains(self):
        opts=[dict(id=x,attributes={}) for x in ['pizza','sushi']]
        m=DecisionModel()
        self.assertIsNone(m.decide('Which food?',['food'],opts)['decision'])
        self.assertTrue(m.decide('Which food?',['food'],opts,True)['forced'])
    def test_order_invariance_when_not_tied(self):
        opts=[dict(id='a',attributes={'concise':1}),dict(id='b',attributes={'speculative':1})]
        m=DecisionModel()
        self.assertEqual(m.decide('Which?',['communication'],opts)['decision'],m.decide('Which?',['communication'],opts[::-1])['decision'])
    def test_validation(self):
        m=DecisionModel()
        opts=[dict(id='a',attributes={}),dict(id='b',attributes={})]
        m.decide('word '*100,['general'],opts)
        with self.assertRaises(ValueError): m.decide('word '*101,['general'],opts)
        with self.assertRaises(ValueError): m.decide('Which?',['unknown'],opts)
        for value in [float('nan'),-1,2,.5]:
            with self.assertRaises(ValueError): m.decide('Which?',['travel'],[dict(id='a',attributes={'shabbat_flight':value}),opts[1]])
    def test_100_probes(self):
        rows=benchmark()
        self.assertEqual(len(rows),100)
        for r in rows:
            with self.subTest(id=r['id']):
                out=DecisionModel().decide(r['question'],r['contexts'],r['options'])
                self.assertEqual(out['decision'],r['expected_decision'])
                self.assertEqual(out['status'],r['expected_status'])

if __name__=='__main__': unittest.main()
