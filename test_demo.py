import unittest
from demo import cases
from model import DecisionModel

class DemoTests(unittest.TestCase):
    def test_documented_outputs(self):
        expected=['no','yossieliaz.netlify.app','one page','concise and verified',None]
        model=DecisionModel()
        actual=[model.decide(r['question'],r['contexts'],r['options'])['decision'] for _,r in cases()]
        self.assertEqual(actual,expected)

if __name__=='__main__': unittest.main()
