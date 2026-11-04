import unittest
import contextlib
import io
from demo import cases
from model import DecisionModel
from stack_demo import run as run_stack_demo

class DemoTests(unittest.TestCase):
    def test_sequential_harness_demo_preserves_both_weight_edits(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = run_stack_demo()
        self.assertEqual(result['contributing_harnesses'], ['codex', 'claude'])
        self.assertEqual({row['id']: row['utility'] for row in result['ranking']},
                         {'concise': 6, 'verified': 8})
        self.assertIn('After Codex: concise', output.getvalue())
        self.assertIn('Codex + Claude: verified', output.getvalue())

    def test_documented_outputs(self):
        expected=['no','yossieliaz.netlify.app','one page','concise and verified',None]
        model=DecisionModel()
        actual=[model.decide(r['question'],r['contexts'],r['options'])['decision'] for _,r in cases()]
        self.assertEqual(actual,expected)

if __name__=='__main__': unittest.main()
