import unittest
import contextlib
import io
from demo import cases
from model import DecisionModel
from stack_demo import run as run_stack_demo
from simplest import main as run_simplest

class DemoTests(unittest.TestCase):
    def test_two_agent_demo_changes_decisions_and_refuses_conflicts(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = run_stack_demo()
        self.assertEqual(result['brain'].contributing_harnesses, ['codex', 'claude'])
        self.assertEqual(len(result['brain'].manifest['patches']), 4)
        self.assertEqual([row['decision'] for row in result['decisions']],
                         ['cheap', 'pricey', 'cheap_offline', None])
        self.assertEqual([row['id'] for row in result['decisions'][1]['vetoed']], ['banned_cheap'])
        self.assertIn('Conflicting edit', result['conflict'])
        self.assertIn('merge refused', output.getvalue())

    def test_simplest_written_weights_change_the_decision(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_simplest(), ['abstain', 'cheap', 'pricey'])

    def test_documented_outputs(self):
        expected=['no','yossieliaz.netlify.app','one page','concise and verified',None]
        model=DecisionModel()
        actual=[model.decide(r['question'],r['contexts'],r['options'])['decision'] for _,r in cases()]
        self.assertEqual(actual,expected)

if __name__=='__main__': unittest.main()
