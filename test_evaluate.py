import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from model import DecisionModel
from network import weights_digest


class EvaluationTests(unittest.TestCase):
    def test_check_detects_stale_artifacts_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch.object(evaluate,'ROOT',folder), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(evaluate.run())
                before={p.name:p.read_bytes() for p in folder.iterdir()}
                self.assertTrue(evaluate.run(check=True))
                self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})
                report=folder/'evaluation_100.json'
                report.write_text('{}\n',encoding='utf-8')
                stale={p.name:p.read_bytes() for p in folder.iterdir()}
                self.assertFalse(evaluate.run(check=True))
                self.assertEqual(stale,{p.name:p.read_bytes() for p in folder.iterdir()})

    def test_direct_weight_change_is_fingerprinted_and_failure_exits_nonzero(self):
        model=DecisionModel()
        original_digest=weights_digest(model.weights)
        # Erase utility differences in the actual tensor; do not fake decisions.
        model.weights[2][:,0]=0
        summary,_=evaluate.build_artifacts(model)
        self.assertNotEqual(summary['weights_sha256'],original_digest)
        self.assertLess(summary['policy_checks_passed'],summary['total_probes'])
        with tempfile.TemporaryDirectory() as tmp:
            script=(
                'import sys; from pathlib import Path; import evaluate; '
                'from model import DecisionModel; '
                'model=DecisionModel(); model.weights[2][:,0]=0; '
                'evaluate.ROOT=Path(sys.argv[1]); '
                'sys.exit(0 if evaluate.run(model=model) else 1)'
            )
            proc=subprocess.run([sys.executable,'-c',script,tmp],cwd=Path(__file__).resolve().parent,
                                capture_output=True,text=True)
            self.assertEqual(proc.returncode,1,proc.stderr)
            self.assertIn('Policy checks failed',proc.stderr)
            report=json.loads((Path(tmp)/'evaluation_100.json').read_text(encoding='utf-8'))
            self.assertEqual(report['weights_sha256'],summary['weights_sha256'])
            self.assertLess(report['policy_checks_passed'],report['total_probes'])


if __name__=='__main__':
    unittest.main()
