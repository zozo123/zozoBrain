"""The README quickstart must work exactly as written."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class ReadmeTests(unittest.TestCase):
    def test_quickstart_runs_as_written(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        block = text.split('## Your brain in 60 seconds')[1].split('```bash\n')[1].split('```')[0]
        with tempfile.TemporaryDirectory() as folder:
            for name in ('brain.py', 'brain_bundle.py', 'network.py', 'weight_patches.py', 'model.py', 'harness.py'):
                shutil.copy2(ROOT / name, folder)
            shutil.copytree(ROOT / 'examples', Path(folder) / 'examples')
            outputs = []
            for line in block.strip().splitlines():
                args = shlex.split(line)
                self.assertEqual(args[:3], ['uv', 'run', '--locked'])
                result = subprocess.run([sys.executable, *args[3:]], cwd=folder, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f'{line}\n{result.stderr}')
                outputs.append(result.stdout)
        self.assertIn('Abstained', outputs[1])
        self.assertIn('Decision: A (Lower-cost option)', outputs[4])
        self.assertIn('Contributors: codex, claude', outputs[4])
        self.assertIn('Decision: B (Offline option)', outputs[-1])
        self.assertIn('B = neuron 1 (-3) + neuron 32 (+10)', outputs[-1])


if __name__ == '__main__':
    unittest.main()
