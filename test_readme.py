"""Every documented shell command must work exactly as written."""
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
RUNTIME = ['brain.py', 'brain_bundle.py', 'network.py', 'weight_patches.py', 'model.py', 'harness.py', 'simplest.py',
           'deep.py', 'stack_demo.py', 'weights.py', 'edit_weights.py']
# Run elsewhere: cloning, the repository's own checks (CI runs them), and the timing benchmark.
SKIP = ('git clone', 'uv run --locked python -m unittest', 'uv run --locked weights.py --check',
        'uv run --locked evaluate.py --check', 'uv run --locked benchmark.py')


def commands(name):
    text = (ROOT / name).read_text(encoding='utf-8')
    return [line for block in re.findall(r'```bash\n(.*?)```', text, re.S) for line in block.strip().splitlines()
            if not line.startswith(SKIP)]


class DocsTests(unittest.TestCase):
    def run_documented(self, lines):
        """Run each line in order in a fresh copy of the runtime and return the outputs."""
        with tempfile.TemporaryDirectory() as folder:
            for name in RUNTIME:
                shutil.copy2(ROOT / name, folder)
            shutil.copytree(ROOT / 'examples', Path(folder) / 'examples')
            outputs = []
            for line in lines:
                args = shlex.split(line, comments=True)
                if args[0] == 'cp':
                    shutil.copy2(Path(folder) / args[1], Path(folder) / args[2])
                    outputs.append('')
                    continue
                self.assertEqual(args[:3], ['uv', 'run', '--locked'], line)
                result = subprocess.run([sys.executable, *args[3:]], cwd=folder, capture_output=True, text=True,
                                        stdin=subprocess.DEVNULL)
                self.assertEqual(result.returncode, 0, f'{line}\n{result.stderr}')
                outputs.append(result.stdout)
            return outputs

    def test_readme_runs_as_written(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        quickstart = text.split('## Your brain in 60 seconds')[1].split('```bash\n')[1].split('```')[0]
        outputs = self.run_documented(quickstart.strip().splitlines())
        self.assertIn('Abstained', outputs[1])
        self.assertIn('Decision: A (Lower-cost option)', outputs[4])
        self.assertIn('Contributors: codex, claude', outputs[4])
        self.assertIn('Decision: B (Offline option)', outputs[-1])
        self.assertIn('B = neuron 1 (-3) + neuron 32 (+10)', outputs[-1])
        self.run_documented(commands('README.md'))

    def test_usage_runs_as_written(self):
        outputs = self.run_documented(commands('USAGE.md'))
        self.assertIn(f"Saved {Path('personal', 'combined.brain')}", outputs[-2])
        self.assertIn('Contributors: codex, manual, claude', outputs[-2])

    def test_weights_and_harness_run_as_written(self):
        self.run_documented(['uv run --locked brain.py init personal/me.brain', *commands('WEIGHTS.md')])
        self.run_documented(commands('HARNESS.md'))


if __name__ == '__main__':
    unittest.main()
