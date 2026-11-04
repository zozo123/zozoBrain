"""Generated profiles must execute the source or artifact weights users actually edit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from init_personal import ROOT, create


class PersonalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = create('test_person', Path(self.tmp.name))

    def run_profile(self, *args, check=True):
        return subprocess.run(
            [sys.executable, str(self.folder / 'model.py'), str(self.folder / 'decision.json'), *args],
            check=check, capture_output=True, text=True,
        )

    def edit_source(self, *rows):
        path = self.folder / 'weights.py'
        source = path.read_text(encoding='utf-8')
        for row in rows:
            source = source.replace('# ' + row, row)
        path.write_text(source, encoding='utf-8')

    def test_blank_profile_then_direct_source_weights(self):
        self.assertFalse((self.folder / 'policy.json').exists())
        self.assertIsNone(json.loads(self.run_profile().stdout)['decision'])
        self.edit_source('w2[0] = [3, 0, 1]', 'w2[1] = [-3, 0, 1]')
        self.assertEqual(json.loads(self.run_profile().stdout)['decision'], 'A')
        source_path = self.folder / 'weights.py'
        timestamp = source_path.stat()
        source = source_path.read_text(encoding='utf-8')
        reversed_source = source.replace('w2[0] = [3, 0, 1]', 'w2[0] = [-3, 0, 1]').replace(
            'w2[1] = [-3, 0, 1]', 'w2[1] = [3, 0, 1]',
        )
        self.assertEqual(len(source), len(reversed_source))
        source_path.write_text(reversed_source, encoding='utf-8')
        # Model rapid same-size edits deterministically; a source-weight run must
        # not silently reuse timestamp-based Python bytecode for the old policy.
        os.utime(source_path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
        self.assertEqual(json.loads(self.run_profile().stdout)['decision'], 'B')

    def test_artifact_edits_are_loaded_only_when_selected(self):
        self.edit_source('w2[0] = [3, 0, 1]', 'w2[1] = [-3, 0, 1]')
        exporter = [sys.executable, str(self.folder / 'weights.py')]
        subprocess.run(exporter, check=True, capture_output=True, text=True)
        subprocess.run([*exporter, '--check'], check=True, capture_output=True, text=True)
        artifact = self.folder / 'weights.npz'
        self.assertEqual(json.loads(self.run_profile('--weights', str(artifact)).stdout)['decision'], 'A')
        with np.load(artifact, allow_pickle=False) as archive:
            tensors = {name: archive[name].copy() for name in archive.files}
        tensors['w2'][0] = [-3, 0, 1]
        tensors['w2'][1] = [3, 0, 1]
        np.savez(artifact, **tensors)
        self.assertEqual(json.loads(self.run_profile('--weights', str(artifact)).stdout)['decision'], 'B')
        self.assertEqual(json.loads(self.run_profile().stdout)['decision'], 'A')
        mismatch = subprocess.run([*exporter, '--check'], capture_output=True, text=True)
        self.assertNotEqual(mismatch.returncode, 0)

    def test_renamed_custom_veto_rejects_fractional_input_and_survives_force(self):
        source_path = self.folder / 'weights.py'
        source_path.write_text(
            source_path.read_text(encoding='utf-8').replace('explicitly_disallowed', 'unsafe_route'),
            encoding='utf-8',
        )
        self.edit_source('w2[3] = [-20, 1, 1]')
        decision_path = self.folder / 'decision.json'
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        decision['options'][0]['attributes'] = {'unsafe_route': 1}
        decision_path.write_text(json.dumps(decision), encoding='utf-8')
        result = json.loads(self.run_profile('--force').stdout)
        self.assertEqual(result['decision'], 'B')
        self.assertEqual([item['id'] for item in result['vetoed']], ['A'])
        decision['options'][0]['attributes']['unsafe_route'] = 0.5
        decision_path.write_text(json.dumps(decision), encoding='utf-8')
        rejected = self.run_profile('--force', check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertRegex(rejected.stderr.lower(), 'binary|0 or 1')

    def test_existing_profiles_and_invalid_names_are_refused(self):
        original = (self.folder / 'weights.py').read_bytes()
        with self.assertRaises(FileExistsError):
            create('test_person', Path(self.tmp.name))
        self.assertEqual((self.folder / 'weights.py').read_bytes(), original)
        for name in ('../outside', '', 'a/b', None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create(name, Path(self.tmp.name))

    def test_profile_carries_locked_runtime_and_private_contribution_defaults(self):
        for filename in ('pyproject.toml', 'uv.lock', 'network.py', 'weight_patches.py', 'edit_weights.py', 'LICENSE'):
            self.assertEqual((self.folder / filename).read_bytes(), (ROOT / filename).read_bytes())
        ignored = (self.folder / '.gitignore').read_text(encoding='utf-8').splitlines()
        self.assertIn('contributions/', ignored)
        self.assertIn('*.patch.json', ignored)


if __name__ == '__main__':
    unittest.main()
