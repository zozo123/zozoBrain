"""Files written by zozoBrain 1.0 stay readable and mean the same thing in every 1.x (FORMATS.md#compatibility)."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from brain_bundle import apply_patches, create_blank, load_bundle
from model import DecisionModel
from network import load_weights, read_json_object, weights_digest

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / 'fixtures' / 'v1'
# Written by the v1.0.1 release. A change to any of these numbers breaks the 1.x promise.
FILES = {
    'base.brain': '6330f66d674d1c71558cca01d75a39dbc89e33df444f12c7ac1f7100828a3697',
    'merged.brain': '8f8e879bc407366a4446f4d18e450dca2ea1fc0bf5f46efced0bc275fd50d154',
    'codex.patch.json': '722f6baa2017617750af81793f40c65f56b78e6da18970854139960f401cf4b2',
    'claude.patch.json': '8170b12b0c0a688c73b07c33b27fb1a20bfd875e1a8a6f2792d6743f9add5371',
    'codex-retry.patch.json': '225d54af553fa5da7c59679a43c2868b967ea5c004b36b622965fac145141fe6',
    'npz.patch.json': '2a851fa3e6e5687e2a594606211e9bddd62315cb685ada8a603e4a2caf16af5d',
    'stacked.npz': 'c1a5a8968dbbc586fc811d6444f51eae3285a02c7da66c5059587fa72fd5937a',
    'stacked.stack.json': 'ee3812bde9775a0aff89afc4c2b2d7d4f9590482a46ac6b5d6890d1e4c906917',
}
BRAINS = {  # weights_sha256, revision_sha256
    'base.brain': ('54f641cae91f0cf4bbeaf8b20ef1acaebc9d06495ba2464260d050e6eaef1df1',
                   '9f67efc7243c2c07e1b1521907f94a085bcae47a2e08f1eb189211cffbcc4d3d'),
    'merged.brain': ('07f2629cf9699b77616456acb88ef18f07bb1aedb63d88f8a943038983c2b465',
                     'af484c42b78cee8e5e5ff44872bbee2e4876e60a0714e1a71e2563a3906c6f2f'),
}
BLANK_SCHEMA = '3689c320a1e60cf9a494eac00b3c2b12ca801a7871ac7bbf1b44eb056a41994d'
STACKED_WEIGHTS = '090814442743522ec9c805e229fa5cfcb2525709dbfb9af4eda04f6288215958'


class FixtureTests(unittest.TestCase):
    def test_fixture_files_are_unchanged(self):
        for name, digest in FILES.items():
            self.assertEqual(hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), digest, name)

    def test_the_default_layout_is_frozen(self):
        # Patches only move between brains with the same layout, so init's default never changes in 1.x.
        self.assertEqual(create_blank().schema_sha256, BLANK_SCHEMA)
        self.assertEqual(load_bundle(FIXTURES / 'base.brain').schema_sha256, BLANK_SCHEMA)

    def test_brains_load_with_the_same_fingerprints_and_decisions(self):
        for name, (weights, revision) in BRAINS.items():
            bundle = load_bundle(FIXTURES / name)
            self.assertEqual((bundle.weights_sha256, bundle.revision_sha256), (weights, revision), name)
        model = DecisionModel.from_bundle(FIXTURES / 'merged.brain')

        def decide(contexts, a, b):
            options = [{'id': 'A', 'attributes': a}, {'id': 'B', 'attributes': b}]
            return model.decide('Which?', contexts, options)['decision']

        self.assertEqual(decide(['general'], {'low_cost': 1}, {'high_cost': 1, 'offline': 1}), 'A')
        self.assertEqual(decide(['general', 'traveling'], {'low_cost': 1}, {'low_cost': 1, 'offline': 1}), 'B')
        self.assertEqual(decide(['general', 'traveling'], {'low_cost': 1}, {'high_cost': 1, 'offline': 1}), 'A')  # 3 vs -3 + 7 - 2
        self.assertEqual(decide(['traveling'], {'low_cost': 1, 'explicitly_disallowed': 1}, {'offline': 1}), 'B')

    def test_patches_merge_onto_the_base_into_the_merged_brain(self):
        patches = [read_json_object(FIXTURES / f'{name}.patch.json') for name in ('codex', 'claude', 'codex-retry')]
        merged = apply_patches(load_bundle(FIXTURES / 'base.brain'), patches)
        self.assertEqual((merged.weights_sha256, merged.revision_sha256), BRAINS['merged.brain'])

    def test_legacy_npz_and_stack_manifest_verify(self):
        self.assertEqual(weights_digest(load_weights(FIXTURES / 'stacked.npz')), STACKED_WEIGHTS)
        result = subprocess.run([sys.executable, str(ROOT / 'model.py'), str(ROOT / 'examples' / 'work.json'),
                                 '--weights', str(FIXTURES / 'stacked.npz'), '--manifest', str(FIXTURES / 'stacked.stack.json')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['contributing_harnesses'], ['owner'])


if __name__ == '__main__':
    unittest.main()
