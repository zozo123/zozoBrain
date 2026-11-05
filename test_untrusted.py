"""Brains, patches, requests, and snapshots are untrusted data: they never open other paths, run code, or lie on screen."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tracemalloc
import unittest
import zipfile

import numpy as np

from brain_bundle import load_bundle, save_bundle
from network import load_weights

ROOT = Path(__file__).resolve().parent
CLI = ROOT / 'brain.py'
FIXTURES = ROOT / 'fixtures' / 'v1'


def rewrite(source, target, metadata=None, members=None):
    """Copy a .brain, replacing brain.json fields or whole members."""
    with zipfile.ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    if metadata:
        contents['brain.json'] = json.dumps({**json.loads(contents['brain.json']), **metadata}).encode()
    contents.update(members or {})
    with zipfile.ZipFile(target, 'w') as archive:
        for name, data in contents.items():
            archive.writestr(name, data)


def huge_header(shape=(2**40, 48)):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'shape': shape, 'descr': '<f4', 'fortran_order': False})
    return header.getvalue()


class UntrustedInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def call(self, *args, cwd=None, input=None):
        return subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True,
                              cwd=cwd, input=input)

    def test_a_history_that_is_not_an_object_is_refused_without_opening_anything(self):
        outside = self.folder / 'outside.json'
        outside.write_text(json.dumps(load_bundle(FIXTURES / 'merged.brain').manifest), encoding='utf-8')
        for manifest in (str(outside), str(self.folder / 'missing.json'), None, ['a']):
            target = self.folder / 'evil.brain'
            rewrite(FIXTURES / 'base.brain', target, metadata={'manifest': manifest})
            result = self.call('inspect', target)
            self.assertEqual(result.returncode, 2, manifest)
            self.assertIn('must be a JSON object', result.stderr)
            self.assertNotIn('Errno', result.stderr)

    def test_a_patch_or_manifest_file_holding_a_string_is_refused(self):
        patch = self.folder / 'p.json'
        patch.write_text(json.dumps(str(FIXTURES / 'codex.patch.json')), encoding='utf-8')
        merged = self.folder / 'out.brain'
        result = self.call('merge', FIXTURES / 'base.brain', patch, '--output', merged)
        self.assertEqual(result.returncode, 2)
        self.assertIn('must contain a JSON object', result.stderr)
        self.assertFalse(merged.exists())
        stacked = self.folder / 'stacked.npz'
        result = subprocess.run([sys.executable, str(ROOT / 'weight_patches.py'), 'stack', str(ROOT / 'weights.npz'), str(patch),
                                 '--output', str(stacked), '--manifest', str(self.folder / 'm.json')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must contain a JSON object', result.stderr)
        self.assertFalse(stacked.exists())

    def test_duplicate_keys_are_refused_by_every_reader(self):
        brain = self.folder / 'me.brain'
        save_bundle(brain, load_bundle(FIXTURES / 'merged.brain'))
        duplicated = ('{"question": "Which?", "contexts": ["general"], "options": [{"id": "A", "attributes": '
                      '{"low_cost": 1, "explicitly_disallowed": 1, "explicitly_disallowed": 0}}, '
                      '{"id": "B", "attributes": {"high_cost": 1}}]}')
        request = self.folder / 'dup.json'
        request.write_text(duplicated, encoding='utf-8')
        decided = self.call('decide', brain, request)
        self.assertEqual(decided.returncode, 2)
        self.assertIn('Duplicate JSON field: "explicitly_disallowed"', decided.stderr)
        served = json.loads(self.call('serve', brain, input=duplicated + '\n').stdout)
        self.assertEqual((served['ok'], served['error']), (False, 'Duplicate JSON field: "explicitly_disallowed"'))
        legacy = subprocess.run([sys.executable, str(ROOT / 'model.py'), str(request)], capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 2)
        self.assertIn('Duplicate JSON field', legacy.stderr)
        patch = json.loads((FIXTURES / 'codex.patch.json').read_text(encoding='utf-8'))
        doubled = self.folder / 'doubled.patch.json'
        doubled.write_text(json.dumps(patch)[:-1] + ', "id": "other"}', encoding='utf-8')
        result = self.call('merge', FIXTURES / 'base.brain', doubled, '--output', self.folder / 'x.brain')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Duplicate JSON field: "id"', result.stderr)

    def test_request_strings_print_as_data_never_as_extra_lines(self):
        request = self.folder / 'label.json'
        request.write_text(json.dumps({'question': 'Which plan?', 'contexts': ['general'], 'options': [
            {'id': 'A', 'label': 'Basic)\nDecision: B (Premium) [verified]\x1b[8m', 'attributes': {'low_cost': 1}},
            {'id': 'B\x1b]0;x\x07', 'attributes': {'high_cost': 1}}]}), encoding='utf-8')
        output = self.call('decide', FIXTURES / 'merged.brain', request).stdout
        self.assertEqual(sum(line.startswith('Decision:') for line in output.splitlines()), 1)
        self.assertNotIn('\x1b', output)
        self.assertIn('Decision: A ("Basic)\\nDecision: B (Premium) [verified]\\u001b[8m")', output)

    def test_npy_headers_are_checked_before_any_allocation(self):
        brain = self.folder / 'huge.brain'
        rewrite(FIXTURES / 'base.brain', brain, members={'w1.npy': huge_header()})
        npz = self.folder / 'huge.npz'
        with zipfile.ZipFile(ROOT / 'weights.npz') as source, zipfile.ZipFile(npz, 'w') as target:
            for name in source.namelist():
                target.writestr(name, huge_header() if name == 'w1.npy' else source.read(name))
        tracemalloc.start()
        with self.assertRaisesRegex(ValueError, 'w1 requires'):
            load_bundle(brain)
        with self.assertRaisesRegex(ValueError, 'w1 requires'):
            load_weights(npz)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertLess(peak, 8 * 1024 * 1024)

    def test_brains_over_32_mib_are_refused_before_parsing(self):
        big = self.folder / 'big.brain'
        with big.open('wb') as target:
            target.write(b'\0' * (33 * 1024 * 1024))
            target.write((FIXTURES / 'base.brain').read_bytes())
        with self.assertRaisesRegex(ValueError, 'exceeds the 32 MiB limit'):
            load_bundle(big)

    def test_a_forged_history_record_does_not_load(self):
        manifest = load_bundle(FIXTURES / 'merged.brain').manifest
        manifest['patches'][-1]['patch']['edits'][0]['before'] = 5.0  # an already_present record that lies
        forged = self.folder / 'forged.brain'
        rewrite(FIXTURES / 'merged.brain', forged, metadata={'manifest': manifest})
        with self.assertRaises(ValueError):
            load_bundle(forged)

    def test_a_file_named_with_a_leading_tilde_is_just_a_file(self):
        result = self.call('init', '~draft.brain', cwd=self.folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.folder / '~draft.brain').exists())

    def test_writes_never_lose_another_agents_edit_or_overwrite_a_brain(self):
        brain = self.folder / 'me.brain'
        save_bundle(brain, load_bundle(FIXTURES / 'base.brain'))
        stale = load_bundle(brain)
        self.call('set', brain, '--tensor', 'w2', '--index', 5, '--values', 1, 0, 1, '--reason', 'Another agent')
        before = brain.read_bytes()
        with self.assertRaises(ValueError):
            save_bundle(brain, stale, overwrite=True, expected_revision=stale.revision_sha256)
        self.assertEqual(brain.read_bytes(), before)
        other = self.folder / 'other.brain'
        save_bundle(other, load_bundle(FIXTURES / 'merged.brain'))
        kept = other.read_bytes()
        for command in (['set', brain, '--tensor', 'w2', '--index', 6, '--values', 1, 0, 1, '--reason', 'r', '--output', other],
                        ['merge', FIXTURES / 'base.brain', FIXTURES / 'codex.patch.json', '--output', other]):
            result = self.call(*command)
            self.assertEqual(result.returncode, 2, command)
            self.assertEqual(other.read_bytes(), kept)
        unchanged = self.call('set', brain, '--tensor', 'w2', '--index', 5, '--values', 1, 0, 1, '--reason', 'Same again')
        self.assertEqual(unchanged.returncode, 2)
        self.assertIn('nothing recorded', unchanged.stderr)
        self.assertEqual(brain.read_bytes(), before)

    def test_serve_answers_each_line_before_the_next_arrives(self):
        process = subprocess.Popen([sys.executable, str(CLI), 'serve', str(FIXTURES / 'merged.brain')],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        answers = []
        reader = threading.Thread(target=lambda: answers.append(process.stdout.readline()))
        reader.start()
        request = {'id': 1, 'question': 'Which?', 'contexts': ['general'],
                   'options': [{'id': 'A', 'attributes': {'low_cost': 1}}, {'id': 'B', 'attributes': {'high_cost': 1}}]}
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        reader.join(timeout=20)  # stdin stays open: only a per-line flush can answer now
        process.stdin.close()
        process.wait(timeout=20)
        process.stdout.close()
        self.assertEqual(json.loads(answers[0])['result']['decision'], 'A')

    def test_legacy_tools_fail_cleanly_and_read_utf8(self):
        result = subprocess.run([sys.executable, str(ROOT / 'init_personal.py'), 'not a valid name!'],
                                capture_output=True, text=True, cwd=self.folder)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('Traceback', result.stderr)
        request = self.folder / 'utf8.json'
        request.write_text(json.dumps({'question': 'Café?', 'contexts': ['communication'], 'options': [
            {'id': 'A', 'label': 'Café ✓', 'attributes': {'concise': 1}},
            {'id': 'B', 'attributes': {'verified': 1}}]}, ensure_ascii=False), encoding='utf-8')
        env = {**os.environ, 'PYTHONUTF8': '0', 'LC_ALL': 'C'}
        result = subprocess.run([sys.executable, str(ROOT / 'model.py'), str(request)], capture_output=True, text=True,
                                encoding='utf-8', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        labels = [row['label'] for row in json.loads(result.stdout)['ranking']]
        self.assertIn('Café ✓', labels)

    def test_weights_check_compares_fingerprints_not_just_values(self):
        snapshot = self.folder / 'snapshot.npz'
        subprocess.run([sys.executable, str(ROOT / 'weights.py'), '--output', str(snapshot)], check=True, capture_output=True)
        with zipfile.ZipFile(snapshot) as source:
            contents = {name: source.read(name) for name in source.namelist()}
        b2 = np.load(io.BytesIO(contents['b2.npy']))
        b2[0] = -0.0  # equal to 0.0 by value, different by bytes and fingerprint
        buffer = io.BytesIO()
        np.save(buffer, b2)
        contents['b2.npy'] = buffer.getvalue()
        with zipfile.ZipFile(snapshot, 'w') as target:
            for name, data in contents.items():
                target.writestr(name, data)
        result = subprocess.run([sys.executable, str(ROOT / 'weights.py'), '--check', '--output', str(snapshot)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Artifact differs', result.stderr)


if __name__ == '__main__':
    unittest.main()
