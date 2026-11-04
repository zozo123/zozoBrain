import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent

class HarnessTests(unittest.TestCase):
    def test_nonfinite_or_malformed_metadata_does_not_kill_stream(self):
        good = json.loads((ROOT/'examples/travel.json').read_text())
        bad = [dict(good, id=float('nan')), dict(good, id={'bad': 'id'}),
               dict(good, options=[{'id': 'a', 'label': float('inf')}, {'id': 'b'}]),
               dict(good, contexts={'travel': False}), dict(good, force='false')]
        data = '\n'.join(json.dumps(row) for row in bad + [good]) + '\n'
        proc = subprocess.run([sys.executable, str(ROOT/'harness.py')], input=data,
                              text=True, capture_output=True, check=True)
        rows = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(len(rows), len(bad) + 1)
        self.assertTrue(all(not row['ok'] for row in rows[:-1]))
        self.assertEqual(rows[-1]['result']['decision'], 'no')
        self.assertEqual(proc.stderr, '')

    def test_stream_roundtrip_error_recovery_and_abstention(self):
        travel=json.loads((ROOT/'examples/travel.json').read_text())
        travel['id']='travel'
        food=json.loads((ROOT/'examples/food.json').read_text())
        food['id']='food'
        data=json.dumps(travel)+'\nnot json\n'+json.dumps(food)+'\n'
        proc=subprocess.run([sys.executable,str(ROOT/'harness.py')],input=data,text=True,capture_output=True,check=True)
        self.assertEqual(proc.stderr,'')
        rows=[json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['id'],'travel')
        self.assertEqual(rows[0]['result']['decision'],'no')
        self.assertFalse(rows[1]['ok'])
        self.assertEqual(rows[2]['id'],'food')
        self.assertIsNone(rows[2]['result']['decision'])

if __name__=='__main__': unittest.main()
