import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parent

class HarnessTests(unittest.TestCase):
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
