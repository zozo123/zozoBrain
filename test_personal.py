import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from init_personal import create

class PersonalTests(unittest.TestCase):
    def test_blank_profile_then_authored_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=create('test_person',Path(tmp))
            command=[sys.executable,str(folder/'model.py'),str(folder/'decision.json')]
            def run():
                return json.loads(subprocess.run(command,check=True,capture_output=True,text=True).stdout)
            self.assertIsNone(run()['decision'])
            policy=json.loads((folder/'policy.json').read_text())
            policy['rules'][0]['utility']=3
            policy['rules'][1]['utility']=-3
            (folder/'policy.json').write_text(json.dumps(policy))
            self.assertEqual(run()['decision'],'A')
            with self.assertRaises(FileExistsError): create('test_person',Path(tmp))
            with self.assertRaises(ValueError): create('../outside',Path(tmp))

if __name__=='__main__': unittest.main()
