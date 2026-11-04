import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from model import Model, ROOT, encode, tokenize
from train import build_data

class Tests(unittest.TestCase):
    def test_decision_constraints_and_no_feasible_options(self):
        m=Model()
        options=[{'id':'a','label':'Concise verified answer','eligible':False},
                 {'id':'b','label':'Long answer','eligible':True}]
        result=m.decide('Which answer style would I prefer?',options)
        self.assertEqual(result['decision'],'b')
        self.assertEqual(result['excluded'][0]['id'],'a')
        options[1]['eligible']=False
        self.assertIsNone(m.decide('Which?',options)['decision'])
        options[1]['eligible']='false'
        with self.assertRaises(ValueError): m.decide('Which?',options)

    def test_decision_and_ranking_use_same_network(self):
        m=Model()
        texts=['Concise direct answers with verified facts','Long padded answers with unsupported claims']
        options=[{'id':str(i),'label':t} for i,t in enumerate(texts)]
        a=m.decide('Which answer style would I prefer?',options)
        b=m.rank(texts,'Which answer style would I prefer?')
        self.assertEqual(a['ranking'][0]['label'],b['selected'])

    def test_total_parameter_budget_and_finite_weights(self):
        m=Model()
        self.assertEqual(m.parameter_count,9272)
        for w in (m.w1,m.b1,m.w2,m.b2):
            self.assertTrue(np.isfinite(w).all())
            self.assertEqual(w.dtype,np.float32)

    def test_token_boundary(self):
        self.assertEqual(len(tokenize('word '*100)),100)
        self.assertEqual(encode('word '*100).shape,(512,))
        for bad in ['', 'word '*101]:
            with self.assertRaises(ValueError): encode(bad)

    def test_choice_is_ranking_top_and_order_independent(self):
        m=Model()
        options=['Concise direct answers','Verbose padded answers','Invented facts']
        a=m.rank(options,'Which answer style would I prefer?')
        b=m.rank(options[::-1],'Which answer style would I prefer?')
        self.assertEqual(a['selected'],a['ranking'][0]['option'])
        self.assertEqual([x['option'] for x in a['ranking']],[x['option'] for x in b['ranking']])
        with self.assertRaises(ValueError): m.rank(['only one'])

    def test_option_token_limit(self):
        with self.assertRaises(ValueError): Model().rank(['a '*100,'b'],'Which?')

    def test_benchmark_ground_truth_and_separation(self):
        data=json.loads((ROOT/'benchmark_100.json').read_text())
        self.assertEqual(len(data),100)
        self.assertEqual(len({r['id'] for r in data}),100)
        train,_,_=build_data()
        train_text={r[0] for r in train}
        for r in data:
            self.assertNotIn(r['question'],train_text)
            if r['domain'] in ['food','music','politics']:
                self.assertIsNone(r['expected'])

    def test_report_matches_current_weights(self):
        import hashlib
        report=json.loads((ROOT/'evaluation_100.json').read_text())
        self.assertEqual(report['weights_sha256'],hashlib.sha256((ROOT/'weights.npz').read_bytes()).hexdigest())
        self.assertEqual(report['labeled']+report['unlabeled'],100)
        self.assertEqual(report['correct'],sum(r['correct'] is True for r in report['results']))

    def test_retraining_reproduces_arrays_and_benchmark_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,str(ROOT/'train.py'),'--output-dir',tmp],check=True,capture_output=True)
            with np.load(Path(tmp)/'weights.npz') as regenerated,np.load(ROOT/'weights.npz') as shipped:
                self.assertEqual(set(regenerated.files),set(shipped.files))
                for key in shipped.files:
                    self.assertEqual(regenerated[key].dtype,np.float32)
                    self.assertTrue(np.isfinite(regenerated[key]).all())
                    # Float32 reductions differ slightly across CPU/BLAS builds,
                    # and those differences accumulate over 1,800 Adam steps.
                    # Bound numerical drift and require identical choices below.
                    np.testing.assert_allclose(regenerated[key],shipped[key],atol=3e-5,rtol=1e-5)
            regenerated_model=Model(Path(tmp)/'weights.npz')
            shipped_model=Model()
            for item in json.loads((ROOT/'benchmark_100.json').read_text()):
                with self.subTest(id=item['id']):
                    if item['mode']=='yes_no':
                        self.assertEqual(regenerated_model.predict(item['question'])['answer'],
                                         shipped_model.predict(item['question'])['answer'])
                    else:
                        regenerated_order=[row['option'] for row in regenerated_model.rank(item['options'],item['question'])['ranking']]
                        shipped_order=[row['option'] for row in shipped_model.rank(item['options'],item['question'])['ranking']]
                        self.assertEqual(regenerated_order,shipped_order)

if __name__=='__main__': unittest.main()
