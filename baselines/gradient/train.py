"""Reproducible NumPy training. No pretrained weights or hidden language model."""
import argparse
import csv
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from model import ROOT, encode, Model

def build_data(data_dir=ROOT):
    with (data_dir/'seeds.tsv').open() as source:
        seeds = list(csv.DictReader(source, delimiter='\t'))
    pairs = json.loads((data_dir/'ranking_seeds.json').read_text())
    # Hold out an entire wording template; same underlying facts remain in training.
    prefixes = ['', 'Please answer: ', 'For Yossi: ', 'Based on my preferences: ']
    train, test = [], []
    for seed in seeds:
        for prefix in prefixes:
            train.append((prefix+seed['question'], int(seed['label']), 0))
        test.append(('Considering what you know about me, '+seed['question'], int(seed['label']), 0))
    for pair in pairs:
        for prefix in prefixes:
            for key, target in [('better',1),('worse',0)]:
                train.append((prefix+pair['context']+' option: '+pair[key], target, 1))
        for key, target in [('better',1),('worse',0)]:
            test.append(('Considering my background, '+pair['context']+' option: '+pair[key], target, 1))
    return train, test, pairs

def run(data_dir=ROOT, output_dir=ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    train, test, pairs = build_data(data_dir)
    x = np.stack([encode(t[0]) for t in train])
    y = np.array([t[1] for t in train],dtype=np.float32)
    heads = np.array([t[2] for t in train])
    params = [rng.normal(0,.15,(512,18)).astype('float32'),np.zeros(18,dtype='float32'),
              rng.normal(0,.15,(18,2)).astype('float32'),np.zeros(2,dtype='float32')]
    moments = [np.zeros_like(p) for p in params]
    variances = [np.zeros_like(p) for p in params]
    rows = np.arange(len(x))
    losses = []
    # Fixed training budget; test labels never drive checkpoint selection.
    for step in range(1,1801):
        w1,b1,w2,b2 = params
        hidden = np.tanh(x@w1+b1)
        logits = hidden@w2+b2
        z = logits[rows,heads]
        prob = 1/(1+np.exp(-np.clip(z,-40,40)))
        dz = np.zeros_like(logits)
        dz[rows,heads] = (prob-y)/len(x)
        dh = (dz@w2.T)*(1-hidden*hidden)
        gradients = [x.T@dh+1e-5*w1,dh.sum(0),hidden.T@dz+1e-5*w2,dz.sum(0)]
        for i,(p,g) in enumerate(zip(params,gradients)):
            moments[i] = .9*moments[i]+.1*g
            variances[i] = .999*variances[i]+.001*g*g
            p -= .015*(moments[i]/(1-.9**step))/(np.sqrt(variances[i]/(1-.999**step))+1e-8)
        if step%300 == 0:
            losses.append({'step':step,'bce':float(np.mean(np.logaddexp(0,z)-y*z))})
    np.savez(output_dir/'weights.npz', **dict(zip(('w1','b1','w2','b2'),params)))
    model=Model(output_dir/'weights.npz')
    def evaluate(data):
        by_head = {}
        for head,name in [(0,'yes_no'),(1,'option_utility')]:
            subset=[r for r in data if r[2]==head]
            correct=0
            for question,label,_ in subset:
                logit=(np.tanh(encode(question)@model.w1+model.b1)@model.w2+model.b2)[head]
                correct+=int((logit>=0)==bool(label))
            by_head[name]={'correct':correct,'total':len(subset),'accuracy':correct/len(subset)}
        return by_head
    pair_correct = sum(model.rank([p['worse'],p['better']], 'Considering my background, '+p['context'])['selected']==p['better'] for p in pairs)
    report={'parameters':model.parameter_count,'float32_weight_bytes':model.parameter_count*4,
            'seed':42,'steps':1800,'train':evaluate(train),'held_out_template':evaluate(test),
            'held_out_template_pairwise':{'correct':pair_correct,'total':len(pairs)},
            'loss':losses,
            'limitations':'Synthetic assistant-written labels from supplied memory. Held-out templates reuse known propositions and option descriptions. Not user validation or unseen-topic evaluation. No food tastes or named-person preference order supplied.'}
    (output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output_dir/'training_data.jsonl').open('w') as out:
        for split,data in [('train',train),('test',test)]:
            for question,label,head in data:
                out.write(json.dumps({'text':question,'label':label,'head':head,'split':split})+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir)
