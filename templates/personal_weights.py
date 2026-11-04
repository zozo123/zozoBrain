"""Compile explicit policy.json values into weights; no fitting or training."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'policy.json').read_text())
CONTEXTS=POLICY['contexts']
RULES=POLICY['rules']
ATTRIBUTES=[r['name'] for r in RULES]
if len(CONTEXTS)!=16 or len(set(CONTEXTS))!=16 or len(RULES)!=32 or len(set(ATTRIBUTES))!=32:
    raise ValueError('Exactly 16 unique contexts and 32 unique attributes required')

def authored_weights():
    w1=np.zeros((48,48),dtype=np.float32)
    b1=np.zeros(48,dtype=np.float32)
    w2=np.zeros((48,3),dtype=np.float32)
    b2=np.zeros(3,dtype=np.float32)
    for i,rule in enumerate(RULES):
        context=rule['context']
        utility=rule['utility']
        veto=rule['veto']
        if not isinstance(utility,(float,int)) or not np.isfinite(utility) or abs(utility)>1000:
            raise ValueError('Utility must be finite and between -1000 and 1000')
        if not isinstance(veto,bool):
            raise ValueError('veto must be boolean')
        w1[i,i]=1
        if context is not None:
            w1[32+CONTEXTS.index(context),i]=1
            b1[i]=-1
        w2[i]=[utility,int(veto),int(utility!=0 or veto)]
    return w1,b1,w2,b2

def forward(x,weights=None):
    for i,rule in enumerate(RULES):
        if rule['veto'] and x[i] not in (0,1):
            raise ValueError('Hard constraint inputs must be binary')
    w1,b1,w2,b2=authored_weights() if weights is None else weights
    return np.maximum(0,x@w1+b1)@w2+b2

if __name__=='__main__':
    np.savez(ROOT/'weights.npz',**dict(zip(['w1','b1','w2','b2'],authored_weights())))
    print('Wrote weights.npz: 2,499 total parameters, no training')
