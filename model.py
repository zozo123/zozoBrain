"""Personal decision policy compiled into a 2,499-parameter neural network."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from weights import ATTRIBUTES,CONTEXTS,authored_weights,forward

def validate_question(question):
    if not isinstance(question,str) or not question.strip():
        raise ValueError('Question must be a nonempty string')
    if len(re.findall(r'\w+|[^\w\s]',question,re.UNICODE))>100:
        raise ValueError('Question exceeds 100 regex word/punctuation tokens')

class DecisionModel:
    def __init__(self):
        self.weights=authored_weights()
        self.parameter_count=sum(a.size for a in self.weights)
        assert self.parameter_count==2499

    def decide(self,question,contexts,options,force=False):
        validate_question(question)
        if not contexts or set(contexts)-set(CONTEXTS):
            raise ValueError('Supply known contexts: '+', '.join(CONTEXTS))
        if len(options)<2:
            raise ValueError('At least two options required')
        ids=[o['id'] for o in options]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Option ids must be unique nonempty strings')
        scored=[]
        for option in options:
            attrs=option.get('attributes',{})
            if set(attrs)-set(ATTRIBUTES):
                raise ValueError('Unknown attributes: '+str(set(attrs)-set(ATTRIBUTES)))
            x=np.zeros(48,dtype=np.float32)
            for attr,value in attrs.items():
                if not isinstance(value,(int,float)) or not np.isfinite(value) or value<0 or value>1:
                    raise ValueError('Attributes must be finite values in [0,1]')
                if attr in ['shabbat_flight','explicitly_disallowed'] and value not in (0,1):
                    raise ValueError('Hard constraint attributes must be 0 or 1')
                x[ATTRIBUTES.index(attr)]=value
            for context in contexts:
                x[32+CONTEXTS.index(context)]=1
            utility,veto,support=forward(x,self.weights).tolist()
            scored.append({'id':option['id'],'label':option.get('label',option['id']),
                           'utility':utility,'veto':veto>0,'modeled_evidence':support>0,
                           'active_attributes':[a for a,v in attrs.items() if v>0]})
        ranked=sorted((r for r in scored if not r['veto']),key=lambda r:-r['utility'])
        vetoed=[r for r in scored if r['veto']]
        status='decision'
        if not ranked:
            status='no_feasible_option'
        elif len(ranked)==1 and vetoed:
            status='only_feasible_option'
        elif not all(r['modeled_evidence'] for r in ranked):
            status='insufficient_preference_evidence'
        elif ranked[0]['utility']==ranked[1]['utility']:
            status='tie'
        selected=ranked[0]['id'] if ranked and (force or status in ['decision','only_feasible_option']) else None
        return {'decision':selected,'status':status,'forced':bool(force and status in ['tie','insufficient_preference_evidence']),
                'ranking':ranked,'vetoed':vetoed,
                'note':'Authored utility policy, not calibrated probability. Attributes and contexts are supplied by caller; question text is display metadata, not language understanding.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('decision_file',type=Path)
    parser.add_argument('--force',action='store_true',help='Resolve unknowns/ties by score then input order; never override veto')
    args=parser.parse_args()
    payload=json.loads(args.decision_file.read_text())
    print(json.dumps(DecisionModel().decide(payload['question'],payload['contexts'],payload['options'],args.force),indent=2))
