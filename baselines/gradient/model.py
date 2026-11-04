"""YossiNN: 9,272 learned scalars; yes/no and contextual option ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
DIM, HIDDEN = 512, 18

def tokenize(question):
    if not isinstance(question, str):
        raise TypeError('Question must be a string')
    tokens = re.findall(r"\w+|[^\w\s]", question.lower(), re.UNICODE)
    if not tokens or len(tokens) > 100:
        raise ValueError('Expected 1–100 regex tokens (words and punctuation)')
    return tokens

def encode(question):
    tokens = tokenize(question)
    x = np.zeros(DIM, dtype=np.float32)
    for n in (1, 2, 3):
        for i in range(len(tokens)-n+1):
            feature = '\x1f'.join(tokens[i:i+n]).encode('utf-8')
            h = int.from_bytes(hashlib.blake2b(feature, digest_size=8).digest(), 'little')
            x[h % DIM] += 1 if (h >> 63) else -1
    return x / max(float(np.linalg.norm(x)), 1.)

class Model:
    def __init__(self, path=None):
        with np.load(path or ROOT / 'weights.npz', allow_pickle=False) as data:
            self.w1, self.b1, self.w2, self.b2 = [data[k] for k in ('w1','b1','w2','b2')]
        assert self.w1.shape == (512,18) and self.b1.shape == (18,)
        assert self.w2.shape == (18,2) and self.b2.shape == (2,)
        assert self.parameter_count == 9272

    @property
    def parameter_count(self):
        return sum(a.size for a in (self.w1,self.b1,self.w2,self.b2))

    def predict(self, question):
        z = (np.tanh(encode(question) @ self.w1 + self.b1) @ self.w2 + self.b2)[0].item()
        score = float(1 / (1 + np.exp(-np.clip(z, -40, 40))))
        return {'answer': 'yes' if score >= .5 else 'no', 'yes_score': score}

    def rank(self, options, question='Which option would I prefer?'):
        if len(options) < 2:
            raise ValueError('Provide at least two options')
        tokenize(question)
        results = []
        for index, option in enumerate(options):
            # Each context/option pair, including markers, has the same 100-token limit.
            x = encode(question + ' option: ' + option)
            z = (np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2)[1].item()
            results.append({'option': option, 'score': float(z), 'input_index': index})
        ranked = sorted(results, key=lambda r: -r['score'])
        # This is a narrow, explicit coverage hint, not an epistemic confidence model.
        unsupported = bool(re.search(r'\b(food|foods|eat|meal|restaurant|person|people|friend|date|music|political|politics|country)\b', question.lower()))
        return {'selected': ranked[0]['option'], 'ranking': ranked,
                'warning': ('No general food, music, political, country or interpersonal preference labels were available. Unless a specific documented constraint applies, this order is an unsupported guess.'
                            if unsupported else 'Experimental synthetic-label ranking; scores are not calibrated preference probabilities. Unknown options may rank arbitrarily.')}

    def decide(self, context, options):
        """Score choices; eligibility is explicitly supplied, never inferred by this model.

        Each option: id, label, optional description, eligible (bool, default True),
        optional exclusion_reason. Constraint filtering is not part of the neural net.
        """
        tokenize(context)
        if len(options)<2:
            raise ValueError('A decision needs at least two candidate options')
        ids=[o['id'] for o in options]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Option ids must be unique nonempty strings')
        scored,excluded=[],[]
        for option in options:
            if not isinstance(option.get('eligible',True),bool):
                raise ValueError('eligible must be boolean')
            if not isinstance(option.get('label'),str) or not option['label'].strip():
                raise ValueError('Each option needs a nonempty label')
            if not option.get('eligible',True):
                excluded.append({'id':option['id'],'reason':option.get('exclusion_reason','Marked ineligible by caller')})
                continue
            description=option.get('description','')
            text=option['label']+(' '+description if description else '')
            x=encode(context+' option: '+text)
            score=(np.tanh(x@self.w1+self.b1)@self.w2+self.b2)[1].item()
            scored.append({'id':option['id'],'label':option['label'],'utility_score':float(score)})
        ranked=sorted(scored,key=lambda r:-r['utility_score'])
        return {'decision':ranked[0]['id'] if ranked else None,
                'status':'experimental_prediction' if ranked else 'no_feasible_option',
                'ranking':ranked,'excluded':excluded,
                'score_margin':ranked[0]['utility_score']-ranked[1]['utility_score'] if len(ranked)>1 else None,
                'warning':'Utility and margin are not confidence. Missing personal preferences produce guesses. Eligibility is caller-supplied; no constraints are inferred. No explanatory reasoning is generated by this network.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='?', help='A yes/no question about Yossi; I/me refers to Yossi')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--options', nargs='+', help='Two or more options to select/rank')
    parser.add_argument('--weights',type=Path,help='Custom weights.npz')
    parser.add_argument('--decision-file',type=Path,help='JSON with context and options for a decision')
    args = parser.parse_args()
    model=Model(args.weights)
    if args.decision_file:
        if args.question or args.options:
            parser.error('--decision-file cannot be combined with question/options')
        payload=json.loads(args.decision_file.read_text())
        result=model.decide(payload['context'],payload['options'])
    else:
        if not args.question:
            parser.error('Supply a question or --decision-file')
        result = model.rank(args.options, args.question) if args.options else model.predict(args.question)
    print(json.dumps(result, indent=2) if args.details or args.options or args.decision_file else result['answer'])
