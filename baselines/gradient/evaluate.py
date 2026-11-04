"""Evaluate frozen weights; unknown ground truth is never scored as correct."""
import hashlib
import json
from collections import defaultdict
from model import ROOT, Model

def run():
    model=Model()
    benchmark=json.loads((ROOT/'benchmark_100.json').read_text())
    assert len(benchmark)==100
    results=[]
    domains=defaultdict(lambda:dict(total=0,labeled=0,correct=0,unlabeled=0))
    for item in benchmark:
        if item['mode']=='yes_no':
            output=model.predict(item['question'])
            predicted=output['answer']=='yes'
        else:
            output=model.rank(item['options'],item['question'])
            predicted=[r['option'] for r in output['ranking']]
        correct=None if item['expected'] is None else predicted==item['expected']
        results.append(dict(item,output=output,correct=correct))
        stats=domains[item['domain']]
        stats['total']+=1
        if correct is None:
            stats['unlabeled']+=1
        else:
            stats['labeled']+=1
            stats['correct']+=int(correct)
    for stats in domains.values():
        stats['accuracy']=stats['correct']/stats['labeled'] if stats['labeled'] else None
    labeled=sum(s['labeled'] for s in domains.values())
    correct=sum(s['correct'] for s in domains.values())
    report={'total':len(results),'labeled':labeled,'correct':correct,'accuracy_on_labeled':correct/labeled,
            'unlabeled':len(results)-labeled,'domains':dict(domains),
            'weights_sha256':hashlib.sha256((ROOT/'weights.npz').read_bytes()).hexdigest(),
            'evaluation_kind':'Single fixed benchmark written after initial training design; not independent, not user-confirmed, not used in gradients or checkpoint selection.',
            'results':results}
    (ROOT/'evaluation_100.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 100-question evaluation','',
           f"Executed all 100 items. {correct}/{labeled} labeled items correct; {100-labeled} have no ground truth and are not scored.",
           '', 'Labels are assistant interpretations of explicit history or stated professional fit, not user-confirmed answers. No food, music, political or general country-preference labels were invented.',
           '', '| Domain | Executed | Labeled | Correct | Accuracy |','|---|---:|---:|---:|---:|']
    for d,s in domains.items():
        acc='N/A' if s['accuracy'] is None else f"{s['accuracy']:.1%}"
        lines.append(f"| {d} | {s['total']} | {s['labeled']} | {s['correct']} | {acc} |")
    lines+=['','Scores are not calibrated. An apparently strong score on an unlabeled item is not evidence about Yossi. These items expose the need for personal labels.','',
            'Full inputs, labels, provenance, scores, and outputs: `evaluation_100.json`. Benchmark never participates in training.',
            '', '## All predictions','', '| ID | Domain | Question | Prediction | Result |','|---|---|---|---|---|']
    for r in results:
        prediction=r['output'].get('answer') or ' > '.join(x['option'] for x in r['output']['ranking'])
        verdict='unlabeled' if r['correct'] is None else ('correct' if r['correct'] else 'incorrect')
        lines.append(f"| {r['id']} | {r['domain']} | {r['question']} | {prediction} | {verdict} |")
    (ROOT/'EVALUATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))

if __name__=='__main__':
    run()
