"""Run a transparent, reproducible tour of the Yossi decision policy."""
import json
from pathlib import Path
from model import DecisionModel

ROOT=Path(__file__).resolve().parent

def cases():
    return [
        ('Travel',json.loads((ROOT/'examples/travel.json').read_text())),
        ('Website',{'question':'Which production domain did Yossi Eliaz request?', 'contexts':['website'],
            'options':[{'id':'yossieliaz.netlify.app','attributes':{'netlify_domain':1}},
                       {'id':'www.yossieliaz.com','attributes':{'old_custom_domain':1}}]}),
        ('CV',{'question':'Which CV length did Yossi Eliaz request?', 'contexts':['cv'],
            'options':[{'id':'one page','attributes':{'single_page':1}},
                       {'id':'multiple pages','attributes':{'multi_page':1}}]}),
        ('Communication',{'question':'Which answer style does Yossi Eliaz prefer?', 'contexts':['communication'],
            'options':[{'id':'concise and verified','attributes':{'concise':1,'verified':1}},
                       {'id':'speculative','attributes':{'speculative':1}}]}),
        ('Food',json.loads((ROOT/'examples/food.json').read_text())),
    ]

if __name__=='__main__':
    brain=DecisionModel()
    print('zozoBrain — Yossi Eliaz decision-policy demo')
    print('2,499 total parameters | directly authored weights | local inference\n')
    for title,request in cases():
        out=brain.decide(request['question'],request['contexts'],request['options'])
        answer=out['decision'] if out['decision'] is not None else 'ABSTAIN'
        print(f"{title}: {answer} ({out['status']})")
    print('\nInputs are structured attributes, visible in demo.py. This is not a natural-language model.')
