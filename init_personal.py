"""Create a private, standalone policy folder without inheriting Yossi's preferences."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parent

def create(name, parent=None):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',name):
        raise ValueError('Use a short name containing letters, numbers, underscore or hyphen')
    target=(parent or ROOT/'personal')/name
    target.mkdir(parents=True,exist_ok=False)
    for filename in ['model.py','harness.py']:
        shutil.copy2(ROOT/filename,target/filename)
    shutil.copy2(ROOT/'templates/personal_weights.py',target/'weights.py')
    contexts=['general']+[f'context_{i}' for i in range(2,17)]
    names=['low_cost','high_cost','offline','explicitly_disallowed']+[f'attribute_{i}' for i in range(5,33)]
    rules=[{'name':n,'context':'general','utility':0,'veto':False} for n in names]
    (target/'policy.json').write_text(json.dumps({'owner':name,'contexts':contexts,'rules':rules},indent=2)+'\n')
    (target/'decision.json').write_text(json.dumps({'question':'Which option would I choose?', 'contexts':['general'],
        'options':[{'id':'A','label':'Lower-cost option','attributes':{'low_cost':1}},
                   {'id':'B','label':'Higher-cost option','attributes':{'high_cost':1}}]},indent=2)+'\n')
    (target/'README.md').write_text('''# Your own policy

All utility weights and veto flags start at zero. No Yossi preferences are inherited.
Edit policy.json: keep 32 unique rules and 16 unique contexts; rename unused slots as needed.
Set utility values from your own explicit preferences, not guesses. For example, only if
you prefer lower cost, set low_cost utility to 3 and high_cost utility to -3.
Set veto true only for an actual hard constraint; its input must be binary.
Run from the repository root: uv run --locked personal/NAME/model.py personal/NAME/decision.json
Replace NAME with this directory's name. Initially this abstains; after the example edits it selects A.
Use uv run --locked personal/NAME/weights.py to export weights.npz. No training occurs.
Supply your own held-out decisions before changing weights. The root benchmark is about
Yossi and is not an evaluation of you. This folder is gitignored by the parent repository.
'''.replace('NAME',name))
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args=parser.parse_args()
    print(create(args.name))
