"""Create a private profile with directly editable weights and no inherited preferences."""
import argparse
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent


def create(name, parent=None):
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name):
        raise ValueError('Use a short name containing letters, numbers, underscore or hyphen')
    target = (Path(parent) if parent is not None else ROOT / 'personal') / name
    # Check required runtime files before creating the destination.
    runtime = [
        'model.py', 'harness.py', 'network.py', 'edit_weights.py', 'weight_patches.py',
        'brain.py', 'brain_bundle.py',
        'pyproject.toml', 'uv.lock', 'LICENSE',
    ]
    sources = [ROOT / filename for filename in runtime] + [ROOT / 'templates/personal_weights.py']
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=False)
    for filename in runtime:
        shutil.copy2(ROOT / filename, target / filename)
    shutil.copy2(ROOT / 'templates/personal_weights.py', target / 'weights.py')
    decision = {
        'question': 'Which option would I choose?',
        'contexts': ['general'],
        'options': [
            {'id': 'A', 'label': 'Lower-cost option', 'attributes': {'low_cost': 1}},
            {'id': 'B', 'label': 'Higher-cost option', 'attributes': {'high_cost': 1}},
        ],
    }
    (target / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
    (target / '.gitignore').write_text(
        '__pycache__/\n*.pyc\n.venv/\ncontributions/\n*.brain\n*.brain.lock\n*.npz\n*.patch.json\n*.stack.json\n*.manifest.json\n',
        encoding='utf-8',
    )
    readme = '''# NAME: a portable brain

A blank decision brain with its own runtime. Nothing is inherited; it abstains
until you state a preference. Run these from inside this folder:

```bash
uv run --locked brain.py init NAME.brain --name NAME
uv run --locked brain.py decide NAME.brain decision.json
uv run --locked brain.py set NAME.brain --tensor w2 --index 0 --values 3 0 1 --reason "I prefer lower cost" --harness manual
uv run --locked brain.py set NAME.brain --tensor w2 --index 1 --values -3 0 1 --reason "I prefer lower cost" --harness manual
uv run --locked brain.py decide NAME.brain decision.json
uv run --locked brain.py inspect NAME.brain
```

The first decision abstains and the last selects A. Only make the two edits if
you actually prefer lower cost. w2 rows are [utility, veto, evidence] for the
attribute of the same index; inspect shows the names, contexts, and history.

## With an agent

```bash
uv run --locked brain.py prompt NAME.brain
```

Paste the output and your statements into Codex, Claude Code, or any agent with a
shell. It edits the file in place and keeps the history. Run prompt again for the
next agent. The handoff also explains rules that hold in one context only. For
agents working at the same time, give each a copy and its own free neurons
(prompt COPY.brain --neurons 40-43), then combine with brain.py diff and merge.
Conflicts fail.
For programs, `brain.py serve NAME.brain` answers one JSON request per line.

## Moving it

NAME.brain is data only: copy it to any zozoBrain runtime. To move everything,
copy this folder; with uv installed the same commands work anywhere. weights.py
holds the source form (`uv run --locked weights.py --output source.npz`, then
`brain.py pack --weights source.npz --output from-source.brain`).

.gitignore keeps brains, snapshots, patches, and contributions/ out of git. They
can contain personal preferences and evidence, so publish them only deliberately.
Guides: https://github.com/zozo123/zozoBrain#readme
'''
    (target / 'README.md').write_text(readme.replace('NAME', name), encoding='utf-8')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    try:
        print(create(args.name))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'error: {exc}\n')
