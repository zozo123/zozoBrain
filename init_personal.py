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
    readme = '''# NAME: one portable brain

Start blank. No utility, veto, evidence, or Yossi preferences are inherited.
Use the unified CLI with one .brain file as the active model and history.
Run these commands from inside this generated folder:

```bash
uv run --locked brain.py init NAME.brain --name NAME
uv run --locked brain.py inspect NAME.brain
uv run --locked brain.py decide NAME.brain decision.json
```

The initial decision abstains. Inspect the ordered attribute/context names and
actual tensor values with:

```bash
uv run --locked brain.py inspect NAME.brain --json
uv run --locked brain.py inspect NAME.brain --tensor w2 --index 0
```

## Edit the actual weights

The w2 columns are utility, veto activation, and evidence activation. The blank
network wires neuron 0 to low_cost and neuron 1 to high_cost in the general
context. Only if you explicitly prefer lower cost, directly set these tensor rows:

```bash
uv run --locked brain.py set NAME.brain --tensor w2 --index 0 --values 3 0 1 --reason "Reference to my explicit lower-cost preference" --harness manual
uv run --locked brain.py set NAME.brain --tensor w2 --index 1 --values -3 0 1 --reason "Reference to my explicit lower-cost preference" --harness manual
uv run --locked brain.py decide NAME.brain decision.json
```

The sample then selects A. Each set command directly assigns the numeric values;
there is no preference compiler or training. Edit individual entries with two
indices, for example --tensor w2 --index 0 0 --values 3. Utilities are authored
hypotheses, not learned estimates or probabilities. Evidence 1 means you supplied
a preference; changing only utility on a blank profile leaves evidence at zero.
Do not guess personal preferences to make the model answer.

A positive veto output excludes an option. Hard constraints also need their input
name in the embedded binary_attributes metadata; explicitly_disallowed is already
listed, with its veto initially zero. Coordinate feature-schema changes explicitly.
A forced decision never overrides a veto. The input has 32 attributes and 16
contexts; w1 and b1 wire inputs to hidden neurons. All 2,499 entries can be edited.

set atomically updates the input file and preserves its contribution history.
Use --output other.brain to keep the original. Updates check the loaded revision
under a write lock; concurrent stale writes fail. A .brain.lock sidecar is only
local write coordination and is not required when moving the brain.

## Use it from any agent

```bash
uv run --locked brain.py prompt NAME.brain
uv run --locked brain.py serve NAME.brain
```

The first command prints a task with this brain's exact path, state fingerprints,
and direct tensor-edit command. Give that task to Codex, Claude Code, or another
shell-capable agent. Use only context you authorize, explicit preferences, and
short evidence references. The second command accepts one structured JSON object
per stdin line and returns one JSON response per line. Restart a running process
after editing the brain. For a single machine-readable decision, add --json to
decide. The supplied decision.json is a request example.

The .brain file embeds the ordered feature layout, four tensors, and verified
contribution history. It is data only: inspecting, editing, deciding, serving,
diffing, and merging do not execute an embedded weights.py. Copy just this file
to another compatible runtime to continue. This generated folder also includes
its own runtime, pyproject.toml, uv.lock, and MIT notice, so you can move the whole
folder elsewhere. With uv installed, the same commands work there. Dependencies
require internet once unless cached; decisions run locally afterward.

For parallel contributions, give each agent the same unchanged base file. After
they directly edit their own copies, capture and combine explicit JSON patches:

```bash
uv run --locked brain.py diff base.brain codex.brain --harness codex --reason "Reference to my explicit preference" --output codex.patch.json
uv run --locked brain.py diff base.brain claude.brain --harness claude --reason "Reference to my next explicit preference" --output claude.patch.json
uv run --locked brain.py merge base.brain codex.patch.json claude.patch.json --output combined.brain
```

These commands assume those agent-edited files exist. Compatible changes combine;
conflicting values or schemas fail. Sequential set edits accumulate directly in
one file; use patches for independent parallel copies. Preserve abstentions and
verify behavior using your own structured held-out decisions. Yossi's root
benchmark does not evaluate your profile. Question text is metadata; supply the
option attributes and contexts yourself.

## Source and NPZ compatibility

weights.py remains directly editable for the original source/NPZ workflow. Its
commented w2 row examples start disabled. The old model.py command reads those
source assignments, whereas brain.py decide reads the selected .brain file.
Source edits never silently overwrite an accumulated .brain model. To explicitly
pack a source snapshot and this trusted local source layout into a new brain:

```bash
uv run --locked weights.py --output source.npz
uv run --locked brain.py pack --weights source.npz --output from-source.brain
```

pack is the explicit bridge that reads trusted local weights.py for its layout.
The original model.py, harness.py, edit_weights.py, and weight_patches.py commands
remain available. See the upstream [weight guide](https://github.com/zozo123/zozoBrain/blob/main/WEIGHTS.md),
[portable formats](https://github.com/zozo123/zozoBrain/blob/main/FORMATS.md), and
[harness guide](https://github.com/zozo123/zozoBrain/blob/main/HARNESS.md).

## Private defaults

The parent repository gitignores personal/. This folder's .gitignore excludes
*.brain, *.brain.lock, NPZ snapshots, contribution patch/manifest JSON, and the
contributions/ directory. Store private evidence there. Bundles and patches can
contain personal preferences and evidence; publish them only deliberately.
Gitignore is a commit default, not access control. Existing profiles and initial
brain files are never overwritten by init.
'''
    (target / 'README.md').write_text(readme.replace('NAME', name), encoding='utf-8')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    print(create(args.name))
