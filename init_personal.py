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
        '__pycache__/\n*.pyc\n.venv/\ncontributions/\n*.npz\n*.patch.json\n*.stack.json\n*.manifest.json\n',
        encoding='utf-8',
    )
    readme = '''# Your own direct weights

This profile starts with zero utility, veto, and evidence outputs. Its first decision
abstains. No Yossi preferences are inherited. There is no policy.json or training step.

## Edit the actual tensors

Open weights.py and edit the numerical assignments in authored_weights(). The w2
columns are utility, veto activation, and evidence activation. For the supplied input,
neuron 0 represents low_cost and neuron 1 high_cost, both in the general context.
Only if you explicitly prefer lower cost, uncomment these assignments:

```python
w2[0] = [3, 0, 1]
w2[1] = [-3, 0, 1]
```

Run from the repository root:

```bash
uv run --locked personal/NAME/model.py personal/NAME/decision.json
```

That command uses the current source assignments on every invocation. After those
example edits it selects A. Utility magnitudes are your authored hypotheses; evidence
1 means you supplied a preference. Scores are not probabilities. Do not add guessed
preferences to make the system answer.

Keep 32 unique attribute names and 16 unique context names. Attribute slots describe
options. The w1 connections and b1 biases determine their context gating. Edit them
directly for your own domains; the template wires attributes to the general context.
A positive veto output excludes an option. When authoring a hard constraint, also add
its attribute name to BINARY_ATTRIBUTES so fractional inputs are rejected. The
explicitly_disallowed attribute is already listed there, but its veto weight starts
at zero. A forced decision never overrides a veto.

## Export or load an explicitly edited artifact

```bash
uv run --locked personal/NAME/weights.py
uv run --locked personal/NAME/weights.py --check
uv run --locked personal/NAME/model.py personal/NAME/decision.json --weights personal/NAME/weights.npz
uv run --locked personal/NAME/edit_weights.py inspect personal/NAME/weights.npz
uv run --locked personal/NAME/edit_weights.py set personal/NAME/weights.npz --tensor w2 --index 0 0 --value 6 --output personal/NAME/my_weights.npz
```

weights.npz is a tensor snapshot. Source edits affect the default run; artifact edits
affect runs that explicitly select that artifact with --weights. Exporting again
replaces the snapshot with the source assignments. The Python API likewise accepts
DecisionModel(weights=path_to_npz). For the artifact editor example to express a
preference, first author the source example's evidence entries; changing only a
utility entry on a blank profile leaves evidence at zero and the model abstains.

## Run the profile on its own

The folder includes its own pyproject.toml, uv.lock, model, harness, tensor runtime,
editor, weight-patch tools, and MIT license notice. You can move or copy the whole folder elsewhere.
With uv installed, run from inside that folder:

```bash
uv run --locked model.py decision.json
uv run --locked weights.py
uv run --locked harness.py
```

NumPy installation requires internet access once unless it is already cached.
Inference then runs locally. The harness accepts one structured JSON object per
stdin line. Question text is metadata; you must supply the option attributes and
contexts yourself. Collect your own held-out decisions before editing weights;
Yossi's benchmark does not measure your profile.

## Carry this brain to another agent

Move the whole folder, including the active snapshot, patches, and matching
manifest. Use this task in Codex, Claude Code, or another shell-capable agent:

```text
Use my profile at /absolute/path/to/this-folder. Read README.md, weights.py, and
the active manifest. Use uv run --locked and the local uv.lock.
Use only explicit preferences, constraints, and corrections from context I have
authorized. Record short evidence references and distinguish stated preferences
from your hypotheses about numerical strengths. Report missing feature coverage.
Load the active NPZ and edit w1, b1, w2, b2 directly. Preserve the feature layout;
coordinate schema changes separately. Do not train, fit, compile preferences, or
average weights. Add evidence output for authored preferences and binary input
metadata plus veto output for hard constraints. Test structured decision cases.
Keep the input snapshot unchanged. Save an edited snapshot, capture its diff with
your harness name and unique contribution ID, and stack it against the active
snapshot. Use --resume-manifest when provenance already exists. Keep the resulting
manifest and verify inference with --weights and --manifest. Preserve abstentions.
Report actual changed entries, their evidence, checks, and remaining gaps.
```

From inside this folder, export the initial base once:

```bash
uv run --locked weights.py --output base.npz
```

After an agent directly edits that base and saves edited.npz, capture and apply its
contribution (replace the harness, ID, and evidence with the actual source):

```bash
uv run --locked weight_patches.py diff base.npz edited.npz --harness codex --id codex-001 --evidence "Reference to my explicit preference" --output codex.patch.json
uv run --locked weight_patches.py stack base.npz codex.patch.json --output active.npz --manifest active.manifest.json
uv run --locked model.py decision.json --weights active.npz --manifest active.manifest.json
```

For a later contribution, the next agent edits active.npz into next-edited.npz.
Capture its diff against active.npz, then retain the earlier history when stacking:

```bash
uv run --locked weight_patches.py diff active.npz next-edited.npz --harness claude --id claude-001 --evidence "Reference to my next explicit preference" --output claude.patch.json
uv run --locked weight_patches.py stack active.npz claude.patch.json --resume-manifest active.manifest.json --output combined.npz --manifest combined.manifest.json
uv run --locked model.py decision.json --weights combined.npz --manifest combined.manifest.json
```

The response reports both contributing harnesses. Keep combined.npz and its manifest
as the next session's active state. Patches with a different feature schema or
conflicting entry values fail instead of being averaged or silently overridden.
For details, see the upstream [weight guide](https://github.com/zozo123/zozoBrain/blob/main/WEIGHTS.md),
[portable formats](https://github.com/zozo123/zozoBrain/blob/main/FORMATS.md), and
[harness guide](https://github.com/zozo123/zozoBrain/blob/main/HARNESS.md).

The parent repository gitignores personal/ by default. This folder's own .gitignore
also excludes snapshots, *.patch.json, *.stack.json, *.manifest.json, and the
contributions/ directory. Store private agent contribution patches and their evidence
in contributions/. These files can contain personal preferences and quoted evidence;
publish them only deliberately. Gitignore is a commit default, not access control.
Existing profiles are never overwritten.
'''
    (target / 'README.md').write_text(readme.replace('NAME', name), encoding='utf-8')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    args = parser.parse_args()
    print(create(args.name))
