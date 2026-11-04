# Using brain.py

Run from the cloned repository with [uv](https://docs.astral.sh/uv/) installed. A `.brain` file holds the tensors, the feature layout, and the verified history. It is data only; loading it runs no code.

## Create, inspect, decide

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py inspect personal/me.brain
uv run --locked brain.py decide personal/me.brain examples/personal.json
```

A new brain abstains. `inspect` lists each attribute's `w2` row as `[utility, veto, evidence]`, the contexts, the binary (hard-rule) inputs, and the history. Add `--json` to `inspect` or `decide` for machine-readable output, or `--tensor w1 --index 32` to read any tensor slice.

A request names contexts and gives each option's attributes, with values in [0,1]:

```json
{"question": "Which option?", "contexts": ["general"],
 "options": [{"id": "A", "attributes": {"low_cost": 1}}, {"id": "B", "attributes": {"high_cost": 1}}]}
```

The question is display text; only contexts and attributes are scored. Unknown names or fields are errors.

## Edit

```bash
uv run --locked brain.py set personal/me.brain --tensor w2 --index 0 --values 3 0 1 --reason "I prefer lower cost" --harness codex
uv run --locked brain.py set personal/me.brain --tensor w2 --index 0 0 --values 6 --reason "Stronger, per my correction"
```

One index assigns a row, two assign a scalar. Each `set` updates the file atomically and appends a verified history record. A concurrent stale write is rejected rather than lost. `--output copy.brain` writes a separate copy and leaves the input unchanged.

Decision rules: a positive veto removes an option. The highest utility wins only if every remaining option has positive evidence. Otherwise the brain abstains, and ties abstain too. A single option left after vetoes is chosen. `--force` breaks ties and gaps but never overrides a veto.

## Hand off to an agent

```bash
uv run --locked brain.py prompt personal/me.brain
```

This prints a task with absolute paths, the current revision, how the weights are wired, an in-place edit command, and a sample request. Paste it with your statements into the next agent. See [HARNESS.md](HARNESS.md).

## Parallel copies

Each agent edits its own copy of the same base, then:

```bash
uv run --locked brain.py diff personal/base.brain personal/codex.brain --harness codex --reason "My cost statement" --output personal/codex.patch.json
uv run --locked brain.py diff personal/base.brain personal/claude.brain --harness claude --reason "My offline statement" --output personal/claude.patch.json
uv run --locked brain.py merge personal/base.brain personal/codex.patch.json personal/claude.patch.json --output personal/combined.brain
```

Different entries combine. Identical edits are kept once with both sources. Conflicting values fail, and nothing is written. `diff` refuses a copy that was not made from the base, so one agent's diff can never silently undo another agent's edits.

## Serve decisions

```bash
uv run --locked brain.py serve personal/me.brain
```

It reads one JSON request per line on stdin and writes one JSON response per line, `{"id": ..., "ok": true, "result": {...}}` or `{"ok": false, "error": ...}`. A bad line fails alone. Restart it to load a newer file.

## Custom layouts and existing NPZ snapshots

`init --layout layout.json` takes `attributes` (32 unique names), `contexts` (16), and `binary_attributes` (a subset of the attributes). Choose names before collecting history, because changing the layout makes old patches incompatible.

`pack --weights snapshot.npz [--manifest stack.json] --output me.brain` converts a source-based snapshot using the trusted local `weights.py` layout. `init_personal.py NAME` creates a private, movable profile folder with its own runtime, editable `weights.py`, and README. The older `model.py`, `harness.py`, `edit_weights.py`, and `weight_patches.py` commands still work. See [WEIGHTS.md](WEIGHTS.md) and [FORMATS.md](FORMATS.md).
