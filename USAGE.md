# Using brain.py

Run from the cloned repository with [uv](https://docs.astral.sh/uv/) installed. A `.brain` file holds the tensors, the feature layout, and the verified history. It is data only; loading it runs no code.

## Create, inspect, decide

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py inspect personal/me.brain
uv run --locked brain.py decide personal/me.brain examples/personal.json
```

A new brain abstains. `inspect` reads every neuron that drives decisions back as the rule it computes, such as `32 ReLU(offline + context:traveling - 1) -> [10, 0, 1]`, then lists the free neurons, the attributes and contexts by `w1` row, the binary (hard-rule) inputs, and the history. `inspect --history` lists every entry each edit changed. `decide` prints each option's utility as the sum of the neurons behind it. Add `--json` to `inspect` or `decide` for machine-readable output, or `--tensor w1 --index 32` to read just one tensor slice.

A request lists every active context and gives each option's attributes, with values in [0,1]. List `general` in every request: a new brain wires each attribute through it, so base preferences apply only when it is listed. Add specific contexts next to it, such as `["general", "traveling"]`:

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

One index assigns a row, two assign a scalar. Repeat `--tensor`, `--index`, `--values` to make several assignments one verified edit, for example a rule for one context in free neuron 32:

```bash
uv run --locked brain.py set personal/me.brain --tensor w1 --index 2 32 --values 1 --tensor w1 --index 33 32 --values 1 --tensor b1 --index 32 --values -1 --tensor w2 --index 32 --values 10 0 1 --reason "When traveling I prefer offline options"
```

Each `set` updates the file atomically and appends a verified history record. A concurrent stale write is rejected rather than lost. `--output copy.brain` writes a separate copy and leaves the input unchanged.

Decision rules: a positive veto removes an option. The highest utility wins only if every remaining option has positive evidence. Otherwise the brain abstains, and ties abstain too. A single option left after vetoes is chosen. `--force` breaks ties and gaps but never overrides a veto.

## Hand off to an agent

```bash
uv run --locked brain.py prompt personal/me.brain
```

This prints a task with absolute paths, the current revision, how the weights are wired, an in-place edit command, and a sample request. Paste it with your statements into the next agent. See [HARNESS.md](HARNESS.md).

## Parallel copies

Give each agent its own copy of the same base and its own free neurons, so their rules never land in the same neuron:

```bash
cp personal/me.brain personal/base.brain
cp personal/base.brain personal/codex.brain
cp personal/base.brain personal/claude.brain
uv run --locked brain.py prompt personal/codex.brain --neurons 40-43
uv run --locked brain.py prompt personal/claude.brain --neurons 44-47
```

Each agent edits its own copy, for example:

```bash
uv run --locked brain.py set personal/codex.brain --tensor w1 --index 0 40 --values 1 --tensor w1 --index 33 40 --values 1 --tensor b1 --index 40 --values -1 --tensor w2 --index 40 --values 2 0 1 --reason "When traveling, cost matters even more" --harness codex
uv run --locked brain.py set personal/claude.brain --tensor w1 --index 1 44 --values 1 --tensor w1 --index 33 44 --values 1 --tensor b1 --index 44 --values -1 --tensor w2 --index 44 --values -2 0 1 --reason "When traveling, I avoid high cost" --harness claude
```

When both are done:

```bash
uv run --locked brain.py diff personal/base.brain personal/codex.brain --harness codex --reason "When traveling, cost matters even more" --output personal/codex.patch.json
uv run --locked brain.py diff personal/base.brain personal/claude.brain --harness claude --reason "When traveling, I avoid high cost" --output personal/claude.patch.json
uv run --locked brain.py merge personal/base.brain personal/codex.patch.json personal/claude.patch.json --output personal/combined.brain
```

Edits to different hidden neurons combine. Identical edits are kept once with both sources. A parallel copy may not re-apply an entry that an earlier patch in the same merge changed, so list parallel copies before revisions built on them. Conflicting values fail, and so do two copies that change the same neuron differently, because merging a neuron entry by entry could build a rule neither agent wrote; nothing is written. `diff` refuses a copy that was not made from the base, so one agent's diff can never silently undo another agent's edits.

## Serve decisions

```bash
uv run --locked brain.py serve personal/me.brain
```

It reads one JSON request per line on stdin and writes one JSON response per line, `{"id": ..., "ok": true, "result": {...}}` or `{"ok": false, "error": ...}`. A bad line fails alone. Restart it to load a newer file.

## Custom layouts and existing NPZ snapshots

`init --layout layout.json` takes `attributes` (32 unique names), `contexts` (16), and `binary_attributes` (a subset of the attributes). Choose names before collecting history, because changing the layout makes old patches incompatible.

`pack --weights snapshot.npz [--manifest stack.json] --output me.brain` converts a source-based snapshot using the trusted local `weights.py` layout. `init_personal.py NAME` creates a private, movable profile folder with its own runtime, editable `weights.py`, and README. The older `model.py`, `harness.py`, `edit_weights.py`, and `weight_patches.py` commands still work. See [WEIGHTS.md](WEIGHTS.md) and [FORMATS.md](FORMATS.md).
