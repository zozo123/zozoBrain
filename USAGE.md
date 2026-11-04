# One file between you, Codex, and Claude

Use `brain.py` for the everyday workflow. A `.brain` file contains the actual
neural tensors, their feature schema, and verified contribution history. It is
data, so another compatible runtime can load it without executing your source.

## Create, inspect, and decide

Run from the cloned repository, with uv installed:

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py inspect personal/me.brain
uv run --locked brain.py decide personal/me.brain examples/personal.json
```

The first result is an abstention: blank brains have no utility, veto, or evidence
outputs. Inspection shows named feature rows, their actual output weights,
contexts, contributors, and short fingerprints. Programs can add `--json` to
`inspect` and `decide` for complete machine-readable output and full fingerprints.

The sample input compares `low_cost` and `high_cost` in `general`. To encode your
own explicit preference, assign the actual tensor rows. This example assumes you
have stated that lower cost is preferable; the numerical strength is a hypothesis:

```bash
uv run --locked brain.py set personal/me.brain --tensor w2 --index 0 --values 3 0 1 --reason "I explicitly prefer lower cost" --harness codex
uv run --locked brain.py set personal/me.brain --tensor w2 --index 1 --values -3 0 1 --reason "I explicitly prefer lower cost" --harness claude
uv run --locked brain.py decide personal/me.brain examples/personal.json
```

Now the model selects `A` and retains both sources. Output columns are utility,
veto activation, and evidence activation. Evidence is explicit; setting utility
alone does not supply it. Missing preference evidence among multiple feasible
options, a tie, or no feasible candidate produces abstention. If vetoes leave one
feasible option, constraint elimination can select it without ranking evidence.
`--force` resolves unknowns/ties without overriding vetoes.

## Inspect or edit any tensor

```bash
uv run --locked brain.py inspect personal/me.brain --tensor w2 --index 0
uv run --locked brain.py set personal/me.brain --tensor w2 --index 0 0 --values 6 --reason "Reference to my explicit correction"
```

One index selects a row of a matrix; two select a scalar. The value count must
match the selection exactly. Vectors use one index. `set` supports all four tensors
and performs direct assignments, with no preference compiler or optimizer.

Input feature order and hidden-neuron meaning are different concepts: the blank
template initially aligns attribute `i` with hidden neuron `i`. Arbitrary `w1`/`b1`
edits can change that circuit; inspect its connections when making such changes.

Default edits update the same file atomically under a writer lock. The revision
check rejects a stale update instead of silently erasing another agent's work.
Use `--output personal/proposed.brain` for a separate copy. Existing alternate
outputs are not overwritten. Save each agent's result separately for parallel work.

## Hand off to another agent

```bash
uv run --locked brain.py prompt personal/me.brain
```

Paste the emitted task into the next harness. It includes exact runtime/data
paths, full state fingerprints, and direct edit/decision commands. Update the
paths if you move the file. The next agent reads the current brain, applies
additional justified tensor edits, and carries forward the existing history.
Preferences come from authorized context; no proprietary chat-history API is
bundled. Handoff text is an instruction template, not automatic synchronization.

## Merge parallel contributions

Each harness works from the same original `.brain`, saves its own edited copy,
and exports its numerical changes with a source/evidence reference:

```bash
uv run --locked brain.py diff personal/base.brain personal/codex.brain --harness codex --reason "Reference to the explicit instruction" --output personal/codex.patch.json
uv run --locked brain.py diff personal/base.brain personal/claude.brain --harness claude --reason "Reference to the explicit instruction" --output personal/claude.patch.json
uv run --locked brain.py merge personal/base.brain personal/codex.patch.json personal/claude.patch.json --output personal/combined.brain
```

Edits to different entries combine; agreeing edits retain both sources without
double-counting; conflicting values fail without replacing your brain. For an
intentional correction, start from the current accumulated state. Patch source
and evidence are caller assertions: review the actual changes, not just the name.
`diff` captures the selected actor's numerical contribution; it does not import
all history from an arbitrary edited branch. Keep the original branch files when
their separate histories matter.

## Low-latency inference

```bash
uv run --locked brain.py serve personal/me.brain
```

Keep that process alive. Send one complete JSON object per line and read one JSON
response per line. It loads and verifies the brain once at startup, flushes each
response, and continues after malformed requests. Restart to load a newer saved
brain. The general request/response schema is in [HARNESS.md](HARNESS.md).

Question text is metadata; callers supply structured attributes and contexts.
The network cannot infer arbitrary tastes from names or chat text. Missing
feature coverage remains explicit rather than being replaced with an LLM guess.

## Existing source/NPZ users

Run `pack` from the runtime directory whose trusted local `weights.py` defines
your snapshot's exact feature layout:

```bash
uv run --locked brain.py pack --weights combined.npz --manifest combined.manifest.json --output personal/me.brain --name me
```

For the shipped example, `--weights weights.npz` works without a prior manifest;
it creates a verified initial state with no claimed contributors. `pack` is the
explicit source-reading migration command. Other bundle commands use only the
embedded data. Existing `model.py`, `harness.py`, NPZ exports, and JSON patches
continue to work. Source changes do not silently replace an accumulated bundle.

## Your own layout

`init --layout layout.json` accepts declarative `attributes` (32 unique names),
`contexts` (16 unique names), and `binary_attributes` (a subset of attributes).
It creates blank outputs with each attribute gated by the first context. Rename
slots before accumulating history; changing a schema makes old patches
incompatible. The Python API accepts the same layout dictionary. For editable
source plus a portable runtime, use `init_personal.py` and its generated README.
