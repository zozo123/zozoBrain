# Edit the weights directly

The executable policy is four float32 tensors: `w1`, `b1`, `w2`, `b2`. There is no
training loop, optimizer, preference-fitting step, or JSON-to-weight compiler in
the main implementation. Both source edits and snapshot edits change these exact
tensors.

The recommended everyday interface is `brain.py set FILE.brain --tensor w2
--index 0 --values 3 0 1 --reason "Reference to an explicit preference"`. That
assigns the actual output row and saves the matching schema/history atomically.
Use two indices and one value to set a scalar; inspect actual values with
`brain.py inspect FILE.brain --tensor w2 --index 0`. [Complete guide](USAGE.md)

Source/NPZ workflows below remain available for authoring and migration. A source
edit affects the source-based interface; an accumulated `.brain` changes through
explicit bundle edits or a newly packed file.

## Source assignments

Default `DecisionModel()` reads the assignments in `weights.py`. In the shipped
policy, the first hidden neurons represent the named attributes. For example:

```python
w2[0] = [3, 0, 1]  # concise: utility, veto activation, evidence activation
w2[1] = [4, 0, 1]  # verified
```

Changing the actual entry `w2[0, 0]` changes the utility for an active concise
feature. A positive second output vetoes a candidate. A positive third output
marks preference evidence. All feasible options need modeled evidence before a
preference ranking is selected; a single feasible option can be selected by
constraint elimination. A utility weight alone on a blank profile does not invent
evidence. No answer is selected for a tie (float32 tolerance `rtol=1e-6`,
`atol=1e-7`) or missing evidence unless `--force` is requested.

You can edit any of the 2,499 entries, including hidden connections and biases.
Context-gated neurons initially compute `ReLU(attribute + context - 1)` through
explicit `w1` connections and `b1=-1`. The last 16 hidden units are available for
additional circuits. The names remain 32 attributes and 16 contexts; names and
input order are part of the schema. Add any hard constraint input to
`BINARY_ATTRIBUTES`, whose values must be exactly 0 or 1.

The default process loads source afresh, including rapid same-size file edits.
A running JSONL harness holds one weight snapshot: restart it after changing
source or selecting another artifact. A Python caller can also construct
`DecisionModel(weights=(w1, b1, w2, b2))` with directly edited arrays.

## Export, edit, and explicitly load a snapshot

Run from the repository root. These numerical edits are a mechanical example,
not new claims about Yossi's preferences:

```bash
uv run --locked weights.py --output base.npz
uv run --locked edit_weights.py inspect base.npz
uv run --locked edit_weights.py set base.npz --tensor w2 --index 0 0 --value 6 --output edited.npz
uv run --locked model.py examples/work.json --weights edited.npz
uv run --locked harness.py --weights edited.npz
```

The editor assigns exactly the specified tensor entry and prints the old value,
new value, and resulting fingerprint. Use a new output path to preserve the base.
For a vector entry, provide one index, e.g. `--tensor b2 --index 0 --value 0`.

`--weights` is explicit: editing `edited.npz` does not change default source
inference. Exporting with `weights.py` writes source values to the chosen path;
it does not read a previously edited snapshot. Archive loading requires exactly
the four named tensors, correct shapes, and finite float32 values; object/pickle
archives are rejected. Shapes cannot prove that two profiles use the same feature
names: carry the source layout with the snapshot, and use schema-bound patches
when combining edits.

## Verification

```bash
uv run --locked weights.py --check
uv run --locked python -m unittest -v
uv run --locked evaluate.py --check
```

`weights.py --check` checks the committed snapshot against source without writing.
After an intentional source change, export again and run `evaluate.py` to update
the report; `--check` subsequently verifies its exact contents and fingerprint.
The evaluator exits nonzero when a policy probe fails. Changing expectations to
hide a regression is not evidence of better personal prediction.

For a new person's policy, generate a blank profile first. The root policy and
its benchmark encode selected Yossi preferences. Authored utility magnitudes are
hypotheses, not calibrated estimates of personal trade-offs. Record independent
choices to evaluate those hypotheses.
