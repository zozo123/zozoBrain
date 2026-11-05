# Editing the weights

The policy is four float32 tensors with 2,499 entries in total. Nothing is trained or compiled: every edit assigns entries directly.

| Tensor | Shape | Meaning |
|---|---|---|
| `w1` | 48 × 48 | input → hidden. Inputs 0–31 are attributes, 32–47 are contexts |
| `b1` | 48 | hidden biases |
| `w2` | 48 × 3 | hidden → `[utility, veto, evidence]` |
| `b2` | 3 | output biases |

## The wiring of a new brain

Hidden neuron `j` computes `ReLU(sum_i input_i * w1[i, j] + b1[j])`: rows 0–31 of `w1` are attributes and rows 32–47 are contexts. `init` wires neuron `i` to attribute `i` in the first context (`general`): `w1[i, i] = 1`, `w1[32, i] = 1`, `b1[i] = -1`, so neuron `i` = ReLU(attribute_i + general − 1), an AND gate. Neurons 32–47 start free. Every output row starts at zero, so the brain abstains. `inspect` prints each neuron as the rule it computes.

| To encode | Edit |
|---|---|
| "I prefer lower cost" | `w2[low_cost] = [3, 0, 1]` and `w2[high_cost] = [-3, 0, 1]` |
| "Never pick explicitly disallowed" | `w2[explicitly_disallowed] = [0, 1, 0]`; the input must be binary |
| ...in every context | also `w1[32, i] = 0` and `b1[i] = 0` for that neuron |
| "When traveling, prefer offline" | free neuron 32: `w1[2, 32] = 1`, `w1[33, 32] = 1`, `b1[32] = -1`, `w2[32] = [10, 0, 1]` |
| A stronger preference | raise the utility, e.g. `w2[0, 0] = 6` |

Each option needs positive evidence to be ranked. That is why both cost rows carry evidence `1`. Utility alone never invents a preference: if no firing neuron gives an option evidence, the brain abstains rather than guess. An attribute left out of a request counts as 0. Utilities are authored strengths, not probabilities, and the outputs of all firing neurons add up: the traveling rule needs a utility above 6 to beat the ±3 cost pair. `decide` prints each utility as that sum.

A context is the same input for every option, so a weight from a context straight to the outputs would shift every option equally. A rule about a context therefore needs a hidden neuron that combines it with an attribute; [deep.py](deep.py) shows why.

```bash
uv run --locked brain.py set personal/me.brain --tensor w2 --index 3 --values 0 1 0 --reason "Hard rule I stated" --harness claude
uv run --locked brain.py set personal/me.brain --tensor w1 --index 32 3 --values 0 --reason "Rule applies in every context"
uv run --locked brain.py set personal/me.brain --tensor b1 --index 3 --values 0 --reason "Rule applies in every context"
uv run --locked brain.py set personal/me.brain --tensor w1 --index 2 32 --values 1 --tensor w1 --index 33 32 --values 1 --tensor b1 --index 32 --values -1 --tensor w2 --index 32 --values 10 0 1 --reason "When traveling I prefer offline options"
```

The last command makes four assignments one verified edit; repeat `--tensor`, `--index`, `--values` for each.

Binary inputs are fixed by the layout (`binary_attributes`). Choose them at `init --layout`.

## Source weights

The shipped Yossi policy is written as NumPy assignments in [weights.py](weights.py), for example `w2[0] = [3, 0, 1]  # concise`. `DecisionModel()` and `model.py` read that source. `weights.py --output FILE.npz` exports a snapshot, and `edit_weights.py` assigns single entries in NPZ files:

```bash
uv run --locked weights.py --output base.npz
uv run --locked edit_weights.py set base.npz --tensor w2 --index 0 0 --value 6 --output edited.npz
uv run --locked model.py examples/work.json --weights edited.npz
```

After changing `weights.py`, re-export `weights.npz` and run `evaluate.py` to refresh the report; `--check` then verifies both. Snapshots must contain exactly the four finite numeric tensors. Pickled objects are rejected.
