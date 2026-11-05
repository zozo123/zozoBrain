# Changelog

zozoBrain uses [semantic versioning](https://semver.org). From 1.0, the file formats, the JSON outputs, and the `brain.py` commands are stable within 1.x ([compatibility](FORMATS.md#compatibility)).

## 1.0.0

- Stable: `zozobrain.brain.v1`, `zozobrain.weight-patch.v1`, `zozobrain.weight-stack.v1`, the JSON of `serve`, `inspect --json` and `decide --json`, and the `brain.py` commands and flags.
- `prompt --neurons 40-47` gives a parallel agent its own free neurons, so parallel copies merge. The handoff splits the brain's free neurons for you and explains auditing.
- `merge` never lets a parallel copy re-apply an entry that an earlier patch in the same merge changed, such as one a revision reverted. Such merges used to depend on patch order.
- `decide` names tied options, says when `general` is missing and which rules are off, and says when the decision is the only option not vetoed.
- `inspect` counts the rules that drive decisions, names the neurons each history entry touched, and prints only the slice with `--tensor`. Huge weights no longer leak numpy warnings.
- `set --values` accepts negative numbers in scientific notation, such as `-1e19`.
- Commands stop quietly when a reader closes the pipe early, as in `inspect --history | head`.
- Errors name at most 10 unknown attributes or fields.
- Every documented shell command runs in CI, in order, in a fresh copy.

## 0.4.0 (2026-11-05)

One layer deeper: rules that depend on context, written into the hidden layer. Still no training.

- `deep.py`: in one matrix a context can never reorder the options; one hand-written neuron, `ReLU(offline + traveling - 1)`, can.
- `set` takes several assignments as one verified edit.
- `inspect` reads every neuron back as the rule it computes and marks rules that fire only with a context; `--history` shows every entry each edit changed.
- `decide` shows which neurons produced each utility and veto, and names options without evidence when it abstains.
- `merge` treats each hidden neuron as one rule: parallel copies that change the same neuron differently are refused.
- Blank brains name a `traveling` context.

## 0.3.0 (2026-11-04)

- One portable `.brain` file holds the weights, the feature layout, and the verified history. `brain.py` covers `init`, `inspect`, `set`, `decide`, `serve`, `diff`, `merge`, `pack`, and `prompt`.
- `diff` refuses a copy that was not made from the base. Loading never decompresses past declared sizes, and crafted NPY headers fail cleanly.
- 34 µs per decision.

## 0.2.0 (2026-11-04)

- Edit `w1`, `b1`, `w2`, `b2` directly; the JSON preference compiler is gone.
- Contributions accumulate as attributable JSON patches with verified provenance, and conflicting edits are surfaced.
