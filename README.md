# zozoBrain

**Your decisions as a 2,499-parameter neural network that Codex, Claude Code, or any agent edits directly, weight by weight, with verified history.**

![Two agents edit one brain](assets/demo.gif)

```bash
git clone https://github.com/zozo123/zozoBrain.git && cd zozoBrain
uv run --locked stack_demo.py
```

- **One small file.** A `.brain` (about 2 KB) holds the weights, the feature names, and every edit with its evidence. No training, API key, or network.
- **Any agent can edit it.** `brain.py prompt` prints a handoff. The agent reads the labeled weights and assigns tensor entries directly.
- **Edits accumulate safely.** Every load replays and verifies the history. Conflicting parallel edits are refused, never averaged.
- **It knows what it doesn't know.** With no stated preference, it abstains.

## Your brain in 60 seconds

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py decide personal/me.brain examples/personal.json
uv run --locked brain.py set personal/me.brain --tensor w2 --index 0 --values 3 0 1 --reason "I prefer lower cost" --harness codex
uv run --locked brain.py set personal/me.brain --tensor w2 --index 1 --values -3 0 1 --reason "I prefer lower cost" --harness claude
uv run --locked brain.py decide personal/me.brain examples/personal.json
```

The first decision abstains. After the two edits it prints `Decision: A (Lower-cost option)` and `Contributors: codex, claude`. `brain.py inspect personal/me.brain` shows the labeled weights and the verified history.

## Hand it to an agent

```bash
uv run --locked brain.py prompt personal/me.brain
```

Paste the output into Codex, Claude Code, Copilot, or any agent with a shell, and add what you want encoded. The agent inspects the labeled rows, edits them in place with `set`, and checks decisions with a request file. For the next agent, run `prompt` again. [Agents and parallel edits](HARNESS.md)

**Tested end to end with real agents.** Starting from a fresh clone, GitHub Copilot CLI (GPT-5.6) encoded a cost preference. Claude Code (Opus 5.5) then added a hard rule that holds in all 16 contexts and an offline tie-breaker to the same file. Both agents appear in the verified history, and every check passed. [Run log](HARNESS.md#tested-with-real-agents)

## How it works

```text
32 option attributes + 16 contexts -> 48 ReLU neurons -> utility, veto, evidence
```

In a new brain, neuron `i` fires when attribute `i` is present in the first context, and row `w2[i]` holds its `[utility, veto, evidence]`. A positive veto removes an option. The best utility wins only if every remaining option has evidence; otherwise the brain abstains. Every one of the 2,499 entries in `w1`, `b1`, `w2`, `b2` is editable. [Weight guide](WEIGHTS.md)

| Command | Does |
|---|---|
| `init FILE` | Create a blank brain |
| `inspect FILE` | Show labeled weights, contexts, history |
| `set FILE --tensor T --index ... --values ...` | Assign a row or scalar in place, with evidence |
| `decide FILE REQUEST.json` | Decide, or abstain |
| `serve FILE` | JSON Lines decisions for programs |
| `prompt FILE` | Print an agent handoff |
| `diff BASE COPY` / `merge BASE PATCH...` | Combine parallel copies; conflicts fail |

[Full CLI guide](USAGE.md) · [File formats](FORMATS.md) · [Speed](PERFORMANCE.md)

## The shipped example

`uv run --locked demo.py` runs a hand-authored policy for Yossi Eliaz: no flights on Shabbat, a one-page CV, concise and verified answers, and an abstention on food, which was never stated. 100 policy probes pass ([evaluation](EVALUATION.md), [question bank](QUESTION_BANK.md)). They check the encoded rules, not how well it predicts a person.

Weights are authored, not learned: the brain encodes what you state and nothing more. The questions are metadata; callers supply structured attributes. An earlier [gradient-trained baseline](baselines/gradient/README.md) is kept for comparison.

## Develop

```bash
uv run --locked python -m unittest
uv run --locked weights.py --check
uv run --locked evaluate.py --check
```

Python 3.11 to 3.13, NumPy only. CI runs on Linux, macOS, and Windows.

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. MIT license.
