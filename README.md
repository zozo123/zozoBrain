# zozoBrain

**Your decisions as a 2,499-parameter neural network that Codex, Claude Code, or any agent edits directly, weight by weight, with verified history.**

![Two agents edit one brain](assets/demo.gif)

```bash
git clone https://github.com/zozo123/zozoBrain.git && cd zozoBrain
uv run --locked simplest.py     # the idea in one screen
uv run --locked deep.py         # one layer deeper: a rule that depends on context
uv run --locked stack_demo.py   # two agents, one brain
```

No training: you write the weights. [simplest.py](simplest.py) is the whole idea with one 4 × 3 matrix:

```python
W = np.zeros((4, 3))   # rows: low_cost, high_cost, offline, disallowed; columns: utility, veto, evidence
decide(W, options)     # abstain: nothing stated yet
W[0] = [3, 0, 1]       # write "I prefer lower cost"
W[1] = [-3, 0, 1]
W[3] = [0, 1, 0]       # write "never pick disallowed options"
decide(W, options)     # cheap (banned_cheap vetoed)
W[1, 0] = 5            # change one weight
decide(W, options)     # pricey
```

**One layer deeper.** A context such as traveling is the same input for every option, so in one matrix it shifts every utility equally and can never reorder the options. [deep.py](deep.py) adds a hidden layer and writes one neuron by hand: an AND gate made of two weights and a bias.

```text
                                  home    traveling
simplest.py's rules as 3 neurons  cheap   cheap
+ neuron 3 = traveling, +10       cheap   cheap        (shifts every option equally)
rewire neuron 3: + offline, -1    cheap   offline_kit  (one neuron, one context rule)

neuron 3 = ReLU(offline + traveling - 1) -> utility 10, veto 0, evidence 1
checked all 32 on/off inputs: veto iff disallowed, neuron 3 iff offline and traveling: holds
```

The real model is the same at 2,499 weights: 48 such neurons, each read back as the rule it computes, with every edit kept in a verified history.

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
uv run --locked brain.py set personal/me.brain --tensor w1 --index 2 32 --values 1 --tensor w1 --index 33 32 --values 1 --tensor b1 --index 32 --values -1 --tensor w2 --index 32 --values 10 0 1 --reason "When traveling I prefer offline options" --harness claude
uv run --locked brain.py decide personal/me.brain examples/personal_traveling.json
```

The first decision abstains. After the two edits it prints `Decision: A (Lower-cost option)` and `Contributors: codex, claude`. The last `set` wires free neuron 32 as `ReLU(offline + traveling - 1)` in one verified edit, so when traveling the offline option wins: `B = neuron 1 (-3) + neuron 32 (+10)`. `brain.py inspect personal/me.brain` reads every neuron back as a rule and lists the history.

## Hand it to an agent

```bash
uv run --locked brain.py prompt personal/me.brain
```

Paste the output into Codex, Claude Code, Copilot, or any agent with a shell, and add what you want encoded. The agent reads every neuron as a rule, edits weights in place with `set`, and checks decisions with a request file. For the next agent, run `prompt` again. [Agents and parallel edits](HARNESS.md)

**Tested end to end with real agents.** Working in parallel, Codex (gpt-6.1-sol) and GitHub Copilot (GPT-5.6) each wired one context rule into their own hidden neurons. `merge` combined their copies the same way in either order, and Claude Code (Opus 5.5) then audited the result against every statement. Earlier runs caught two agents choosing the same neuron, which `merge` refused, and two planted wiring bugs that Claude Code found from the weights alone. Every result was checked independently in all 16 contexts. [Run log](HARNESS.md#tested-with-real-agents)

## How it works

```text
32 option attributes + 16 contexts -> 48 ReLU neurons -> utility, veto, evidence
```

Neuron `j` computes `ReLU(inputs · w1[:, j] + b1[j])`, and row `w2[j]` holds its `[utility, veto, evidence]`. In a new brain, neuron `i` fires when attribute `i` is present in the `general` context, and neurons 32–47 start free; `inspect` lists which are free. `inspect` reads every neuron back as a rule, and `decide` shows which neurons produced each utility. A positive veto removes an option. The best utility wins only if every remaining option has evidence; otherwise the brain abstains. Every one of the 2,499 entries in `w1`, `b1`, `w2`, `b2` is editable. [Weight guide](WEIGHTS.md)

| Command | Does |
|---|---|
| `init FILE` | Create a blank brain |
| `inspect FILE` | Read every neuron as a rule; list inputs and history |
| `set FILE --tensor T --index ... --values ...` | Assign rows or scalars in place as one verified edit; repeat the three flags for several |
| `decide FILE REQUEST.json` | Decide or abstain, with each utility's neurons |
| `serve FILE` | JSON Lines decisions for programs |
| `prompt FILE [--neurons 40-47]` | Print an agent handoff, optionally with its own free neurons |
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
