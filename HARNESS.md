# Agents and programs

An agent turns what you say into tensor edits; the brain runs them locally. Any agent that can run shell commands works: Codex, Claude Code, GitHub Copilot CLI, Cursor, Gemini CLI. No SDK, API key, or chat-history access is involved.

## One brain, several agents

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py prompt personal/me.brain
```

1. Paste the printed handoff into the first agent, followed by your statements and an evidence reference for each.
2. The agent reads each neuron as a rule with `inspect`, assigns entries in place with `brain.py set ... --harness NAME`, and checks decisions.
3. Run `prompt` again and paste it into the next agent. The file carries every earlier edit and its evidence.

`brain.py inspect personal/me.brain` shows who changed what:

```text
Contributors: copilot, claude-code
History: 4 verified contributions:
  1 copilot: 2 edits to neuron 0, evidence "owner-2026-11-04-S1"
  2 copilot: 2 edits to neuron 1, evidence "owner-2026-11-04-S1"
  3 claude-code: 3 edits to neuron 3, evidence "owner-2026-11-04-S2"
  4 claude-code: 2 edits to neuron 2, evidence "owner-2026-11-04-S3"
```

For agents working at the same time, give each a copy and combine them with `diff` and `merge` ([USAGE.md](USAGE.md#parallel-copies)). Conflicting values are refused, and so are two copies that change the same hidden neuron differently, so give each agent its own neurons with `prompt --neurons`. The harness name and evidence are claims made by the caller, so review the numbers as well.

## Tested with real agents

**November 5, 2026: the 1.0 candidate, parallel agents with their own neurons**, from a fresh clone. On the same base, Codex CLI (gpt-6.1-sol, 154 s) got `prompt --neurons 32-39` and "when traveling, offline", and GitHub Copilot CLI (GPT-5.6, 40 s) got `--neurons 40-47` and "when urgent, fast". Each agent used only its own neurons (32 and 40) and chose utility 7. `merge` combined the copies, and both merge orders gave the same weights. Independent checks passed in every context. Claude Code (Opus 5.5) audited the result against all four statements and made no edits (6.9 min).

**November 5, 2026: parallel agents on the v0.4.0 release**, from a fresh clone of the tag. The owner stated a cost preference and a hard rule. Codex CLI (gpt-6.1-sol, 128 s) and GitHub Copilot CLI (GPT-5.6, 40 s) then each encoded one rule in its own copy at the same time, "when traveling, offline" and "when urgent, fast", and both chose utility 7 to beat the 6-point cost gap. Both also chose neuron 32, and `merge` refused the two copies instead of building a rule neither wrote. After Codex's copy was merged, Copilot ran again on the result and took neuron 33 (32 s). Claude Code (Opus 5.5) audited the result against all four statements, confirmed each one in every context, and made no edits (9.7 min). Because of this run, `prompt --neurons` gives each parallel agent its own neurons.

**November 5, 2026: the hidden layer.** Each agent got only the `prompt` handoff and the statements below, on its own copy of a brain that already prefers lower cost (`w2[0] = [3, 0, 1]`, `w2[1] = [-3, 0, 1]`). Every result was checked independently: decisions in all 16 contexts and a diff of every changed entry.

| Task | Agent | Result | Time |
|---|---|---|---|
| Encode "When I am traveling, I prefer offline options, even if they cost more" | Codex CLI (gpt-6.1-sol) | Wired free neuron 32 in one edit: `w1[2, 32] = 1`, `w1[33, 32] = 1`, `b1[32] = -1`, `w2[32] = [7, 0, 1]`. Exactly 5 entries changed; all checks pass | 2.7 min |
| The same statement, 6 runs | GitHub Copilot CLI (GPT-5.6 Luna) | The same neuron every time. Three first tries set utility 3 or 4, too small to close the 6-point cost gap; two of those runs caught it in their own check and raised it to 7. 5 of 6 pass | 25–58 s |
| Audit a brain against three statements; two wiring bugs planted | Claude Code (Opus 5.5) | Found both from the weights: the veto fired only with `general`, and the traveling neuron read `high_cost`. Fixed 4 entries in 2 edits; the veto now holds in all 16 contexts | 6.6 min |

What the run changed: `decide` prints each utility as the sum of its neurons, and the handoff says outputs add up and to check with every active context. `inspect` marks rules that fire only with a context, lists the attributes rules read, and with `--history` shows every entry each edit changed. Abstentions name the option without evidence. The traveling example now has a third option, so a neuron wired to the wrong attribute shows up as a tie.

**November 4, 2026: output rows**, from a fresh clone of `main`. Each agent got only the `prompt` handoff and the owner's statements:

| Statement | Agent | Edits it made | Time |
|---|---|---|---|
| S1 "I prefer lower cost" | GitHub Copilot CLI (GPT-5.6 Luna) | `w2[0] = [1, 0, 1]`, `w2[1] = [-1, 0, 1]` | 25 s |
| S2 "Never pick explicitly disallowed options, in every context" | Claude Code (Opus 5.5) | `w1[32, 3] = 0`, `b1[3] = 0`, `w2[3] = [0, 1, 0]` | 6.3 min, S2 and S3 together |
| S3 "Prefer offline when otherwise equal" | Claude Code | `w2[2] = [0.01, 0, 1]` | |

Copilot's rows survived Claude's edits. Every check passed:

- cheap beats pricey, 1 to −1;
- the banned option is vetoed in all 16 contexts, even with `--force`;
- cheap_offline beats cheap_cloud, 1.01 to 1;
- options with no stated preference still abstain.

Claude Code also reported what the layout cannot express: a tie-breaker weight is exact only for 0/1 attributes.

## Programs: JSON Lines

```bash
uv run --locked brain.py serve personal/me.brain
```

Send one request object per line:

```json
{"id": "q1", "question": "Which option?", "contexts": ["general"], "options": [{"id": "A", "attributes": {"low_cost": 1}}, {"id": "B", "attributes": {"high_cost": 1}}]}
```

Each line gets one flushed response with `id`, `ok`, and either `result` or `error`. The `result` has `decision` (or null), `status`, `ranking`, `vetoed`, `forced`, `weights_sha256`, `schema_sha256`, and `contributing_harnesses`. Ids may be strings, integers, or null; `force` is a boolean. Malformed, non-finite, or undecodable lines get `ok: false`, and the stream continues.

From Python:

```python
import json, subprocess
request = json.load(open('examples/personal.json'))
out = subprocess.run(['uv', 'run', '--locked', 'brain.py', 'serve', 'personal/me.brain'],
                     input=json.dumps(request) + '\n', text=True, capture_output=True, check=True)
print(json.loads(out.stdout)['result']['decision'])
```

The legacy source path still works: `uv run --locked harness.py [--weights FILE.npz --manifest FILE.json]` speaks the same protocol, and `weight_patches.py diff/stack` combines NPZ snapshots with a stack manifest ([FORMATS.md](FORMATS.md)).
