# Agents and programs

An agent turns what you say into tensor edits; the brain runs them locally. Any agent that can run shell commands works: Codex, Claude Code, GitHub Copilot CLI, Cursor, Gemini CLI. No SDK, API key, or chat-history access is involved.

## One brain, several agents

```bash
uv run --locked brain.py init personal/me.brain
uv run --locked brain.py prompt personal/me.brain
```

1. Paste the printed handoff into the first agent, followed by your statements and an evidence reference for each.
2. The agent inspects the labeled rows, assigns entries in place with `brain.py set ... --harness NAME`, and checks decisions.
3. Run `prompt` again and paste it into the next agent. The file carries every earlier edit and its evidence.

`brain.py inspect personal/me.brain` shows who changed what:

```text
Contributors: copilot, claude-code
History: 4 verified contributions:
  1 copilot: 2 edits, evidence "owner-2026-11-04-S1"
  2 copilot: 2 edits, evidence "owner-2026-11-04-S1"
  3 claude-code: 3 edits, evidence "owner-2026-11-04-S2"
  4 claude-code: 2 edits, evidence "owner-2026-11-04-S3"
```

For agents working at the same time, give each a copy and combine them with `diff` and `merge` ([USAGE.md](USAGE.md#parallel-copies)). Conflicting values are refused. The harness name and evidence are claims made by the caller, so review the numbers as well.

## Tested with real agents

November 4, 2026, from a fresh clone of `main`. Each agent got only the `prompt` handoff and the owner's statements:

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
