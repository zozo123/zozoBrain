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
History: 3 verified contributions:
  1 copilot: 2 edits, evidence "owner-2026-11-04-S1"
  2 claude-code: 3 edits, evidence "owner-2026-11-04-S2"
  ...
```

For agents working at the same time, give each a copy and combine them with `diff` and `merge` ([USAGE.md](USAGE.md#parallel-copies)). Conflicting values are refused. The harness name and evidence are claims made by the caller, so review the numbers as well.

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
