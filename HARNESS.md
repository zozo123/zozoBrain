# Codex, Claude Code, and other harnesses

A harness interprets your explicit preferences and authors numerical edits. The
portable decision model runs the resulting tensors locally. Any agent that can
read/write files and run Python can use the same workflow. No provider SDK, API
key, hosted service, or automatic chat-history access is bundled.

## Recommended: one portable file

Use the [one-file workflow](USAGE.md) for new integrations. `brain.py set`
directly edits scalar/row tensor entries and retains evidence in the same
`.brain` file; `brain.py decide` loads its matching schema and history together.
`brain.py prompt FILE` generates an agent handoff with current paths and hashes.
`brain.py serve FILE` provides the JSONL protocol below with one startup
verification. Keep it alive for repeated requests; restart for a saved update.

The snapshot/manifest commands below remain supported as lower-level interfaces.

## Accumulate Codex then Claude

Create a blank brain once. All following commands run inside that directory:

```bash
uv run --locked init_personal.py me
cd personal/me
uv run --locked weights.py --output base.npz
```

Keep `base.npz` unchanged. Codex directly edits a copy of its actual tensors and
saves `codex.npz`. For example, the two explicit tensor assignments in the
profile README express a hypothetical lower-cost preference; use only your own
stated preferences. Capture the real numerical diff and build the first state:

```bash
uv run --locked weight_patches.py diff base.npz codex.npz --harness codex --id codex-001 --evidence "Reference to the explicit instruction used for these edits" --output codex.patch.json
uv run --locked weight_patches.py stack base.npz codex.patch.json --output after-codex.npz --manifest after-codex.manifest.json
```

Take the whole directory to Claude. Claude loads `after-codex.npz`, preserves the
feature layout and existing entries, directly edits additional tensor entries,
and saves `claude.npz`. Its contribution is based on the accumulated state:

```bash
uv run --locked weight_patches.py diff after-codex.npz claude.npz --harness claude --id claude-001 --evidence "Reference to the next explicit instruction" --output claude.patch.json
uv run --locked weight_patches.py stack after-codex.npz claude.patch.json --resume-manifest after-codex.manifest.json --output combined.npz --manifest combined.manifest.json
uv run --locked model.py decision.json --weights combined.npz --manifest combined.manifest.json
uv run --locked harness.py --weights combined.npz --manifest combined.manifest.json
```

The final inference result identifies both contributing harnesses and the exact
tensor/schema fingerprints. The optional manifest is verified against the loaded
weights and its recorded history before inference starts. Use the final snapshot
and manifest as the next session's input; give each contribution a unique ID.

If agents work in parallel, start both from the same base and stack both patches
in one command. Disjoint edits combine; same-entry edits that agree are retained
without double-counting; disagreements fail. To revise a weight intentionally,
start from the current accumulated snapshot, then capture the revision against
that state. Stacking never averages weights or treats one harness as inherently
more authoritative.

```bash
uv run --locked weight_patches.py stack base.npz codex.patch.json claude.patch.json --output combined.npz --manifest combined.manifest.json
```

That parallel example assumes **both** patches name `base.npz`; it is an
alternative to the sequential commands above. Preserve all snapshots and patch
files if you want to rebuild or audit the full history. Keep contribution evidence
private unless you choose to publish it. Generated profiles have their own locked
dependencies and can run outside this repository.

## Task to paste into either agent

Replace the directory, active snapshot, manifest, and contribution ID below:

```text
Use my existing zozoBrain profile at /absolute/path/to/my-brain.
Read README.md, weights.py, and the current manifest before changing anything.
Use the profile's uv.lock and uv run --locked; do not install globally.

From the context I have authorized you to use, collect only explicit preferences,
constraints, and corrections. Keep a short evidence reference for each change.
Separate those statements from your hypotheses about numerical utility strength.
Do not guess tastes, turn unrelated facts into preferences, or import other
accounts' history. If evidence or feature coverage is missing, report the gap.

Load the active NPZ tensors and edit w1, b1, w2, b2 directly. No training,
optimizer, preference compiler, or weight averaging. Preserve the current feature
layout; coordinate schema changes separately. A new preference needs appropriate
feature wiring and evidence output, and a hard constraint needs the veto output
and binary-input metadata. Test the behavior using structured options.

Keep the input snapshot unchanged. Save an edited NPZ, generate a versioned diff
with your harness name, unique contribution ID, and evidence. Stack against the
active snapshot, using --resume-manifest for its existing provenance. Resolve
conflicts explicitly; preserve abstentions. Verify inference with the resulting
snapshot and manifest. Report actual weight changes, checks, and remaining gaps.

For subsequent decisions, extract only supported contexts and option attributes,
show the supplied attributes, and call the local model with --weights and
--manifest. Return its actual decision or abstention. Identify the language
interpretation step as external to the 2,499-parameter network.
```

For inference alone, skip the editing/contribution steps. The names Codex and
Claude describe callers; tests exercise their shared file/subprocess protocol,
not proprietary agent APIs or access to their stored conversations.

## JSON Lines subprocess protocol

Start `uv run --locked harness.py`, optionally with `--weights` and `--manifest`.
Send one complete JSON object per stdin line. Each nonblank line produces one
JSON response, flushed immediately. The model initializes once per process.
Restart after changing source or selecting another snapshot.

Example using the shipped Yossi brain from the repository root:

```json
{"id":"decision-1","question":"Should I fly on Shabbat?","contexts":["travel"],"options":[{"id":"yes","attributes":{"shabbat_flight":1}},{"id":"no","attributes":{"avoids_shabbat":1}}]}
```

The response has `id`, `ok: true`, and `result`, including decision, ranking,
vetoed options, status, forced flag, `weights_sha256`, and `schema_sha256`.
With a verified manifest it also includes `contributing_harnesses`. That request
selects `no` with status `only_feasible_option`. Generated blank profiles use their
own named attributes; use their `decision.json` instead of this Yossi example.

Request IDs may be strings, integers, or null. Option IDs are unique nonempty
strings; labels are strings. Contexts are a list of known unique names;
attributes are finite numeric values in [0,1], with binary hard constraints.
`force` is a boolean, default false. Nonfinite JSON numbers and malformed inputs
produce `ok: false` with an error; the next request still runs. Inspect each
response's `ok` field even when the process exit code is zero.

A Python application can use the same process protocol:

```python
import json
import subprocess
from pathlib import Path

brain = Path('/absolute/path/to/my-brain')
request = json.loads((brain / 'decision.json').read_text())
response = subprocess.run(
    ['uv', 'run', '--locked', '--project', str(brain), str(brain / 'harness.py'),
     '--weights', str(brain / 'combined.npz'),
     '--manifest', str(brain / 'combined.manifest.json')],
    input=json.dumps(request) + '\n', text=True, capture_output=True, check=True,
)
message = json.loads(response.stdout)
assert message['ok'], message
print(message['result']['decision'])
```

Question text is metadata, limited to 100 regex word/punctuation tokens. The
network consumes structured attributes and contexts, not arbitrary chat text.
`decision: null` is expected for unknown preferences, ties, or no feasible option.
Forced answers never override the currently encoded vetoes. The root's 100
policy probes measure its authored behavior, not independent personality accuracy.
