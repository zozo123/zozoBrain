# Run from Codex, Claude Code, or your own harness

This is an ordinary local Python program. An agent with shell access can clone it and execute it. No special plugin, account, API key, MCP server, or model download is required. Internet access is needed only to clone the repository and install NumPy. Inference is local.

## Paste this task into your coding agent

```text
Clone https://github.com/zozo123/zozoBrain and read README.md and HARNESS.md.
Create a local Python 3.11+ virtual environment and install requirements.txt.
Run python -m unittest -v and python evaluate.py, then run
python model.py examples/travel.json and python model.py examples/food.json.
Report the actual outputs and any failures.

For my subsequent decision requests, create a structured JSON input using only
the contexts and attributes defined in weights.py and invoke the model.
Show me the attributes you supplied. Do not invent personal preferences, edit
weights to force a desired result, or replace the network's answer with your own.
Preserve abstentions. The language-model interpretation step is external to
the 2,499-parameter network and must be identified separately.
```

Codex and Claude Code are examples of shell-capable callers, not bundled dependencies. The repository has been tested through a Python subprocess harness; no claim is made that specific versions of those agents have been separately tested.

## Direct shell use

```bash
git clone https://github.com/zozo123/zozoBrain.git
cd zozoBrain
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python evaluate.py
python model.py examples/travel.json
```

On Windows, activate with `.venv\Scripts\Activate.ps1`, or invoke `.venv\Scripts\python.exe` directly.

## JSON Lines subprocess protocol

Start `python harness.py`. Send one complete JSON object per stdin line. Each nonblank input line returns one JSON object on stdout, flushed immediately. Do not send pretty-printed multiline JSON to this interface. Logs/formatting do not pollute stdout.

Example request:

```json
{"id":"decision-1","question":"Should I fly on Shabbat?","contexts":["travel"],"options":[{"id":"yes","attributes":{"shabbat_flight":1}},{"id":"no","attributes":{"avoids_shabbat":1}}]}
```

Successful responses contain `id`, `ok: true`, and `result` (decision, ranking, vetoed options, status, forced flag, note). This request returns decision `no` with status `only_feasible_option`. Invalid requests return `ok: false` and an error message; subsequent requests continue. Process exit code zero alone does not mean all requests succeeded: inspect each response's `ok` field. `force` is optional and defaults to false. Forced answers never override a veto.

For a single call from another Python application:

```python
import json
import subprocess
import sys
from pathlib import Path

repo = Path('/absolute/path/to/zozoBrain')
request = json.loads((repo / 'examples/travel.json').read_text())
request['id'] = 'decision-1'
response = subprocess.run(
    [sys.executable, str(repo / 'harness.py')],
    input=json.dumps(request) + '\n',
    text=True, capture_output=True, check=True,
)
message = json.loads(response.stdout)
assert message['ok'], message
print(message['result']['decision'])
```

Use the Python executable from the environment where NumPy is installed. For multiple requests, keep the process running and read one response line for each nonblank request. The model initializes once per process.

## Boundaries

- This is a model of selected Yossi preferences, not the preferences of whoever clones it. Others must explicitly author their own policy to personalize it.
- The 100-token question is metadata. Structured attributes determine scores. A coding agent may help extract attributes, but then the overall system includes that agent's larger language model.
- Do not set `explicitly_preferred` merely because an agent guesses an option is better. Use it only for an actual stated preference.
- `decision: null` is an expected response when evidence is missing, choices tie, or every option is vetoed.
- The 100 structured evaluation cases check policy conformance, not general personality prediction.
