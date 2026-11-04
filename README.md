# zozoBrain

### Your preferences. A tiny neural decision policy.

**Yossi Eliaz’s decision-policy experiment: directly written weights, local execution, transparent evaluation.**

[Run the demo](#try-yossis-decisions) · [Codex / Claude Code / harness guide](HARNESS.md) · [100 questions about Yossi](QUESTION_BANK.md) · [Evaluation](EVALUATION.md)

2,499 total parameters, 150 nonzero entries. No gradient descent, optimizer, training loop, pretrained backbone, external model, or network call in the main implementation. The network scores options using a small set of documented qualitative preferences attributed to Yossi Eliaz. Unknown preferences remain unknown.

This is **weight-space programming**, not a tiny conversational clone. The weights are explicit NumPy assignments in [weights.py](weights.py). The code, actual weight artifact, examples, tests, and evaluation are included.

## Try Yossi’s decisions

With [uv installed](https://docs.astral.sh/uv/getting-started/installation/), run (uv creates the environment automatically):

```bash
uv run --locked demo.py
```

| Decision | Demo output |
|---|---|
| Fly on Shabbat to attend every conference day? | No; take the feasible alternative |
| Which personal website domain? | yossieliaz.netlify.app |
| Which CV length? | One page |
| Which answer style? | Concise and verified |
| Sushi, pizza, or falafel? | Abstain—no known preference |

These outputs are tested. All supplied attributes are visible in `demo.py`. The numerical weights express a small authored policy, not a measured replica of Yossi’s mind. **Clone it, run it, inspect it, then adapt the explicit rules to your own preferences.**

For broader questions, start with [the 100-question bank about Yossi Eliaz](QUESTION_BANK.md). It separates reference answers from unknown tastes rather than manufacturing labels.

## What it does

Given a context and structured candidate options, return:

- A selected option or an explicit abstention.
- Ranked feasible options with raw utility scores.
- Options vetoed by encoded hard constraints.
- Whether modeled preference evidence is present.

Yes/no decisions are simply two options named `yes` and `no`. Lists of food, companies, countries, music, or people use the same interface. Names alone do not reveal preferences. For example, it can apply the no-flying-on-Shabbat constraint; it cannot know whether Yossi prefers sushi to pizza or OpenAI to Anthropic without evidence.

**Input contract:** a question of up to 100 regex word/punctuation tokens, explicit context tags, and structured option attributes. The question is validated and retained as metadata; the network does **not** parse arbitrary natural language. Attribute extraction is a separate caller responsibility and is not hidden in the parameter budget. This narrower contract makes the weight-space implementation honest and reproducible. The raw-text experiment is retained separately as a baseline.

## Run from an agent or harness

See [HARNESS.md](HARNESS.md) for a copy-paste Codex / Claude Code task, shell setup, and the tested JSON Lines subprocess interface (`uv run --locked harness.py`). No API key or special integration is needed.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once. The committed `uv.lock` pins dependencies; no manual virtual environment activation is needed. Python 3.11–3.13 is supported. `uv run --locked` refuses dependency drift. See the [uv locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

```bash
git clone https://github.com/zozo123/zozoBrain.git
cd zozoBrain
uv run --locked model.py examples/travel.json
uv run --locked model.py examples/work.json
uv run --locked model.py examples/food.json
uv run --locked python -m unittest -v
uv run --locked evaluate.py
```

The travel example selects `no`: do not fly on Shabbat. The food example abstains because no food preferences were supplied. `--force` requests an answer for unknowns/ties, using utility then input order, and marks the result as forced. It never overrides vetoes. A forced order is not evidence of a preference.

Python API:

```python
from model import DecisionModel
brain = DecisionModel()
result = brain.decide(
    question="Should I fly on Shabbat to attend every conference day?",
    contexts=["travel"],
    options=[
        {"id": "yes", "attributes": {"shabbat_flight": 1}},
        {"id": "no", "attributes": {"avoids_shabbat": 1,
                                    "misses_one_or_two_days": 1}},
    ],
)
print(result["decision"])  # no
```

Features are in [weights.py](weights.py). Contexts are binary; option attributes are in [0,1]. Hard constraint attributes must be binary. Feature values describe the option, not its desired answer. `explicitly_preferred` and `explicitly_disliked` are for actual caller-supplied preferences, not guesses. Scores are authored utility units, not probabilities or measured psychological quantities. Printed active attributes are input diagnostics, not a generated explanation.

## Neural architecture

`48 inputs → 48 ReLU neurons → 3 linear outputs`

The inputs are 32 option attributes and 16 context indicators. Outputs are utility, veto activation, and modeled-evidence activation. Each context-dependent neuron computes `ReLU(attribute + context - 1)`. Universal explicit-preference rules use the attribute directly. Linear output weights express the policy. All 16 currently unused hidden units still count toward the parameter budget.

| Tensor | Shape | Parameters |
|---|---|---:|
| W1 | 48 × 48 | 2,304 |
| b1 | 48 | 48 |
| W2 | 48 × 3 | 144 |
| b2 | 3 | 3 |
| **Total** | | **2,499** |

There are 150 nonzero entries, but **2,499 is the relevant total**. Float32 tensor storage is 9,996 bytes before container overhead. `uv run --locked weights.py` deterministically emits [weights.npz](weights.npz) from the assignments; inference builds those same weights. No training is needed. Sorting, abstention, validation, and veto filtering are ordinary deterministic wrapper logic; the veto activations themselves come from the neural forward pass.

## What is encoded—and what is not

The qualitative policy uses stated travel constraints, concise verified communication, completed work, scientific AI and sandbox work, reproducibility, selected design revisions, single-page CVs, website domain choice, course requirements, and analytics requirements. Numerical strengths such as 3 versus 4 are **author-chosen hypotheses**, not learned estimates of Yossi's trade-offs. The network is additive within context; cross-domain trade-offs have not been validated.

There are no inferred political affiliations, musical tastes, food preferences, national rankings, or personal rankings of named individuals. Career fit is not company preference; professional expertise is not human worth. No private financial holdings, immigration records, contact details, or email bodies are published. Historical preferences can change.

## Evaluation: two different questions

| Evaluation | Result | Meaning |
|---|---|---|
| Main network: 100 structured policy probes | 100/100 checks pass; 26 decisions, 74 abstentions | Encoded rules and unknown handling behave as specified |
| Main network: independent real user choices | Not available | Personal predictive accuracy is unmeasured |
| Earlier 9,272-parameter gradient baseline: 100 raw-text questions | 15/24 supported labels correct; 76 unscored | Limited generalization; unstated tastes lack ground truth |

The main 100 probes cover 20 each in food, countries/travel, music, politics, and work/companies/people. Ten country probes exercise the same travel constraint across destinations. Five work probes use explicitly **hypothetical** supplied preferences. These are conformance tests, not 100 independent personal decisions. Inputs explicitly provide the facts needed by the policy. **100/100 does not mean the model predicts Yossi with 100% accuracy.**

See [EVALUATION.md](EVALUATION.md), [benchmark_100.json](benchmark_100.json), and [evaluation_100.json](evaluation_100.json) for all inputs, outputs, provenance notes, and expectations. The [gradient baseline](baselines/gradient/README.md) and its weights/training code are preserved for reproducibility. It uses a different input contract, so the numbers are not an apples-to-apples model comparison.

## Make your own in three steps

```bash
uv run --locked init_personal.py alice
uv run --locked personal/alice/model.py personal/alice/decision.json
```

1. **Start blank.** The generator creates a private, gitignored `personal/alice/` folder. All utilities and veto flags start at zero: no Yossi preferences are inherited. The first run abstains.
2. **Author your policy.** Edit `personal/alice/policy.json`. If you actually prefer lower cost, set the `low_cost` utility to `3` and `high_cost` to `-3`. Run the same command again: the sample selects `A`. Keep 32 uniquely named attributes and 16 contexts; rename unused slots for your own domains. Veto flags represent your explicit hard constraints. The generated README explains the schema.
3. **Validate separately.** Supply your own options in `decision.json`. Record real choices before changing weights and keep held-out cases. Yossi’s benchmark is not an evaluation of you. Export your authored weights with `uv run --locked personal/alice/weights.py`.

Your folder contains its own model, harness, weights compiler, policy, and example. The JSONL interface also works: `uv run --locked personal/alice/harness.py`. The generated folder is excluded from git commits by default; publishing it is an explicit choice. The generator refuses to overwrite an existing profile. The original Yossi model is untouched.

Attributes describe options; utilities describe your preferences. Language interpretation remains external to the tiny network. Do not encode guessed preferences as if you supplied them. All profiles keep the same 2,499-total-parameter architecture, without fitting or optimization.

No optimization is required for the main model. The optional historical baseline can be reproduced with:

```bash
cd baselines/gradient
OPENBLAS_NUM_THREADS=1 uv run --locked train.py --output-dir reproduced
uv run --locked python -m unittest -v
```

The motivating weight-space examples were supplied by the user; this repository does not independently validate their published performance claims. This implementation demonstrates direct policy compilation into a small neural net. It does not claim a general digital twin, robust language understanding, or authority to make decisions on anyone's behalf.

MIT license applies to code, weights, and example artifacts.
