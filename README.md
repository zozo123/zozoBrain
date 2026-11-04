# zozoBrain

**A personal decision policy compiled directly into neural weights.**

2,499 total parameters, 150 nonzero entries. No gradient descent, optimizer, training loop, pretrained backbone, external model, or network call in the main implementation. The network scores options using a small set of documented qualitative preferences attributed to Yossi Eliaz. Unknown preferences remain unknown.

This is **weight-space programming**, not a tiny conversational clone. The weights are explicit NumPy assignments in [weights.py](weights.py). The code, actual weight artifact, examples, tests, and evaluation are included.

## What it does

Given a context and structured candidate options, return:

- A selected option or an explicit abstention.
- Ranked feasible options with raw utility scores.
- Options vetoed by encoded hard constraints.
- Whether modeled preference evidence is present.

Yes/no decisions are simply two options named `yes` and `no`. Lists of food, companies, countries, music, or people use the same interface. Names alone do not reveal preferences. For example, it can apply the no-flying-on-Shabbat constraint; it cannot know whether Yossi prefers sushi to pizza or OpenAI to Anthropic without evidence.

**Input contract:** a question of up to 100 regex word/punctuation tokens, explicit context tags, and structured option attributes. The question is validated and retained as metadata; the network does **not** parse arbitrary natural language. Attribute extraction is a separate caller responsibility and is not hidden in the parameter budget. This narrower contract makes the weight-space implementation honest and reproducible. The raw-text experiment is retained separately as a baseline.

## Run

Python 3.11+:

```bash
git clone https://github.com/zozo123/zozoBrain.git
cd zozoBrain
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model.py examples/travel.json
python model.py examples/work.json
python model.py examples/food.json
python -m unittest -v
python evaluate.py
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

There are 150 nonzero entries, but **2,499 is the relevant total**. Float32 tensor storage is 9,996 bytes before container overhead. `python weights.py` deterministically emits [weights.npz](weights.npz) from the assignments; inference builds those same weights. No training is needed. Sorting, abstention, validation, and veto filtering are ordinary deterministic wrapper logic; the veto activations themselves come from the neural forward pass.

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

## Replicate or personalize

1. Define a narrow decision domain and collect explicit preferences and constraints.
2. Edit the rules and direct assignments in `weights.py`. Do not infer missing preferences from identity or mere interest in a topic.
3. Supply transparent option attributes. Keep attribute extraction separate from neural decision-making. If an LLM supplies them, disclose that additional model rather than calling the entire system a 2,499-parameter model.
4. Run `python weights.py`, `python -m unittest -v`, and `python evaluate.py`.
5. Collect independent user-labeled choices **before** editing weights. Reserve those choices for a real predictive evaluation. Never report development probes as held-out human accuracy.

No optimization is required for the main model. The optional historical baseline can be reproduced with:

```bash
cd baselines/gradient
OPENBLAS_NUM_THREADS=1 python train.py --output-dir reproduced
python -m unittest -v
```

The motivating weight-space examples were supplied by the user; this repository does not independently validate their published performance claims. This implementation demonstrates direct policy compilation into a small neural net. It does not claim a general digital twin, robust language understanding, or authority to make decisions on anyone's behalf.

MIT license applies to code, weights, and example artifacts.
