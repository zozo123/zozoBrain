# zozoBrain

A tiny experimental **personal decision model** of Yossi Eliaz: **9,272 total learned parameters**, with actual trained weights. Given a situation and candidate options, it estimates which option Yossi would choose. It also answers yes/no and ranks lists. Runs locally with NumPy; no LLM, embeddings API, retrieval service, or network call at inference.

**This is a narrow preference approximation, not a digital mind.** The shipped labels are assistant-written interpretations of conversation history, not a questionnaire answered by Yossi. The model is poor at genuinely new phrasing and cannot know unstated tastes.

## Run the shipped model

Python 3.11+:

```bash
git clone https://github.com/zozo123/zozoBrain.git
cd zozoBrain
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model.py --decision-file examples/decision.json
python model.py 'Do I avoid flying on Shabbat?'
python model.py 'Which answer style would I prefer?' --options 'Concise direct answers with verified facts' 'Long padded answers with unsupported claims'
python model.py 'Which companies fit my AI systems career interests?' --options 'OpenAI research engineering and reinforcement learning' 'Anthropic AI research and agent infrastructure' 'An unrelated company offering only retail sales'
python model.py 'Which food would I prefer?' --options 'Sushi' 'Pizza' 'Falafel'
```

The food example demonstrates the interface, **not a known preference**. Likewise, bare company or person names may receive arbitrary scores. Career relevance does not establish a preference between OpenAI and Anthropic. A list of people is ranked only as model output, never as a measure of human worth.

For Python:

```python
from model import Model
brain = Model()
print(brain.predict('Do I prefer concise answers?'))
print(brain.rank(['Jazz', 'Rock', 'Classical'], 'Which music would I enjoy?'))
print(brain.decide('Which answer style would I prefer?', [
    {'id': 'short', 'label': 'Concise direct answers with verified facts'},
    {'id': 'long', 'label': 'Long padded answers with unsupported claims'}
]))
```

`I` and `me` refer to Yossi. Other users retrain with their own labels. Default yes/no CLI output is just `yes` or `no`; `--details` includes an **uncalibrated** sigmoid score. Ranking returns a selected option and descending raw utility scores. Stable input order breaks exact ties. A warning is always included for rankings; a keyword heuristic additionally flags some unsupported domains. This heuristic is **not** a learned confidence or out-of-distribution detector. Unsupported yes/no questions still return a forced answer.

The main `decide(context, options)` interface returns a selected option ID, ranked feasible alternatives, raw score margin, excluded options, and a warning. Give options short descriptions so the network sees the relevant attributes, not just opaque names. Option `eligible: false` excludes it deterministically **outside** the network; it must be set by the caller from an explicit constraint. The example marks a Shabbat flight ineligible. No date parser or hidden religious rules infer eligibility. If all options are excluded, the decision is `null` with `no_feasible_option`. A single feasible option wins by feasibility, not evidence of preference. Scores/margins are not confidence, and the model does not generate reasons. With all options eligible and no descriptions, decisions and rankings use exactly the same neural scores.

## Architecture and parameter accounting

| Component | Shape / operation | Learned scalars |
|---|---|---:|
| Tokenizer | Unicode regex words and punctuation | 0 |
| Features | Signed BLAKE2b hashing of word 1-, 2-, 3-grams into 512 bins; L2 normalization | 0 |
| Hidden weights | 512 × 18 | 9,216 |
| Hidden bias | 18; tanh activation | 18 |
| Output weights | 18 × 2 | 36 |
| Output bias | 2 | 2 |
| **Total** | Shared encoder + yes/no and option-utility heads | **9,272** |

All float32 arrays occupy 37,088 bytes, excluding NPZ container overhead. Counts include every dense weight, including any zero. No pretrained backbone or vocabulary lookup table hides extra learned parameters. The fixed hash function provides lexical features, not pretrained semantic knowledge.

Inputs allow 1–100 **regex tokens**, not GPT/BPE tokens. Overlength inputs are rejected, not silently truncated. Each ranking candidate is scored as `question + ' option: ' + option`; the combined text including markers must fit 100 tokens. Lists are unbounded in length, evaluated candidate by candidate. Pair selection is the first item of the same ranking. Context can affect the utility, but this lexical architecture has limited compositional understanding and negation handling. English training only.

## Reproduce training and evaluation

```bash
OPENBLAS_NUM_THREADS=1 python train.py --output-dir reproduced
python model.py 'Do I avoid flying on Shabbat?' --weights reproduced/weights.npz
python evaluate.py
python -m unittest -v
```

The shipped model was trained from random initialization with seed 42, full-batch Adam, learning rate 0.015, 1,800 fixed steps, masked binary cross-entropy over the two heads, and 1e-5 weight regularization. 74 yes/no seeds plus 22 preference pairs yield 472 augmented training examples. Ranking is trained with preferred/rejected pointwise utility targets; it is not a listwise loss. There is no benchmark-based checkpoint selection. `train.py` never reads the 100-item benchmark. CPU/BLAS differences can yield small floating point differences; tests compare arrays with tolerances.

`metrics.json` reports 73/74 correct on held-out yes/no prefix variants and 22/22 pair orders. These are **same-fact template tests**, not a measure of generalization to new topics.

The harder [100-question evaluation](EVALUATION.md) executes 20 items each in food, countries/travel, music, politics, and work (including companies and people):

| Domain | Executed | Ground-truth labels available | Correct |
|---|---:|---:|---:|
| Food | 20 | 0 | N/A |
| Countries / travel | 20 | 7 | 4 |
| Music | 20 | 0 | N/A |
| Politics | 20 | 0 | N/A |
| Work / companies / people | 20 | 17 | 11 |
| **Total** | **100** | **24** | **15 (62.5% of labeled)** |

The other **76 outputs are unscored**, not successes. Political views, food tastes, musical preferences, and relative country preferences were not inferred from nationality, religious observance, questions asked, or travel history. Country labels cover documented logistical constraints, not national rankings. The benchmark was authored after the initial training design and is not an independent blinded assessment. It contains fresh phrasings of some known topics plus unsupported topics; accuracy on it cannot validate a personality replica. Every output, score, expected label, provenance note, and error is in [evaluation_100.json](evaluation_100.json).

## Make your own

1. Copy `seeds.tsv` and `ranking_seeds.json` into a private `personal/` directory. Replace the example data with your own explicit answers. Each TSV row is `label` (0 or 1), `question`, `evidence`. Each pair has `context`, `better`, `worse`, `evidence`.
2. Include both positive and negative answers, difficult negations, alternative phrasings, and realistic competing options. For food, music, countries, companies, and people, collect actual choices. Do not treat a language model's guesses as personal ground truth.
3. Train with `python train.py --data-dir personal --output-dir personal/model` and pass `--weights personal/model/weights.npz` at inference.
4. Keep a separate, user-labeled test set **before** training. Do not copy benchmark answers into training and then claim held-out accuracy. The included 100-item benchmark is specific to this shipped model; fork `evaluate.py` and its benchmark paths for a new person.

`personal/` is gitignored. The public data includes selected work, communication, design, teaching, and travel preferences; excludes financial holdings, immigration case details, private contacts, and private messages. The weights may encode the same information as their dataset; neither is anonymized. Historical preferences can change.

## Relation to directly written weights

The user supplied a screenshot of a claim that an LLM hand-wrote sparse legal-citation classifier weights. That is inspiration, not a replicated experiment or evidence for this model's accuracy. **This implementation uses gradient-trained weights** on synthetic labels, not hand-authored numerical weights. Both approaches must count all parameters consistently and test against genuinely known labels. A larger sparse model's nonzero count is different from its total parameter count.

Code, weights, example data, and reports are included under the MIT license. Treat all personalized predictions as experimental; they do not authorize decisions on someone's behalf.
