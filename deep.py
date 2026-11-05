"""One layer deeper than simplest.py: hidden neurons are rules you write by hand. Still no training.

A context such as traveling is the same input for every option, so in one matrix it adds
the same row to every score and can never reorder the options. A hidden ReLU neuron can
combine it with an option attribute: ReLU(offline + traveling - 1) fires only when both are 1.
"""
import itertools

import numpy as np

from simplest import pick

INPUTS = ['low_cost', 'high_cost', 'offline', 'disallowed', 'traveling']  # option attributes, then a context
OPTIONS = {'cheap': ['low_cost'], 'offline_kit': ['high_cost', 'offline'], 'banned_cheap': ['low_cost', 'disallowed']}


def forward(W1, b1, W2, features):
    hidden = np.maximum(0, np.isin(INPUTS, features).astype(float) @ W1 + b1)  # ReLU hidden layer
    return hidden, hidden @ W2                                                 # utility, veto, evidence


def decide(W1, b1, W2, context):
    """The same pick as simplest.py, one layer deeper."""
    return pick({name: forward(W1, b1, W2, features + context)[1] for name, features in OPTIONS.items()})


def rule(W1, b1, W2, j):
    """Read neuron j back from its weights."""
    terms = ' + '.join(name if w == 1 else f'{w:g}*{name}' for name, w in zip(INPUTS, W1[:, j]) if w)
    bias = f' - {-b1[j]:g}' if b1[j] < 0 else f' + {b1[j]:g}' if b1[j] else ''
    utility, veto, evidence = W2[j]
    return (f'neuron {j} = ReLU({terms}{bias})'.ljust(41)
            + f'-> utility {utility:g}, veto {veto:g}, evidence {evidence:g}')


def main():
    W1, b1, W2 = np.zeros((5, 4)), np.zeros(4), np.zeros((4, 3))  # 5 inputs -> 4 hidden neurons -> 3 outputs
    for j, (name, row) in enumerate([('low_cost', [3, 0, 1]), ('high_cost', [-3, 0, 1]), ('disallowed', [0, 1, 0])]):
        W1[INPUTS.index(name), j], W2[j] = 1, row  # neuron j copies one input: simplest.py's three rows
    steps = []

    def step(title, note=''):
        steps.append((title, decide(W1, b1, W2, []), decide(W1, b1, W2, ['traveling']), note))

    step("simplest.py's rules as 3 neurons")
    W1[4, 3], W2[3] = 1, [10, 0, 1]  # neuron 3 = traveling, worth +10
    step('+ neuron 3 = traveling, +10', '(shifts every option equally)')
    W1[2, 3], b1[3] = 1, -1          # rewire neuron 3: ReLU(offline + traveling - 1) needs both
    step('rewire neuron 3: + offline, -1', '(one neuron, one context rule)')

    print(f'{"":34}{"home":8}traveling')
    for title, home, traveling, note in steps:
        print(f'{title:34}{home:8}{traveling:13}{note}'.rstrip())
    print()
    for j in range(4):
        print(rule(W1, b1, W2, j))
    inputs = [[name for name, bit in zip(INPUTS, bits) if bit]
              for bits in itertools.product([0, 1], repeat=len(INPUTS))]  # every on/off combination
    holds = all((forward(W1, b1, W2, on)[1][1] > 0) == ('disallowed' in on)
                and (forward(W1, b1, W2, on)[0][3] > 0) == ({'offline', 'traveling'} <= set(on)) for on in inputs)
    print(f'\nchecked all {len(inputs)} on/off inputs: veto iff disallowed, neuron 3 iff offline and traveling: '
          + ('holds' if holds else 'FAILS'))
    return [(home, traveling) for _, home, traveling, _ in steps], holds


if __name__ == '__main__':
    main()
