"""The whole idea on one screen: a neural network whose weights you write by hand. No training."""
import numpy as np

FEATURES = ['low_cost', 'high_cost', 'offline', 'disallowed']  # inputs, one per row of W
OUTPUTS = ['utility', 'veto', 'evidence']                       # outputs, one per column of W


def pick(scores):
    """Drop vetoed options; pick the best utility or abstain."""
    allowed = {name: score for name, score in scores.items() if score[1] <= 0}
    if not allowed or any(score[2] <= 0 for score in allowed.values()):
        return 'abstain'  # some option has no stated preference behind it
    ranked = sorted(allowed, key=lambda name: -allowed[name][0])
    if len(ranked) > 1 and allowed[ranked[0]][0] == allowed[ranked[1]][0]:
        return 'abstain'  # tie
    return ranked[0]


def decide(W, options):
    """One matrix product per option, then pick."""
    return pick({name: np.isin(FEATURES, features).astype(float) @ W for name, features in options.items()})


def main():
    options = {'cheap': ['low_cost'], 'pricey': ['high_cost'], 'banned_cheap': ['low_cost', 'disallowed']}
    W = np.zeros((len(FEATURES), len(OUTPUTS)))  # 12 weights, all zero: no preferences yet
    steps = [('blank network', decide(W, options))]

    W[0] = [3, 0, 1]   # write "I prefer lower cost": utility, veto, evidence
    W[1] = [-3, 0, 1]
    W[3] = [0, 1, 0]   # write "never pick disallowed options": veto only
    steps.append(('after writing 3 rows', decide(W, options)))

    W[1, 0] = 5        # change one weight: now the pricey option is preferred
    steps.append(('after changing one weight', decide(W, options)))

    for title, decision in steps:
        print(f'{title:27} -> {decision}')
    print(f'\nweights (rows {", ".join(FEATURES)}; columns {", ".join(OUTPUTS)}):\n{W}')
    return [decision for _, decision in steps]


if __name__ == '__main__':
    main()
