"""Your directly editable neural weights. No training or preference inheritance.

Edit the tensor assignments in authored_weights(), then rerun the model.
Output columns are [utility, veto activation, preference evidence].
The commented examples are hypothetical; enable them only for your own preferences.
"""
import argparse
from pathlib import Path

import numpy as np

from network import forward as network_forward
from network import load_weights, save_weights, validate_weights, weights_digest

CONTEXTS = ['general'] + [f'context_{i}' for i in range(2, 17)]
ATTRIBUTES = ['low_cost', 'high_cost', 'offline', 'explicitly_disallowed'] + [
    f'attribute_{i}' for i in range(5, 33)
]
# Add or rename attributes here when you author a hard constraint.
# Their input values must be exactly 0 or 1, including when loading a .npz file.
BINARY_ATTRIBUTES = ['explicitly_disallowed']


def authored_weights():
    w1 = np.zeros((48, 48), dtype=np.float32)
    b1 = np.zeros(48, dtype=np.float32)
    w2 = np.zeros((48, 3), dtype=np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    # Neuron i activates attribute i only in the 'general' context:
    # ReLU(attribute + context - 1). Edit these connections for other contexts.
    for i in range(32):
        w1[i, i] = 1
        w1[32, i] = 1
        b1[i] = -1

    # All outputs stay zero until you directly author your numerical weights.
    # If you explicitly prefer lower cost, uncomment these two assignments:
    # w2[0] = [3, 0, 1]       # low_cost: utility, veto, evidence
    # w2[1] = [-3, 0, 1]      # high_cost
    # If explicitly_disallowed is an actual hard constraint, uncomment:
    # w2[3] = [-20, 1, 1]     # forbidden choices are vetoed even with --force

    # Hidden units 32..47 remain available for your own direct connections.
    return validate_weights((w1, b1, w2, b2))


def forward(x, weights=None):
    """Compatibility helper; inference uses the same shared tensor runtime."""
    return network_forward(x, authored_weights() if weights is None else weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('weights.npz'))
    parser.add_argument('--check', action='store_true', help='Verify the export matches the source assignments')
    args = parser.parse_args(argv)
    try:
        tensors = authored_weights()
        if args.check:
            if weights_digest(load_weights(args.output)) != weights_digest(tensors):
                raise ValueError('Export differs from source weights; export again or use --weights explicitly')
            print('Export matches direct source weights: ' + weights_digest(tensors))
        else:
            save_weights(args.output, tensors)
            print('Wrote ' + str(args.output) + ': 2,499 total parameters; no training')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
