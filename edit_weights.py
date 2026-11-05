"""Inspect or directly set a tensor entry in an NPZ weight snapshot. No fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from network import TENSOR_SHAPES, load_weights, save_weights, validate_weights, weights_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    inspect = actions.add_parser('inspect', help='Show tensor shapes, counts, and fingerprint')
    inspect.add_argument('weights', type=Path)
    edit = actions.add_parser('set', help='Assign one numeric tensor entry directly')
    edit.add_argument('weights', type=Path)
    edit.add_argument('--tensor', choices=TENSOR_SHAPES, required=True)
    edit.add_argument('--index', nargs='+', type=int, required=True)
    edit.add_argument('--value', type=float, required=True)
    edit.add_argument('--output', type=Path, required=True, help='Destination snapshot (may explicitly equal input)')
    args = parser.parse_args()
    try:
        tensors = load_weights(args.weights)
        result = {}
        if args.action == 'set':
            tensor = tensors[list(TENSOR_SHAPES).index(args.tensor)]
            index = tuple(args.index)
            if len(index) != tensor.ndim or any(i < 0 or i >= bound for i, bound in zip(index, tensor.shape)):
                raise ValueError(f'Index must address {args.tensor} with shape {tensor.shape}')
            if not np.isfinite(args.value) or abs(args.value) > float(np.finfo(np.float32).max):
                raise ValueError('Value must be finite and representable as float32')
            previous = float(tensor[index])
            tensor[index] = args.value
            tensors = validate_weights(tensors)
            save_weights(args.output, tensors)
            result.update(output=str(args.output), tensor=args.tensor, index=args.index, previous=previous, value=float(tensor[index]))
        result.update(
            total_parameters=sum(tensor.size for tensor in tensors),
            nonzero_parameters=sum(int(np.count_nonzero(tensor)) for tensor in tensors),
            tensors={name: list(shape) for name, shape in TENSOR_SHAPES.items()},
            weights_sha256=weights_digest(tensors),
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
