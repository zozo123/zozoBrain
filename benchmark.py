"""Measure local latency without turning environment-specific timings into tests."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import timeit

import numpy as np

from brain_bundle import create_blank, pack_bundle, save_bundle
from model import DecisionModel
from network import save_weights
from weight_patches import diff, stack, validate_manifest


REQUEST = {'question': 'Which answer style?', 'contexts': ['communication'], 'options': [
    {'id': 'concise', 'attributes': {'concise': 1}},
    {'id': 'verified', 'attributes': {'verified': 1}},
]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--history', type=int, default=1000)
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.history) < 1:
        parser.error('iterations, repeats, and history must be positive')
    model = DecisionModel()
    _, empty = stack(model.weights, [], schema_sha256=model.schema_sha256)
    guarded = DecisionModel(model.weights, empty, layout=model.layout)

    def per_call_us(call):
        return statistics.median(timeit.repeat(call, number=args.iterations, repeat=args.repeats)) * 1e6 / args.iterations

    model.decide(**REQUEST)
    measurements = {
        'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform(),
        'iterations': args.iterations, 'repeats': args.repeats, 'statistic': 'median',
        'decision_us': per_call_us(lambda: model.decide(**REQUEST)),
        'decision_with_manifest_guard_us': per_call_us(lambda: guarded.decide(**REQUEST)),
    }
    patches = []
    current = tuple(tensor.copy() for tensor in model.weights)
    for i in range(args.history):
        edited = tuple(tensor.copy() for tensor in current)
        edited[2][0, 0] += 1
        patches.append(diff(current, edited, harness='benchmark', evidence='Synthetic performance measurement',
                            patch_id=f'benchmark-{i}', schema_sha256=model.schema_sha256))
        current = edited
    tensors, history = stack(model.weights, patches, schema_sha256=model.schema_sha256)
    measurements['history_edits'] = args.history
    measurements['history_stack_ms'] = statistics.median(timeit.repeat(
        lambda: stack(model.weights, patches, schema_sha256=model.schema_sha256),
        number=1, repeat=args.repeats,
    )) * 1000
    measurements['history_validation_ms'] = statistics.median(timeit.repeat(
        lambda: validate_manifest(history, tensors, schema_sha256=model.schema_sha256),
        number=1, repeat=args.repeats,
    )) * 1000
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        save_weights(folder / 'weights.npz', model.weights)
        save_bundle(folder / 'example.brain', pack_bundle(model.weights, model.layout))
        save_bundle(folder / 'blank.brain', create_blank())
        measurements['npz_bytes'] = (folder / 'weights.npz').stat().st_size
        measurements['example_brain_bytes'] = (folder / 'example.brain').stat().st_size
        measurements['blank_brain_bytes'] = (folder / 'blank.brain').stat().st_size
    request = json.dumps(REQUEST) + '\n'
    command = [sys.executable, str(Path(__file__).with_name('harness.py'))]

    def process(payload):
        subprocess.run(command, input=payload, text=True, capture_output=True, check=True)

    measurements['fresh_process_one_request_ms'] = statistics.median(timeit.repeat(
        lambda: process(request), number=1, repeat=args.repeats,
    )) * 1000
    measurements['one_process_all_requests_ms'] = statistics.median(timeit.repeat(
        lambda: process(request * args.iterations), number=1, repeat=args.repeats,
    )) * 1000
    print(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    main()
