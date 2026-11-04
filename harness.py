"""Generic subprocess adapter: one JSON request and response per line."""
import argparse
import json
from pathlib import Path
import sys

from model import DecisionModel


def reject_nonfinite(value):
    raise ValueError(f'Nonfinite JSON number is not allowed: {value}')


def run(source=sys.stdin, sink=sys.stdout, weights=None, manifest=None, layout=None, model=None):
    if model is None:
        model = DecisionModel(weights, manifest, layout=layout)
    elif weights is not None or manifest is not None or layout is not None:
        raise ValueError('Supply a prepared model or weights/layout, not both')
    for line in source:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line, parse_constant=reject_nonfinite)
            if not isinstance(request, dict):
                raise ValueError('Request must be an object')
            request_id = request.get('id')
            if request_id is not None and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
                request_id = None
                raise ValueError('Request id must be a string, integer, or null')
            result = model.decide(request['question'], request['contexts'], request['options'], request.get('force', False))
            response = {'id': request_id, 'ok': True, 'result': result}
            encoded = json.dumps(response, allow_nan=False)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            encoded = json.dumps({'id': request_id, 'ok': False, 'error': str(exc)}, allow_nan=False)
        sink.write(encoded + '\n')
        sink.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, help='Load an edited NPZ snapshot; default: authored weights.py')
    parser.add_argument('--manifest', type=Path, help='Verify matching stack provenance and report contributing harnesses')
    args = parser.parse_args()
    try:
        run(weights=args.weights, manifest=args.manifest)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'error: {exc}\n')
