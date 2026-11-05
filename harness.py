"""Generic subprocess adapter: one JSON request and response per line."""
import argparse
import json
from pathlib import Path
import sys

from network import parse_json
from model import DecisionModel, check_request_fields


def run(source=sys.stdin, sink=sys.stdout, weights=None, manifest=None, layout=None, model=None):
    if model is None:
        model = DecisionModel(weights, manifest, layout=layout)
    elif weights is not None or manifest is not None or layout is not None:
        raise ValueError('Supply a prepared model or weights/layout, not both')
    # Decode per line so one undecodable or deeply nested request fails alone.
    for line in getattr(source, 'buffer', source):
        if not line.strip():
            continue
        request_id = None
        try:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            request = parse_json(line)
            if not isinstance(request, dict):
                raise ValueError('Request must be an object')
            request_id = request.get('id')
            if request_id is not None and (isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
                request_id = None
                raise ValueError('Request id must be a string, integer, or null')
            check_request_fields(request)
            result = model.decide(request['question'], request['contexts'], request['options'], request.get('force', False))
            response = {'id': request_id, 'ok': True, 'result': result}
            encoded = json.dumps(response, allow_nan=False)
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
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
