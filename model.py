"""Run a directly authored 2,499-parameter personal decision network."""
import argparse
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np

from network import forward, layout_digest, load_weight_source, load_weights, validate_weights, weights_digest

_source = load_weight_source(Path(__file__).with_name('weights.py'))
ATTRIBUTES = _source.ATTRIBUTES
BINARY_ATTRIBUTES = _source.BINARY_ATTRIBUTES
CONTEXTS = _source.CONTEXTS
authored_weights = _source.authored_weights


def validate_question(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question must be a nonempty string')
    if len(re.findall(r'\w+|[^\w\s]', question, re.UNICODE)) > 100:
        raise ValueError('Question exceeds 100 regex word/punctuation tokens')


def validate_schema():
    for names, count, label in [(ATTRIBUTES, 32, 'attributes'), (CONTEXTS, 16, 'contexts')]:
        if not isinstance(names, (list, tuple)) or len(names) != count:
            raise ValueError(f'Exactly {count} unique {label} required')
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != count:
            raise ValueError(f'{label} must be unique nonempty strings')
    if not isinstance(BINARY_ATTRIBUTES, (list, tuple, set, frozenset)) or any(
        name not in ATTRIBUTES for name in BINARY_ATTRIBUTES
    ):
        raise ValueError('BINARY_ATTRIBUTES must name known attributes')


class DecisionModel:
    def __init__(self, weights=None, manifest=None):
        validate_schema()
        if isinstance(weights, (str, Path)):
            weights = load_weights(weights)
        self.weights = validate_weights(authored_weights() if weights is None else weights)
        self.parameter_count = sum(array.size for array in self.weights)
        self.schema_sha256 = layout_digest(ATTRIBUTES, CONTEXTS, BINARY_ATTRIBUTES)
        self.contributing_harnesses = None
        self._manifest_weights_sha256 = None
        if manifest is not None:
            from weight_patches import validate_manifest
            if isinstance(manifest, (str, Path)):
                manifest = json.loads(Path(manifest).read_text(encoding='utf-8'))
            provenance = validate_manifest(manifest, self.weights)
            if provenance['schema_sha256'] != self.schema_sha256:
                raise ValueError('Manifest schema changed after model initialization; restart the process')
            self._manifest_weights_sha256 = provenance['final_weights_sha256']
            self.contributing_harnesses = list(dict.fromkeys(
                record['patch']['source']['harness'] for record in provenance['patches']
            ))

    def decide(self, question, contexts, options, force=False):
        validate_question(question)
        if not isinstance(force, bool):
            raise ValueError('force must be boolean')
        if not isinstance(contexts, (list, tuple)) or not contexts or any(
            not isinstance(context, str) or context not in CONTEXTS for context in contexts
        ):
            raise ValueError('Supply a list of known contexts: ' + ', '.join(CONTEXTS))
        if len(set(contexts)) != len(contexts):
            raise ValueError('Contexts must be unique')
        if not isinstance(options, (list, tuple)) or len(options) < 2:
            raise ValueError('At least two options required in a list')
        ids = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError('Each option must be an object')
            option_id = option.get('id')
            if not isinstance(option_id, str) or not option_id.strip() or option_id in ids:
                raise ValueError('Option ids must be unique nonempty strings')
            ids.append(option_id)
            if 'label' in option and not isinstance(option['label'], str):
                raise ValueError('Option labels must be strings')
        tensors = validate_weights(self.weights)
        if self._manifest_weights_sha256 is not None and weights_digest(tensors) != self._manifest_weights_sha256:
            raise ValueError('Weights changed after manifest verification; capture a new contribution first')
        scored = []
        for option in options:
            attrs = option.get('attributes', {})
            if not isinstance(attrs, dict):
                raise ValueError('Option attributes must be an object')
            unknown = [name for name in attrs if name not in ATTRIBUTES]
            if unknown:
                raise ValueError('Unknown attributes: ' + str(unknown))
            x = np.zeros(48, dtype=np.float32)
            for attr, value in attrs.items():
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 <= value <= 1:
                    raise ValueError('Attributes must be finite numeric values in [0,1]')
                if attr in BINARY_ATTRIBUTES and value not in (0, 1):
                    raise ValueError('Hard constraint attributes must be 0 or 1')
                x[ATTRIBUTES.index(attr)] = value
            for context in contexts:
                x[32 + CONTEXTS.index(context)] = 1
            utility, veto, support = forward(x, tensors).tolist()
            scored.append({
                'id': option['id'], 'label': option.get('label', option['id']),
                'utility': utility, 'veto': veto > 0, 'modeled_evidence': support > 0,
                'active_attributes': [attr for attr, value in attrs.items() if value > 0],
            })
        ranked = sorted((row for row in scored if not row['veto']), key=lambda row: -row['utility'])
        vetoed = [row for row in scored if row['veto']]
        status = 'decision'
        if not ranked:
            status = 'no_feasible_option'
        elif len(ranked) == 1 and vetoed:
            status = 'only_feasible_option'
        elif not all(row['modeled_evidence'] for row in ranked):
            status = 'insufficient_preference_evidence'
        elif np.isclose(ranked[0]['utility'], ranked[1]['utility'], rtol=1e-6, atol=1e-7):
            status = 'tie'
        selected = ranked[0]['id'] if ranked and (force or status in ('decision', 'only_feasible_option')) else None
        result = {
            'decision': selected, 'status': status,
            'forced': bool(force and status in ('tie', 'insufficient_preference_evidence')),
            'ranking': ranked, 'vetoed': vetoed, 'weights_sha256': weights_digest(tensors),
            'schema_sha256': self.schema_sha256,
            'note': 'Authored utility policy, not calibrated probability. Attributes and contexts are supplied by caller; question text is display metadata, not language understanding.',
        }
        if self.contributing_harnesses is not None:
            result['contributing_harnesses'] = self.contributing_harnesses
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('decision_file', type=Path)
    parser.add_argument('--weights', type=Path, help='Load an edited NPZ snapshot; default: authored weights.py')
    parser.add_argument('--manifest', type=Path, help='Verify matching stack provenance and report contributing harnesses')
    parser.add_argument('--force', action='store_true', help='Resolve unknowns/ties by score then input order; never override veto')
    args = parser.parse_args()
    try:
        payload = json.loads(args.decision_file.read_text())
        result = DecisionModel(args.weights, args.manifest).decide(payload['question'], payload['contexts'], payload['options'], args.force)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
