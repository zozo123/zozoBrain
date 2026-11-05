"""Shared tensor validation and inference. No training or optimization."""
import hashlib
import json
from pathlib import Path
import tempfile
import tokenize
from types import ModuleType
import unicodedata
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

TENSOR_SHAPES = {'w1': (48, 48), 'b1': (48,), 'w2': (48, 3), 'b2': (3,)}


def check_label(value, label, limit=128):
    """Names are printed into terminals and agent handoffs: one printable line only."""
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(unicodedata.category(char) in ('Cc', 'Cf', 'Cs', 'Zl', 'Zp') for char in value)):
        raise ValueError(f'{label} must be one printable line of at most {limit} characters')
    return value


def validate_layout(layout):
    """Normalize the data-only feature schema independently of executable source."""
    if not isinstance(layout, dict) or set(layout) != {'attributes', 'contexts', 'binary_attributes'}:
        raise ValueError('Layout must contain attributes, contexts, and binary_attributes')
    result = {}
    for key, count in [('attributes', 32), ('contexts', 16)]:
        names = layout[key]
        if not isinstance(names, (list, tuple)) or len(names) != count:
            raise ValueError(f'Exactly {count} unique {key} required')
        for name in names:
            check_label(name, f'Each of the {key}')
        if len(set(names)) != count:
            raise ValueError(f'{key} must be unique')
        result[key] = list(names)
    binary = layout['binary_attributes']
    if not isinstance(binary, (list, tuple, set, frozenset)) or any(
        not isinstance(name, str) or name not in result['attributes'] for name in binary
    ):
        raise ValueError('binary_attributes must name known attributes')
    if len(set(binary)) != len(binary):
        raise ValueError('binary_attributes must be unique')
    result['binary_attributes'] = sorted(binary)
    return result


def layout_digest(attributes, contexts, binary_attributes):
    schema = {'attributes': list(attributes), 'contexts': list(contexts),
              'binary_attributes': sorted(binary_attributes), 'dtype': 'float32',
              'tensor_shapes': TENSOR_SHAPES}
    encoded = json.dumps(schema, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def load_weight_source(path):
    """Read authored source directly so rapid same-size edits cannot reuse stale pyc."""
    path = Path(path).resolve()
    module = ModuleType('authored_weight_source')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_weights(weights):
    """Return independent float32 tensors with the fixed 2,499-parameter layout."""
    if not isinstance(weights, (tuple, list)) or len(weights) != 4:
        raise ValueError('Weights must contain w1, b1, w2, b2')
    result = []
    for (name, shape), value in zip(TENSOR_SHAPES.items(), weights):
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in 'fiu':
            raise ValueError(f'{name} must be a real numeric tensor with shape {shape}')
        try:
            with np.errstate(over='raise', invalid='raise'):
                array = np.array(array, dtype=np.float32, copy=True)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise ValueError(f'{name} must contain finite float32 values') from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{name} must contain finite float32 values')
        result.append(array)
    return tuple(result)


def load_weights(path):
    """Load exactly four numeric tensors; never deserialize Python objects."""
    try:
        with Path(path).open('rb') as source:
            archive = np.load(source, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError('Expected an NPZ archive')
            with archive:
                if len(archive.files) != 4 or set(archive.files) != set(TENSOR_SHAPES):
                    raise ValueError('Weight archive must contain exactly w1, b1, w2, b2')
                return validate_weights(tuple(archive[name] for name in TENSOR_SHAPES))
    except (OSError, TypeError, ValueError, EOFError, BadZipFile, ZlibError, RuntimeError, tokenize.TokenError) as exc:
        raise ValueError(f'Cannot load weights from {path}: {exc}') from exc


def save_weights(path, weights):
    """Validate before atomically replacing an exported weight snapshot."""
    tensors = validate_weights(weights)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **dict(zip(TENSOR_SHAPES, tensors)))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def weights_digest(weights):
    """Hash named canonical float32 tensors, independent of ZIP timestamps."""
    return digest_validated_weights(validate_weights(weights))


def digest_validated_weights(tensors):
    """Internal hash for the independent snapshot returned by validate_weights.

    Callers must validate first. Hashing that same snapshot avoids another copy
    while keeping the fingerprint tied to the tensors used for this inference.
    """
    digest = hashlib.sha256()
    for name, tensor in zip(TENSOR_SHAPES, tensors):
        digest.update(name.encode('ascii') + b'\0')
        digest.update(tensor.astype('<f4', copy=False).tobytes(order='C'))
    return digest.hexdigest()


def neuron_outputs(x, weights):
    """Each hidden neuron's share of [utility, veto, evidence] for one input; rows sum to forward(x) - b2."""
    w1, b1, w2, _ = weights
    with np.errstate(over='ignore', invalid='ignore'):
        return np.maximum(0, x @ w1 + b1)[:, None] * w2


def forward(x, weights):
    w1, b1, w2, b2 = weights
    try:
        with np.errstate(over='raise', invalid='raise'):
            output = np.maximum(0, x @ w1 + b1) @ w2 + b2
    except FloatingPointError as exc:
        raise ValueError('Weights produce nonfinite activations; reduce their magnitude') from exc
    if not np.all(np.isfinite(output)):
        raise ValueError('Weights produce nonfinite activations; reduce their magnitude')
    return output
