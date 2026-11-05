"""Shared tensor validation and inference. No training or optimization."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import tokenize
from types import ModuleType
import unicodedata
import zipfile
from zipfile import BadZipFile
from zlib import error as ZlibError

import numpy as np

TENSOR_SHAPES = {'w1': (48, 48), 'b1': (48,), 'w2': (48, 3), 'b2': (3,)}


def unique_object(pairs):
    """JSON object hook: a repeated key is an error, never a silent last-one-wins."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {json.dumps(key)}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON number is not allowed: {value}')


def parse_json(text):
    """Parse untrusted JSON: repeated keys and NaN or Infinity are errors."""
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=_reject_constant)


def read_json_object(path, label='File'):
    """Read a UTF-8 JSON file that must hold one object. Its content is never taken as another path."""
    value = parse_json(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'{label} {path} must contain a JSON object')
    return value


def read_npy(raw, name, exact=True):
    """Parse one NPY member, checking its header before allocating anything from it.

    exact: require C-order little-endian float32 (the .brain format); otherwise any real
    numeric dtype up to 8 bytes, which validate_weights then converts to float32.
    """
    stream = io.BytesIO(raw)
    try:
        major, _ = np.lib.format.read_magic(stream)
        if major not in ((1,) if exact else (1, 2)):
            raise ValueError
        reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=1024)
    except Exception as exc:  # crafted headers can raise tokenizer or syntax errors
        raise ValueError(f'{name} has an invalid NPY header') from exc
    expected = TENSOR_SHAPES[name]
    allowed = dtype == np.dtype('<f4') if exact else dtype.kind in 'fiu' and dtype.itemsize <= 8
    if shape != expected or fortran or not allowed:
        kind = 'C-order little-endian float32' if exact else 'C-order real numeric values'
        raise ValueError(f'{name} requires {kind} with shape {expected}')
    if len(raw) - stream.tell() != int(np.prod(expected)) * dtype.itemsize:
        raise ValueError(f'{name} has an invalid tensor byte count')
    return np.frombuffer(raw, dtype=dtype, offset=stream.tell()).reshape(expected).copy()


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
    """Load exactly four numeric tensors from an NPZ; never trust its headers or deserialize objects."""
    try:
        with Path(path).open('rb') as source, zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if sorted(member.filename for member in members) != sorted(f'{name}.npy' for name in TENSOR_SHAPES):
                raise ValueError('Weight archive must contain exactly w1, b1, w2, b2')
            tensors = []
            for name in TENSOR_SHAPES:
                member = archive.getinfo(f'{name}.npy')
                # The largest valid member is a 48 x 48 float64 array plus its header.
                if member.file_size > 32 * 1024 or member.flag_bits & 1:
                    raise ValueError(f'Invalid or oversized weight member: {member.filename}')
                with archive.open(member) as handle:
                    raw = handle.read(member.file_size + 1)
                if len(raw) != member.file_size:
                    raise ValueError(f'Invalid weight member size: {member.filename}')
                tensors.append(read_npy(raw, name, exact=False))
            return validate_weights(tuple(tensors))
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
    """Each hidden neuron's share of [utility, veto, evidence] for one input.

    The products are float64: a float32 share can overflow even when forward's total stays finite.
    """
    w1, b1, w2, _ = weights
    with np.errstate(over='ignore', invalid='ignore'):
        hidden = np.maximum(0, x @ w1 + b1)
    return hidden.astype(np.float64)[:, None] * w2.astype(np.float64)


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
