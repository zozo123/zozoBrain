"""A portable, data-only brain: four standard NPY tensors, schema, and provenance."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
from numbers import Real
import os
from pathlib import Path
import stat
import tempfile
import uuid
import zipfile
from zlib import error as ZlibError

import numpy as np

from network import (TENSOR_SHAPES, check_label, layout_digest, parse_json, read_npy, unique_object, validate_layout,
                     validate_weights, weights_digest)
from weight_patches import STACK_FORMAT, diff, stack, validate_manifest

FORMAT = 'zozobrain.brain.v1'
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_TENSOR_BYTES = 16384
MEMBERS = {'brain.json', *(name + '.npy' for name in TENSOR_SHAPES)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _metadata(bundle):
    return {'format': FORMAT, 'name': bundle.name, 'layout': bundle.layout,
            'weights_sha256': bundle.weights_sha256, 'schema_sha256': bundle.schema_sha256,
            'manifest': bundle.manifest}


@dataclass
class BrainBundle:
    name: str
    layout: dict
    weights: tuple
    manifest: dict

    @property
    def weights_sha256(self):
        return weights_digest(self.weights)

    @property
    def schema_sha256(self):
        return layout_digest(**self.layout)

    @property
    def revision_sha256(self):
        return hashlib.sha256(_canonical(_metadata(self)).encode('utf-8')).hexdigest()

    @property
    def contributing_harnesses(self):
        return list(dict.fromkeys(record['patch']['source']['harness'] for record in self.manifest['patches']))


def pack_bundle(weights, layout, manifest=None, name='brain'):
    check_label(name, 'Brain name')
    layout = validate_layout(layout)
    tensors = validate_weights(weights)
    fingerprint = weights_digest(tensors)
    schema = layout_digest(**layout)
    if manifest is None:
        manifest = {'format': STACK_FORMAT, 'schema_sha256': schema,
                    'original_weights_sha256': fingerprint, 'final_weights_sha256': fingerprint,
                    'patches': [], 'duplicate_patch_ids_skipped': []}
    manifest = validate_manifest(manifest, tensors, schema_sha256=schema)
    return BrainBundle(name.strip(), layout, tensors, manifest)


def create_blank(name='my-brain', layout=None):
    layout = validate_layout(layout) if layout is not None else {
        'attributes': ['low_cost', 'high_cost', 'offline', 'explicitly_disallowed']
                      + [f'attribute_{i}' for i in range(5, 33)],
        'contexts': ['general', 'traveling'] + [f'context_{i}' for i in range(3, 17)],
        'binary_attributes': ['explicitly_disallowed'],
    }
    w1 = np.zeros((48, 48), dtype=np.float32)
    b1 = np.zeros(48, dtype=np.float32)
    w2 = np.zeros((48, 3), dtype=np.float32)
    b2 = np.zeros(3, dtype=np.float32)
    for i in range(32):
        w1[i, i] = 1
        w1[32, i] = 1
        b1[i] = -1
    return pack_bundle((w1, b1, w2, b2), layout, name=name)


_unique_object = unique_object  # kept for callers of earlier versions


def _read_member(archive, name):
    # Declared sizes were checked against the limits; never inflate past them,
    # whatever the compressed stream holds.
    size = archive.getinfo(name).file_size
    with archive.open(name) as handle:
        data = handle.read(size + 1)
    if len(data) != size:
        raise ValueError(f'Invalid brain member size: {name}')
    return data


def _read_tensor(raw, name):
    return read_npy(raw, name, exact=True)


def load_bundle(path):
    path = Path(path)
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('Brain archive exceeds the 32 MiB limit')
        with path.open('rb') as source, zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != len(MEMBERS) or set(names) != MEMBERS:
                raise ValueError('Brain archive must contain exactly brain.json and four named NPY tensors')
            for member in archive.infolist():
                limit = MAX_METADATA_BYTES if member.filename == 'brain.json' else MAX_TENSOR_BYTES
                if member.file_size > limit or member.flag_bits & 1:
                    raise ValueError(f'Invalid or oversized brain member: {member.filename}')
                if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError('Brain archives support only stored or DEFLATE members')
            metadata = parse_json(_read_member(archive, 'brain.json').decode('utf-8'))
            if not isinstance(metadata, dict) or set(metadata) != {
                'format', 'name', 'layout', 'weights_sha256', 'schema_sha256', 'manifest'
            }:
                raise ValueError('Invalid brain metadata fields')
            if metadata['format'] != FORMAT:
                raise ValueError(f'Unsupported brain format; expected {FORMAT}')
            if not isinstance(metadata['manifest'], dict):
                raise ValueError('The brain history (manifest) must be a JSON object')
            tensors = tuple(_read_tensor(_read_member(archive, name + '.npy'), name)
                            for name in TENSOR_SHAPES)
        bundle = pack_bundle(tensors, metadata['layout'], metadata['manifest'], metadata['name'])
        if bundle.weights_sha256 != metadata['weights_sha256'] or bundle.schema_sha256 != metadata['schema_sha256']:
            raise ValueError('Brain tensor or schema fingerprint does not match metadata')
        return bundle
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, ZlibError, RuntimeError, RecursionError) as exc:
        raise ValueError(f'Cannot load brain {path}: {exc}') from exc


@contextmanager
def _write_lock(path):
    # Keep the empty sidecar inode stable: removing an OS lock file creates a
    # race between writers. The sidecar is bookkeeping, never part of the brain.
    with path.with_name(path.name + '.lock').open('a+b') as handle:
        locked = False
        try:
            try:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0, os.SEEK_END)
                    if not handle.tell():
                        handle.write(b'\0')
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as exc:
                raise ValueError('Brain is being edited by another process; retry or use a separate copy') from exc
            yield
        finally:
            if locked:
                if os.name == 'nt':
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_bundle(path, bundle, overwrite=False, expected_revision=None):
    bundle = pack_bundle(bundle.weights, bundle.layout, bundle.manifest, bundle.name)
    metadata = _canonical(_metadata(bundle)).encode('utf-8')
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError('Brain history exceeds the 16 MiB metadata limit')
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    with _write_lock(path):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f'Brain already exists: {path}; choose a new path')
            if expected_revision is not None and load_bundle(path).revision_sha256 != expected_revision:
                raise ValueError('Brain changed since it was loaded; reload before applying your edit')
        elif expected_revision is not None:
            raise ValueError('Brain disappeared since it was loaded; choose a new output path')
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.brain', delete=False) as handle:
                temporary = Path(handle.name)
                with zipfile.ZipFile(handle, 'w') as archive:
                    members = {'brain.json': metadata}
                    for name, tensor in zip(TENSOR_SHAPES, bundle.weights):
                        data = io.BytesIO()
                        np.save(data, np.ascontiguousarray(tensor, dtype='<f4'), allow_pickle=False)
                        members[name + '.npy'] = data.getvalue()
                    for name, data in members.items():
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        info.create_system = 3
                        info.external_attr = 0o600 << 16
                        archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def diff_bundles(base, edited, *, harness, evidence, patch_id=None):
    if base.schema_sha256 != edited.schema_sha256:
        raise ValueError('Cannot diff brains with different feature layouts')
    # Without a shared history, the diff would attribute reverts of another
    # agent's edits to this harness.
    history = base.manifest['patches']
    if (edited.manifest['original_weights_sha256'] != base.manifest['original_weights_sha256']
            or edited.manifest['patches'][:len(history)] != history):
        raise ValueError('The edited brain is not a copy of this base; diff against the brain it was copied from')
    return diff(base.weights, edited.weights, harness=harness, evidence=evidence,
                patch_id=patch_id or f'{harness}-{uuid.uuid4().hex}', schema_sha256=base.schema_sha256)


def apply_patches(bundle, patches):
    tensors, manifest = stack(bundle.weights, patches, previous_manifest=bundle.manifest,
                              schema_sha256=bundle.schema_sha256)
    return pack_bundle(tensors, bundle.layout, manifest, bundle.name)


def _assign(tensors, tensor, index, values, touched):
    if tensor not in TENSOR_SHAPES:
        raise ValueError('Tensor must be one of: ' + ', '.join(TENSOR_SHAPES))
    shape = TENSOR_SHAPES[tensor]
    if not isinstance(index, (tuple, list)) or not 1 <= len(index) <= len(shape) or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < bound for i, bound in zip(index, shape)
    ):
        raise ValueError(f'Index must select an entry or row within {tensor} {shape}')
    if not isinstance(values, (tuple, list)) or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values
    ):
        raise ValueError('Values must be finite float32 numbers')
    try:
        values = [float(value) for value in values]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError('Values must be finite float32 numbers') from exc
    if any(not np.isfinite(value) or abs(value) > float(np.finfo(np.float32).max) for value in values):
        raise ValueError('Values must be finite float32 numbers')
    if np.any(touched[tensor][tuple(index)]):
        raise ValueError(f'{tensor}{list(index)} overlaps an earlier assignment in this edit; give each entry once')
    touched[tensor][tuple(index)] = True
    target = tensors[list(TENSOR_SHAPES).index(tensor)]
    selected_shape = target[tuple(index)].shape
    count = int(np.prod(selected_shape)) if selected_shape else 1
    if len(values) != count:
        raise ValueError(f'Selected {tensor}{tuple(index)} needs exactly {count} values')
    target[tuple(index)] = np.asarray(values, dtype=np.float32).reshape(selected_shape)


def edit_bundle(bundle, *, tensor=None, index=None, values=None, assignments=None, harness='manual', evidence,
                patch_id=None):
    """Assign one entry or row, or several (tensor, index, values) assignments, as one verified edit."""
    if assignments is None:
        assignments = [(tensor, index, values)]
    elif tensor is not None or index is not None or values is not None:
        raise ValueError('Give either one tensor, index, and values or a list of assignments')
    if not isinstance(assignments, (tuple, list)) or not assignments or any(
        not isinstance(assignment, (tuple, list)) or len(assignment) != 3 for assignment in assignments
    ):
        raise ValueError('Assignments must be a nonempty list of (tensor, index, values)')
    tensors = list(validate_weights(bundle.weights))
    touched = {name: np.zeros(shape, dtype=bool) for name, shape in TENSOR_SHAPES.items()}
    for tensor, index, values in assignments:
        _assign(tensors, tensor, index, values, touched)
    edited = BrainBundle(bundle.name, bundle.layout, tuple(tensors), bundle.manifest)
    patch = diff_bundles(bundle, edited, harness=harness, evidence=evidence, patch_id=patch_id)
    if not patch['edits']:
        return pack_bundle(bundle.weights, bundle.layout, bundle.manifest, bundle.name)
    return apply_patches(bundle, [patch])
