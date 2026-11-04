# Portable brain formats

| File | Role | Why this format |
|---|---|---|
| `*.brain` | Portable active model, layout, and verified history | One data-only, atomically saved file; no mismatched snapshot/manifest paths |
| `weights.py` | Readable direct tensor assignments and feature layout | An agent or person can inspect and edit every weight |
| `*.npz` | Executable numeric snapshot | NumPy-native float32 tensors, loaded with pickle disabled |
| `*.json` weight patch | Versioned, attributable direct tensor changes | Reviewable before/after values with schema and base checks |
| `*.json` stack manifest | Ordered contribution history | Retains full patches, source evidence, and state fingerprints |
| JSON Lines stdin/stdout | Local inference transport | Works with any shell-capable harness without provider SDKs |

For everyday use, move the `.brain` file between compatible runtimes. To carry
the runtime too, move a generated personal profile directory with its locked
dependencies and CLI. Source/NPZ/patch interfaces remain available. The numbers do not contain arbitrary chat
memories: they encode only the decisions and features explicitly authored into
the network. Keep original evidence separately when a short reference is enough.

## Brain bundle v1

The extension `.brain` identifies a standard ZIP archive containing exactly five
members: `brain.json`, `w1.npy`, `b1.npy`, `w2.npy`, and `b2.npy`. Arrays use standard
NPY 1.0, C order, little-endian float32. NPY is a documented NumPy array format
designed to work inside ZIP archives. [NumPy format specification](https://numpy.org/neps/nep-0001-npy-format.html)

The writer uses DEFLATE level 1, fixed archive timestamps, and canonical JSON.
This keeps small sparse brains compact, writes quickly, and produces identical
archive bytes for the same complete state. DEFLATE is supported by the Python
3.11–3.13 runtime and common ZIP tooling. [Python ZIP documentation](https://docs.python.org/3.13/library/zipfile.html)

`brain.json` has exactly `format` (`zozobrain.brain.v1`), `name`, `layout`,
`weights_sha256`, `schema_sha256`, and `manifest`. The layout carries ordered
`attributes`, ordered `contexts`, and `binary_attributes`. The embedded manifest
uses the stack format below. A revision fingerprint hashes canonical complete
metadata, including the tensor fingerprint, so provenance-only changes are also
detected when updating the file.

Import validates names, sizes, tensor headers/shapes/types, finite values,
fingerprints, and the full contribution chain. It neither extracts archive paths
nor executes Python/pickle data. Tensor members are limited to 16 KiB each;
metadata to 16 MiB; archive size to 32 MiB. Oversized histories fail explicitly
without truncation. Keep long-term archives before starting any new baseline.

Saving replaces one complete file atomically. A writer lock and expected revision
reject stale same-file updates. The `.brain.lock` sidecar is OS locking
bookkeeping, contains no brain state, and is not needed when moving the brain.
There is no global active pointer: the named `.brain` file is the chosen state.

`brain.py pack` migrates an NPZ with its trusted local source layout and optional
existing manifest. Pack is explicit; normal `.brain` use reads the embedded
schema and does not execute `weights.py`. `.npz` remains a useful uncompressed
numeric interchange format; JSON patches remain useful for reviewing parallel
contributions, and JSONL remains the streaming inference transport.

## Tensor snapshots

An NPZ archive must contain exactly these numeric arrays:

| Name | Shape | Meaning |
|---|---|---|
| `w1` | `[48,48]` | Input-to-hidden connections |
| `b1` | `[48]` | Hidden biases |
| `w2` | `[48,3]` | Hidden-to-output connections |
| `b2` | `[3]` | Output biases |

The runtime canonicalizes real numeric values to finite float32. A
`weights_sha256` is computed from the names and little-endian tensor bytes in the
order above, independently of ZIP metadata. The schema digest additionally binds
the attribute names/order, contexts/order, binary input names, dtype, and shapes.
Profiles with different feature layouts are incompatible even when their tensor
shapes match; profile directory/owner names alone do not affect compatibility.

## Weight patch v1

`weight_patches.py diff` creates the exact v1 structure below. Placeholder digest
strings in this illustration must be replaced by the generated 64-character
SHA-256 values; prefer generating a patch from actual snapshots.

```json
{
  "format": "zozobrain.weight-patch.v1",
  "id": "codex-001",
  "source": {"harness": "codex", "evidence": "Reference to an explicit user instruction"},
  "schema_sha256": "<schema digest>",
  "base_weights_sha256": "<base tensor digest>",
  "edits": [
    {"tensor": "w2", "index": [0, 0], "before": 3.0, "after": 6.0,
     "reason": "Reference to the instruction motivating this numerical hypothesis"}
  ]
}
```

Each edit directly assigns one entry. Values, coordinates, exact field names,
unique entries, and nonempty evidence are validated. Source/evidence are
caller-provided provenance, not cryptographic proof that a harness ran or that a
statement is true. Keep numbers and evidence reviewable.

## Combining contributions

Stacking checks each patch's schema, declared base, and before values. Independent
edits from the same base combine; edits to the same entry that disagree fail.
An intentional revision must be authored against the current accumulated state.
Overlapping edits that agree on the exact new value retain both sources without
applying the same change twice.
An exact duplicate patch ID/content is skipped; reusing an ID with different
content fails. Compatible contributions remain one 2,499-parameter network;
there is no averaging of model weights or growth in parameter count.

The `zozobrain.weight-stack.v1` manifest records original/final tensor digests,
schema digest, full ordered patches, their input/output digests, per-entry
`edit_actions` (`applied` or `already_present`), and skipped duplicates. Preserve it when continuing in another session with
`--resume-manifest`. Keep the original snapshots and patches if you want to
rebuild or audit the complete chain.

JSONL inference requests/responses are documented in [HARNESS.md](HARNESS.md).
All weights and their schema remain local; tools perform no network calls during
inference, direct editing, diffing, or stacking.
