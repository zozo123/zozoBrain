# File formats

| File | Holds |
|---|---|
| `*.brain` | Tensors, feature layout, and verified history in one data-only file. This is the one to move around |
| `*.patch.json` | One contribution: exact before/after values with harness and evidence |
| `*.npz` + stack manifest | Source-workflow snapshot and its ordered history |
| `weights.py` | Readable NumPy assignments of the shipped example |

Nothing here contains chat history, only the numbers and the evidence references you supply. No tool makes network calls.

## Brain bundle v1

A standard ZIP with exactly five members: `brain.json`, `w1.npy`, `b1.npy`, `w2.npy`, `b2.npy`. The tensors are [NPY 1.0](https://numpy.org/neps/nep-0001-npy-format.html) arrays, C-order little-endian float32, with the shapes in [WEIGHTS.md](WEIGHTS.md). Members use DEFLATE level 1 and fixed timestamps, and `brain.json` is canonical JSON, so the same state always produces the same bytes.

`brain.json` has exactly `format` (`zozobrain.brain.v1`), `name`, `layout` (`attributes`, `contexts`, `binary_attributes`), `weights_sha256`, `schema_sha256`, and `manifest` (the stack manifest below). A revision hash covers all of it.

Loading checks member names and declared sizes (16 KiB per tensor, 16 MiB of metadata, 32 MiB per file) and decompresses no more than those sizes. It also checks NPY headers before allocating, requires finite values, and verifies the fingerprints. Finally it replays the whole history against the tensors. Nothing is extracted to disk and nothing is executed. Names must be single printable lines. Saving writes a temporary file and renames it, under a lock, with a revision check. The `.brain.lock` sidecar is local bookkeeping and does not need to travel with the brain.

`weights_sha256` hashes each tensor's name and little-endian bytes in the order `w1, b1, w2, b2`. `schema_sha256` binds the attribute and context names and order, the binary inputs, the dtype, and the shapes. Brains with different layouts cannot exchange patches.

## Weight patch v1

```json
{
  "format": "zozobrain.weight-patch.v1",
  "id": "codex-001",
  "source": {"harness": "codex", "evidence": "My statement of 2026-11-04"},
  "schema_sha256": "<64 hex>",
  "base_weights_sha256": "<64 hex>",
  "edits": [{"tensor": "w2", "index": [0, 0], "before": 3.0, "after": 6.0, "reason": "My statement of 2026-11-04"}]
}
```

Each edit assigns one entry. Merging checks the schema, the declared base, and every `before` value. Agreeing overlaps are recorded as `already_present`; disagreements fail. A duplicate patch (same id and content) is skipped, and a reused id with different content fails. Duplicate JSON keys are rejected. The harness and evidence fields are the caller's claims, not proof.

## Stack manifest v1

`zozobrain.weight-stack.v1` records the original and final tensor digests, the schema digest, every patch in order with its input and output digests and per-edit `edit_actions`, and skipped duplicate ids. Inside a `.brain` it is the history. For NPZ snapshots, `weight_patches.py stack --resume-manifest` continues it across sessions.
