# Speed and size

```bash
uv run --locked benchmark.py --iterations 1000 --repeats 7 --history 1000
```

Measured November 5, 2026 on macOS 26.7 arm64, Python 3.13.11, NumPy 2.3.5 (median of 7 runs, 1,000 requests each, on a busy machine with load average about 50):

| Measurement | Result |
|---|---:|
| Two-option decision, in process | 33.8 µs |
| Same, with the verified-history guard | 34.1 µs |
| One request in a fresh process | 81.3 ms |
| 1,000 requests through one process, including startup | 133.3 ms |
| Apply 1,000 sequential patches | 87.0 ms |
| Verify a 1,000-patch history | 55.5 ms |
| Blank `.brain` | 1,356 bytes |
| Shipped example as `.brain` / as uncompressed NPZ | 1,690 / 10,962 bytes |

Python startup dominates single requests, so keep `brain.py serve` running for repeated calls. Each decision validates and fingerprints the tensors it uses, so a direct in-memory edit is never served stale. Compared with v0.2.0, warm decisions dropped from 60.7 to 34.9 ms per 1,000, and verifying a 1,000-edit history went from 13.5 MiB to 1.6 MiB peak traced memory.

The benchmark prints JSON and edits nothing on disk, and CI runs it as documented. Tests bound the memory of verifying a 1,000-edit history (under 4 MiB) but not wall-clock speed.
