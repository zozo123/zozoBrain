# Speed and size

```bash
uv run --locked benchmark.py --iterations 1000 --repeats 7 --history 1000
```

Measured November 4, 2026 on macOS 26.7 arm64, Python 3.13.11, NumPy 2.3.5 (median of 7 runs, 1,000 requests each, on a busy machine with load average 36):

| Measurement | Result |
|---|---:|
| Two-option decision, in process | 34.1 µs |
| Same, with the verified-history guard | 34.0 µs |
| One request in a fresh process | 170.6 ms |
| 1,000 requests through one process, including startup | 213.6 ms |
| Apply 1,000 sequential patches | 77.8 ms |
| Verify a 1,000-patch history | 54.7 ms |
| Blank `.brain` | 1,354 bytes |
| Shipped example as `.brain` / as uncompressed NPZ | 1,690 / 10,962 bytes |

Python startup dominates single requests (101.5 ms and 151.1 ms for the two process rows on an idle machine earlier the same day), so keep `brain.py serve` running for repeated calls. Each decision validates and fingerprints the tensors it uses, so a direct in-memory edit is never served stale. Compared with v0.2.0, warm decisions dropped from 60.7 to 34.9 ms per 1,000, and verifying a 1,000-edit history went from 13.5 MiB to 1.6 MiB peak traced memory.

The benchmark prints JSON and edits nothing on disk. Tests check behavior, not wall-clock speed.
