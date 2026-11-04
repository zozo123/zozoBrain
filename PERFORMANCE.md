# Format and latency measurements

Run `uv run --locked benchmark.py` to measure your environment. Timings are
observations, not CI thresholds or model accuracy claims.

Reference environment: macOS 26.7, arm64, Python 3.13.11, NumPy 2.3.5. Measurements
used 1,000 requests and median of seven runs on November 4, 2026.

| Measurement | Result |
|---|---:|
| Two-option in-process decision | 36.0 microseconds/request |
| Same decision with verified-manifest mutation guard | 36.6 microseconds/request |
| One fresh Python process, one request | 101.5 ms total |
| One persistent process, 1,000 requests | 151.1 ms total, including startup |
| Combine 1,000 synthetic sequential weight patches | 80.4 ms |
| Verify their complete history | 79.6 ms |
| Shipped weights as uncompressed NPZ | 10,962 bytes |
| Shipped weights + layout + initial history as `.brain` | 1,690 bytes |
| Blank personal `.brain` | 1,354 bytes |

Warm latency excludes request JSON parsing/serialization; subprocess latency
includes both. Subprocess measurements use the legacy JSONL entry point and the
same shared inference engine as `brain.py serve`. Persistent processes are the
preferred path for repeated agent calls because Python startup dominates a
single decision. A long-lived process holds a snapshot; restart after saving a
new brain.

## Changes from v0.2.0

The same local comparison before/after optimization measured 1,000 warm
two-option decisions at 60.7 ms versus 34.9 ms, and verified-manifest decisions
at 76.1 ms versus 35.9 ms. Inference now validates/copies one independent tensor
snapshot and hashes that same snapshot once. Input indices are precomputed and
scalar tie checks preserve the prior float32 tolerance. No mutable-weight cache
can hide a direct edit: the next decision still observes it and updates its
fingerprint, or rejects it when verified provenance no longer matches.

Patch operations resolve the source schema once per operation. Data-only bundle
operations supply their embedded schema directly. History verification replays
the chain while retaining only historical states referenced by later patches.
A separate traced-memory audit of 1,000 linear edits measured peak memory at
13.53 MiB before and 1.58 MiB after; validation under tracing fell from 2.815 s to
0.291 s. Tracing timings differ from the untraced measurements above. Dense
branch histories can require more retained state than a linear chain.

## Reproduce

```bash
uv run --locked benchmark.py --iterations 1000 --repeats 7 --history 1000
```

The command prints machine-readable JSON with environment, latency, and size
measurements. Its history is synthetic, created solely for this measurement.
Arrays and existing brains are not edited. Tests check unchanged behavior,
direct-mutation detection, corrupt input handling, and history correctness;
they do not assert wall-clock speed. Compression gains depend on tensor sparsity
and history size. General dense large-model storage has different requirements.
