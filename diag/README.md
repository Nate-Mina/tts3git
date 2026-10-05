# diag/ — diagnostic probes

Throwaway scripts written while bringing MegaTTS3 up on DirectML (AMD RX 6700 XT).
Kept for reference; none are part of the app.

Run any of them from the project root with the DML venv:

```
PYTHONPATH="" ./env-dml/Scripts/python.exe diag/<script>.py
```

`PYTHONPATH=""` matters: the ambient `PYTHONPATH` points at another venv's
site-packages and shadows this one (wrong-ABI numpy etc.).

## What each one answers

| script | question |
|---|---|
| `hermes-stage.py` | VRAM after each pipeline stage (load → dit → decode) |
| `hermes-pool-growth.py`, `hermes-pool2.py` | does the DML pool leak or plateau across requests? |
| `hermes-longrun.py` | long-lived process, varied text lengths — finds the ratchet |
| `hermes-threshold.py`, `hermes-seq-vs-chars.py` | is failure predicted by text length? (**no** — see below) |
| `hermes-attn-math.py` | closed-form O(seq²) attention cost vs. the card |
| `hermes-variant.py` | decode variants: immediate / free-first / CPU fallback |
| `hermes-tiled2.py` | `disable_tiled_resources` — it makes things **worse** |
| `hermes-release.py` | does dropping refs + gc return DML memory? |
| `hermes-app-repro.py` | app-level reproduction over HTTP |
| `hermes-verify-longfix.py` | the long-request OOM, before/after |
| `hermes-chunk-probe.py` | does `chunk_text_english` enforce its own `max_chars`? |
| `hermes-firstreq-probe.py`, `hermes-warmup-probe.py` | early (disproven) warm-up theory |
| `hermes-verify-chunkfix.py`, `hermes-verify-retry.py`, `hermes-verify-retry4.py` | the two fixes, before/after |
| `hermes-tempdir-why.py`, `hermes-who-writes.py`, `hermes-http-contain.py` | which side (server vs client) writes where |
| `hermes-verify-contained.py`, `hermes-verify-final-contained.py` | artifact containment |

## Findings worth keeping

- **DirectML has no `empty_cache`.** `torch_directml.gpu_memory()` returns all
  zeros; the only honest probe is to try an allocation.
- **Probe with `torch_directml.device()`, not the `'privateuseone:0'` string.**
  The string raises `ModuleNotFoundError: No module named 'torch.privateuseone'`
  unless the backend is already registered — a bare `except` turns that into a
  false "no memory".
- **The allocator is tile-based** and caps a single allocation at ~3 GB on a
  12 GB card (3072 MB ok, 4096 MB refused on a near-empty device).
- **Freed memory goes to a per-process pool**, not the driver. Run hardware
  probes in their own subprocess or each one poisons the next.
- **The pool ratchets**: once a long request runs, usage stays pinned near the
  ceiling (~11.7 of 11.8 GB), so a later request can fail on a ~10 MB
  allocation. This is the "finished but no audio" symptom.
- **Failure is NOT predicted by input length.** A 144-char request passed at
  11.35 GB while a 127-char one failed at 11.73 GB, and a short request can fail
  right after a long one succeeds. It is fragmentation near the ceiling. Two
  earlier theories here — "the first request after load" and "long text is the
  wall" — were both wrong; the sweep in `hermes-seq-vs-chars.py` is what settled
  it. Measure before shrinking anything.
- **The mitigation that worked:** retry the DiT call on the allocation error
  (4 attempts, `gc.collect()` between, re-raise the last). 1-in-5 failures
  dropped to 0/12. A mitigation, not a cure — the margin is still ~0.2 GB.
- **Attention query-chunking is exact but not sufficient.** SDPA materialises a
  `[batch, heads, q, kv]` score matrix; chunking the query axis cut that tensor
  8.6x at seq 4400 (3.46 -> 0.40 GB) with max diff 6e-07. It did **not** fix the
  OOM (1/12 still failed) because the 24-layer stack holds ~2.4 GB of hidden
  activations on its own. Keep it as worst-case insurance; do not call it a fix.
- **Sequential CFG is the fix.** Batched CFG concatenates 3 conditions into one
  batch of 3, tripling every activation. Running them one at a time (batch 1)
  cuts peak activation memory ~3x. Every op is per-batch-element independent,
  so slicing is exact (max diff 2.027e-06 vs batched). Measured: 1-in-5 failures
  -> **0/12**. Trade-off: ~3x compute per denoise step. Threshold: seq > 2100
  frames on a 12 GB card. Loop over `local_cond.size(0)` (the condition axis),
  NOT `x.size(0)` (the latent batch, which is 1).

## Known separate bug, fixed

`chunk_text_english` only split *between* sentences, so a long single sentence
passed through whole and defeated its own `max_chars` (141 chars → one 141-char
chunk). Now hard-splits at word boundaries (`141 → [129, 11]`, word order
preserved). This was **not** the cause of the OOM above, but it was a real bug.


## outputs/

Gradio's temp dir is redirected here by `app.py` (`GRADIO_TEMP_DIR`), so
generated audio and uploaded references stay inside the project instead of
scattering into the system temp folder. Safe to delete; it refills on use.

Note: `gradio_client` in a *separate* process still caches downloads in its own
system-temp dir — that is the client library, not this app.
