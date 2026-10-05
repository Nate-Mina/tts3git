"""Does dropping references + gc actually return DirectML memory to the driver?

If the pool ratchets to the ceiling, the only fix is to release between
requests. Test: after a long request, drop the pipeline's transient refs and
see whether the driver counter falls.
"""
import os, sys, subprocess, gc, io, time
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import torch, torch_directml
DEV = torch_directml.device()

PS = ("Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -EA SilentlyContinue | "
      "Select-Object -ExpandProperty CounterSamples | "
      "Where-Object {$_.CookedValue -gt 50MB} | "
      "ForEach-Object { '{0:N3}' -f ($_.CookedValue/1GB) }")


def gpu():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=60)
    v = [float(x) for x in r.stdout.split() if x.strip()]
    return max(v) if v else -1.0


def try_release():
    """Every plausible release lever; report what each one buys."""
    before = gpu()
    out = {"before": before}

    gc.collect()
    out["gc"] = gpu()

    # torch's caching-allocator API, if the DML build exposes it
    for fn in ("empty_cache", "synchronize"):
        f = getattr(torch.cuda, fn, None)
        if f is None:
            continue
        try:
            f(DEV)
        except Exception:
            try:
                f()
            except Exception:
                continue
    out["cuda_api"] = gpu()

    # allocating a fresh tensor can force a pool defrag/reuse
    try:
        t = torch.empty(1, device=DEV); del t
    except Exception:
        pass
    out["nudge"] = gpu()
    return out


from tts.infer_cli import MegaTTS3DiTInfer

infer = MegaTTS3DiTInfer()
print(f"loaded: {gpu():.3f} GB  g2p={infer.g2p_device}")

with open("example/reference.wav", "rb") as f:
    audio = f.read()

LONG = ("and an even longer passage of text so that the diffusion transformer "
        "has to work with a substantially longer sequence which costs more memory")

print(f"\n{'stage':<30} {'driver':>10}")
print("-" * 44)
print(f"{'before long request':<30} {gpu():>8.3f} GB")

with torch.no_grad():
    ctx = infer.preprocess(audio)
    out = infer.forward(ctx, LONG, time_step=32, p_w=1.4, t_w=3.0)
print(f"{'after long request':<30} {gpu():>8.3f} GB")

del ctx, out, audio
res = try_release()
for k in ("gc", "cuda_api", "nudge"):
    print(f"{'after ' + k:<30} {res[k]:>8.3f} GB")

# now a short request: does it fit?
try:
    with torch.no_grad():
        ctx2 = infer.preprocess(open("example/reference.wav", "rb").read())
        out2 = infer.forward(ctx2, "short one", time_step=32, p_w=1.4, t_w=3.0)
    print(f"{'short request after release':<30} {gpu():>8.3f} GB  OK")
except RuntimeError as e:
    print(f"{'short request after release':<30} {gpu():>8.3f} GB  FAIL {str(e)[:40]}")
