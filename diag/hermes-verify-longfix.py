"""Does freeing the DiT intermediates before decode fix the long-request OOM?

Reproduces the failing case exactly: load once, run a LONG request that
previously OOM'd in F.scaled_dot_product_attention / the decoder conv stack.
"""
import os, sys, subprocess, gc, io, time
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import torch, wave

PS = ("Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -EA SilentlyContinue | "
      "Select-Object -ExpandProperty CounterSamples | "
      "Where-Object {$_.CookedValue -gt 50MB} | "
      "ForEach-Object { '{0:N3}' -f ($_.CookedValue/1GB) }")


def gpu():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=60)
    v = [float(x) for x in r.stdout.split() if x.strip()]
    return max(v) if v else -1.0


from tts.infer_cli import MegaTTS3DiTInfer

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

LONG = ("and an even longer passage of text so that the diffusion transformer "
        "has to work with a substantially longer sequence which costs more memory")

TEXTS = [
    LONG,                                    # the case that just OOM'd
    "hey i suck dick real good",             # the user's short case
    LONG,                                    # long again, after a short one
    "another short request to confirm reuse",
]

print(f"\n{'#':>3} {'len':>4} {'driver':>10}  result")
print("-" * 60)
fails = 0
for i, TEXT in enumerate(TEXTS, 1):
    try:
        with torch.no_grad():
            ctx = infer.preprocess(audio)
            out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, out
        gc.collect()
        print(f"{i:>3} {len(TEXT):>4} {gpu():>8.3f} GB  OK {n/24000:.1f}s", flush=True)
    except RuntimeError as e:
        fails += 1
        gc.collect()
        print(f"{i:>3} {len(TEXT):>4} {gpu():>8.3f} GB  FAIL {str(e)[:44]}", flush=True)

print("-" * 60)
print(f"failures: {fails}/{len(TEXTS)}   peak={gpu():.3f} GB")
print("VERDICT:", "FIXED" if fails == 0 else "STILL FAILING")
