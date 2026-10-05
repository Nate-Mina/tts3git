"""Hypothesis: only the FIRST request after load can OOM; a retry succeeds.

If so, the fix is a retry-on-OOM (or a warm-up pass at load), not a smaller
chunk size.
"""
import os, sys, subprocess, gc, io
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

LONG = ("and an even longer passage of text so that the diffusion transformer "
        "has to work with a substantially longer sequence which costs more memory")

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}\n", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()


def one_request(text):
    with torch.no_grad():
        ctx = infer.preprocess(audio)
        out = infer.forward(ctx, text, time_step=32, p_w=1.4, t_w=3.0)
    with wave.open(io.BytesIO(out), "rb") as w:
        n = w.getnframes()
    return n


print("=== first request, with one immediate retry on OOM ===")
for attempt in (1, 2, 3):
    try:
        n = one_request(LONG)
        print(f"  attempt {attempt}: OK ({n/24000:.1f}s)  driver={gpu():.3f} GB")
        break
    except RuntimeError as e:
        gc.collect()
        print(f"  attempt {attempt}: FAIL {str(e)[:52]}  driver={gpu():.3f} GB")

print()
print("=== does a warm-up at load time prevent it? ===")
# fresh instance, run a throwaway short request, then the long one
infer2 = MegaTTS3DiTInfer()
print(f"  reloaded: {gpu():.3f} GB")
try:
    n = one_request("warm up")
    print(f"  warm-up request: OK ({n/24000:.1f}s)  driver={gpu():.3f} GB")
except RuntimeError as e:
    print(f"  warm-up request: FAIL {str(e)[:50]}")
try:
    n = one_request(LONG)
    print(f"  long after warm-up: OK ({n/24000:.1f}s)  driver={gpu():.3f} GB")
    print()
    print("VERDICT: WARM-UP PREVENTS IT" )
except RuntimeError as e:
    print(f"  long after warm-up: FAIL {str(e)[:50]}")
    print()
    print("VERDICT: warm-up does NOT prevent it")
