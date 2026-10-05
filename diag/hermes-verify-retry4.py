"""Does a 4-attempt retry survive a long run at the ceiling?

Prior evidence: 1-in-5 requests failed with a single attempt; a 1-retry loop
still failed at request 3 of 6. Run 12 varied requests and count failures.
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

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}\n", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

WORDS = ("the quick brown fox jumps over the lazy dog and then keeps running "
         "through the field while the sun slowly sets behind the hills in a "
         "long golden wash of light across everything").split()

TEXTS = [
    "hey i suck dick real good",
    " ".join(WORDS[:8]),
    " ".join(WORDS[:24]),
    "short",
    " ".join(WORDS[:12]),
    " ".join(WORDS[:28]),
    "hey i suck dick real good",
    " ".join(WORDS[:16]),
    "medium length sentence right here",
    " ".join(WORDS[:20]),
    "final short one",
    " ".join(WORDS[:24]),
]

print(f"{'#':>3} {'chars':>6} {'driver':>10}  result")
print("-" * 58)
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
        print(f"{i:>3} {len(TEXT):>6} {gpu():>8.3f} GB  OK {n/24000:5.1f}s", flush=True)
    except RuntimeError as e:
        fails += 1
        gc.collect()
        print(f"{i:>3} {len(TEXT):>6} {gpu():>8.3f} GB  FAIL {str(e)[:36]}", flush=True)

print("-" * 58)
print(f"failures: {fails}/{len(TEXTS)}   peak={gpu():.3f} GB")
print("VERDICT:", "SURVIVED" if fails == 0 else f"{fails} still failed")
