"""Does the chunking fix let long text generate?

The exact input that OOM'd twice before (141 chars, single sentence).
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
from tts.utils.text_utils.split_text import chunk_text_english

LONG = ("and an even longer passage of text so that the diffusion transformer "
        "has to work with a substantially longer sequence which costs more memory")
print("chunks for the long input:", [len(c.encode()) for c in chunk_text_english(LONG)])

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}\n", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

TEXTS = [
    (LONG, "long, previously OOM'd"),
    ("hey i suck dick real good", "short, user's case"),
    (LONG, "long again"),
    ("a third long one that runs on and on without any punctuation to break it "
     "up at all which is exactly the shape that used to blow the card up badly",
     "very long, no punctuation"),
    ("short one", "short again"),
]

print(f"{'#':>3} {'chars':>6} {'driver':>10}  result")
print("-" * 64)
fails = 0
for i, (TEXT, label) in enumerate(TEXTS, 1):
    try:
        with torch.no_grad():
            ctx = infer.preprocess(audio)
            out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, out
        gc.collect()
        print(f"{i:>3} {len(TEXT):>6} {gpu():>8.3f} GB  OK {n/24000:5.1f}s  [{label}]", flush=True)
    except RuntimeError as e:
        fails += 1
        gc.collect()
        print(f"{i:>3} {len(TEXT):>6} {gpu():>8.3f} GB  FAIL {str(e)[:38]}  [{label}]", flush=True)

print("-" * 64)
print(f"failures: {fails}/{len(TEXTS)}   peak={gpu():.3f} GB")
print("VERDICT:", "FIXED" if fails == 0 else "STILL FAILING")
