"""Does DirectML's pool grow across generations in ONE long-lived process?

Reads only the driver counter (via subprocess) so the measurement itself cannot
grow the DML pool. This simulates the app: load once, serve N requests.
"""
import os, sys, subprocess, gc, io
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import torch
import wave

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

print(f"{'stage':<28} {'driver GPU':>11}   result")
print("-" * 62)
print(f"{'baseline (before load)':<28} {gpu():>9.3f} GB")

infer = MegaTTS3DiTInfer()
print(f"{'model loaded':<28} {gpu():>9.3f} GB   g2p={infer.g2p_device}")

with open("example/reference.wav", "rb") as f:
    audio = f.read()

peak = gpu()
for i in range(1, 9):
    try:
        ctx = infer.preprocess(audio)
        out = infer.forward(ctx, f"pool growth test number {i}", time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, out
        gc.collect()
        g = gpu()
        peak = max(peak, g)
        print(f"{'after gen ' + str(i):<28} {g:>9.3f} GB   {n/24000:.1f}s audio OK")
    except RuntimeError as e:
        gc.collect()
        g = gpu()
        peak = max(peak, g)
        print(f"{'gen ' + str(i):<28} {g:>9.3f} GB   FAIL: {str(e)[:48]}")

print("-" * 62)
print(f"peak driver usage: {peak:.3f} GB")
