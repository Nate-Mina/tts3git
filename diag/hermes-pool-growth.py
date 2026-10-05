"""Does DirectML's pool grow monotonically across generations in ONE process?

Simulates the real app: load once, then serve N requests without restarting.
Records driver-visible GPU usage and the largest allocatable tensor per step.
"""
import os, sys, subprocess, gc, io, time
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import numpy as np
import torch
import torch_directml
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


DEV = torch_directml.device()


def max_alloc_mb():
    """Largest single tensor we can still allocate (binary-ish search)."""
    best = 0
    for mb in (4096, 3072, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 1):
        try:
            t = torch.empty(int(mb * 1e6 / 4), dtype=torch.float32, device=DEV)
            del t
            best = mb
            break
        except Exception:
            continue
    return best


from tts.infer_cli import MegaTTS3DiTInfer

print(f"{'stage':<26} {'driver':>9}  {'max_alloc':>10}")
print("-" * 50)
print(f"{'baseline':<26} {gpu():>8.3f}G  {max_alloc_mb():>8} MB")

infer = MegaTTS3DiTInfer()
print(f"{'model loaded':<26} {gpu():>8.3f}G  {max_alloc_mb():>8} MB")

with open("example/reference.wav", "rb") as f:
    audio = f.read()

for i in range(1, 7):
    try:
        ctx = infer.preprocess(audio)
        out = infer.forward(ctx, f"pool growth test {i}", time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, out
        gc.collect()
        print(f"{'after gen ' + str(i):<26} {gpu():>8.3f}G  {max_alloc_mb():>8} MB"
              f"   ({n/24000:.1f}s audio)")
    except RuntimeError as e:
        gc.collect()
        print(f"{'gen ' + str(i) + ' FAILED':<26} {gpu():>8.3f}G  {max_alloc_mb():>8} MB"
              f"   {str(e)[:60]}")
