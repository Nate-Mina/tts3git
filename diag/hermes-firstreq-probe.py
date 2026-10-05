"""Is the first request after load the one that OOMs? Single model instance.

Earlier probes loaded a second MegaTTS3DiTInfer while the first was still
resident, which filled the card and invalidated the result. One instance here.
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


def one(text):
    with torch.no_grad():
        ctx = infer.preprocess(audio)
        out = infer.forward(ctx, text, time_step=32, p_w=1.4, t_w=3.0)
    with wave.open(io.BytesIO(out), "rb") as w:
        n = w.getnframes()
    del ctx, out
    gc.collect()
    return n


print("=== 10 sequential requests, SAME long text, one model instance ===")
print(f"{'#':>3} {'driver':>10}  result")
print("-" * 44)
ok = fails = 0
first_fail_at = None
for i in range(1, 11):
    try:
        n = one(LONG)
        ok += 1
        print(f"{i:>3} {gpu():>8.3f} GB  OK {n/24000:.1f}s", flush=True)
    except RuntimeError as e:
        fails += 1
        if first_fail_at is None:
            first_fail_at = i
        gc.collect()
        print(f"{i:>3} {gpu():>8.3f} GB  FAIL {str(e)[:40]}", flush=True)

print("-" * 44)
print(f"ok={ok}  fail={fails}  first failure at request {first_fail_at}")
print("VERDICT:", "only the first request fails" if first_fail_at == 1 and fails == 1
      else ("intermittent" if fails else "all pass"))
