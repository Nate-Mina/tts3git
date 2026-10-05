"""Measure DiT sequence length vs text length, and where it starts to fail.

The reference prompt contributes a fixed number of mel frames to every request,
so even short text has a sizeable sequence. Find the text length at which the
total exceeds what the card can hold, so max_chars can be set from data.
"""
import os, sys, subprocess, gc, io, threading, time
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


PEAK = [0.0]; STOP = threading.Event()
def sampler():
    while not STOP.is_set():
        PEAK[0] = max(PEAK[0], gpu()); time.sleep(1.0)


from tts.infer_cli import MegaTTS3DiTInfer
from tts.frontend_function import g2p, dur_pred, prepare_inputs_for_dit
from tts.utils.text_utils.split_text import chunk_text_english

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}\n", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

# the reference prompt's contribution, measured once
with torch.no_grad():
    ctx = infer.preprocess(audio)
    ref_frames = ctx['mel2ph_ref'].size(1)
print(f"reference prompt contributes {ref_frames} mel frames to every request\n")

WORDS = ("the quick brown fox jumps over the lazy dog and then keeps running "
         "through the field while the sun slowly sets behind the hills in a "
         "long golden wash of light across everything").split()

print(f"{'words':>6} {'chars':>6} {'mel':>6} {'seq':>6} {'driver':>10}  result")
print("-" * 64)
th = threading.Thread(target=sampler, daemon=True); th.start()
try:
    for nw in (4, 8, 12, 16, 20, 24, 28, 32):
        TEXT = " ".join(WORDS[:nw])
        try:
            with torch.no_grad():
                ph, tn = g2p(infer, TEXT)
                mel = dur_pred(infer, ctx['ctx_dur_tokens'],
                               ctx['incremental_state_dur_prompt'], ph, tn,
                               0, 0.1, 1.0, is_first=True, is_final=True)
                inp = prepare_inputs_for_dit(infer, ctx['mel2ph_ref'], mel,
                                             ctx['ph_ref'], ctx['tone_ref'],
                                             ph, tn, ctx['vae_latent'])
                seq = inp['dur'].size(1)
                out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
            with wave.open(io.BytesIO(out), "rb") as w:
                n = w.getnframes()
            del inp, out
            gc.collect()
            print(f"{nw:>6} {len(TEXT):>6} {mel.size(1):>6} {seq:>6} {gpu():>8.3f} GB  OK {n/24000:.1f}s", flush=True)
        except RuntimeError as e:
            gc.collect()
            print(f"{nw:>6} {len(TEXT):>6} {'?':>6} {'?':>6} {gpu():>8.3f} GB  FAIL {str(e)[:30]}", flush=True)
finally:
    STOP.set(); th.join(timeout=5)

print("-" * 64)
print(f"peak driver: {PEAK[0]:.3f} GB")
