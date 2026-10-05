"""Find the text-length threshold where the DiT OOMs.

Measures resulting sequence length and success for increasing text lengths,
so max_chars can be calibrated to the hardware.
"""
import os, sys, subprocess, gc, io
os.chdir(r"D:\__dev__\__TTS3\MegaTTS3-Voice-Cloning") if False else None
import os as _o
_o.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, _o.getcwd())
_o.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

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
from tts.frontend_function import g2p, dur_pred, prepare_inputs_for_dit

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}\n", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

WORDS = ("the quick brown fox jumps over the lazy dog and then keeps running "
         "through the field while the sun sets slowly behind the hills").split()

def text_of(nwords):
    return " ".join(WORDS[:nwords])

print(f"{'words':>6} {'chars':>6} {'mel':>6} {'seq':>6} {'driver':>10}  result")
print("-" * 62)
for nw in (4, 8, 12, 16, 20, 24, 28):
    TEXT = text_of(nw)
    try:
        with torch.no_grad():
            ctx = infer.preprocess(audio)
            ph_pred, tone_pred = g2p(infer, TEXT)
            mel2ph_pred = dur_pred(infer, ctx['ctx_dur_tokens'],
                                   ctx['incremental_state_dur_prompt'],
                                   ph_pred, tone_pred, 0, 0.1, 1.0,
                                   is_first=True, is_final=True)
            inputs = prepare_inputs_for_dit(infer, ctx['mel2ph_ref'], mel2ph_pred,
                                            ctx['ph_ref'], ctx['tone_ref'],
                                            ph_pred, tone_pred, ctx['vae_latent'])
            seq = inputs['dur'].size(1)
            out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, inputs, out
        gc.collect()
        print(f"{nw:>6} {len(TEXT):>6} {mel2ph_pred.size(1):>6} {seq:>6} {gpu():>8.3f} GB  OK {n/24000:.1f}s", flush=True)
    except RuntimeError as e:
        gc.collect()
        print(f"{nw:>6} {len(TEXT):>6} {'?':>6} {'?':>6} {gpu():>8.3f} GB  FAIL {str(e)[:30]}", flush=True)
