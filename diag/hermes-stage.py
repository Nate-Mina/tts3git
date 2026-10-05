"""Where does the activation memory go during ONE request?

Measures driver VRAM after each pipeline stage, in a single process, for the
user's exact short input. Reads only the driver counter so the measurement
cannot itself grow the DML pool.
"""
import os, sys, subprocess, gc, io, time, threading
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import torch
import wave

PS = ("Get-Counter '\\GPU Adapter Memory(*)\Dedicated Usage' -EA SilentlyContinue | "
      "Select-Object -ExpandProperty CounterSamples | "
      "Where-Object {$_.CookedValue -gt 50MB} | "
      "ForEach-Object { '{0:N3}' -f ($_.CookedValue/1GB) }")


def gpu():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=60)
    v = [float(x) for x in r.stdout.split() if x.strip()]
    return max(v) if v else -1.0


# background sampler to catch the true peak between stages
PEAK = [0.0]
STOP = threading.Event()


def sampler():
    while not STOP.is_set():
        PEAK[0] = max(PEAK[0], gpu())
        time.sleep(1.5)


from tts.infer_cli import MegaTTS3DiTInfer
from tts.frontend_function import g2p, dur_pred, prepare_inputs_for_dit, make_dur_prompt
from tts.dml_device import autocast_ctx

print(f"{'stage':<34} {'driver':>10}")
print("-" * 46)
print(f"{'baseline':<34} {gpu():>8.3f} GB")

infer = MegaTTS3DiTInfer()
print(f"{'model loaded':<34} {gpu():>8.3f} GB   g2p={infer.g2p_device}")

th = threading.Thread(target=sampler, daemon=True); th.start()

with open("example/reference.wav", "rb") as f:
    audio = f.read()

ctx = infer.preprocess(audio)
print(f"{'preprocess (vae encode+dur prompt)':<34} {gpu():>8.3f} GB")

TEXT = "hey i suck dick real good"

ph_pred, tone_pred = g2p(infer, TEXT)
print(f"{'g2p':<34} {gpu():>8.3f} GB")

mel2ph_pred = dur_pred(infer, ctx['ctx_dur_tokens'],
                       ctx['incremental_state_dur_prompt'], ph_pred, tone_pred,
                       0, 0.1, 1.0, is_first=True, is_final=True)
print(f"{'dur_pred':<34} {gpu():>8.3f} GB   mel frames={mel2ph_pred.size(1)}")

inputs = prepare_inputs_for_dit(infer, ctx['mel2ph_ref'], mel2ph_pred,
                                ctx['ph_ref'], ctx['tone_ref'],
                                ph_pred, tone_pred, ctx['vae_latent'])
print(f"{'prepare_inputs_for_dit':<34} {gpu():>8.3f} GB   seq={inputs['dur'].size(1)}")

with autocast_ctx(infer.device):
    x = infer.dit.inference(inputs, timesteps=32, seq_cfg_w=[1.4, 3.0]).float()
print(f"{'dit.inference (32 steps)':<34} {gpu():>8.3f} GB   x={list(x.shape)}")

x[:, :ctx['vae_latent'].size(1)] = ctx['vae_latent']
wav = infer.wavvae.decode(x)[0, 0].to(torch.float32)
print(f"{'wavvae.decode':<34} {gpu():>8.3f} GB   samples={wav.numel()}")

del x, wav, inputs, ctx
gc.collect()
print(f"{'after cleanup':<34} {gpu():>8.3f} GB")

STOP.set(); th.join(timeout=5)
print("-" * 46)
print(f"sampled peak during request: {PEAK[0]:.3f} GB")
