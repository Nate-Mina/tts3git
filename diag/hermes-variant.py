"""Correct measurement: everything under no_grad, like the app's forward().

VARIANT=current  -> decode immediately (today's behaviour)
VARIANT=free     -> del DiT intermediates + gc before decode
VARIANT=cpufb    -> decode on CPU when the GPU decode OOMs
"""
import os, sys, subprocess, gc, io, time, threading
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

VARIANT = os.environ.get("VARIANT", "current")
import torch, torch_directml
from tts.dml_device import autocast_ctx

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
        PEAK[0] = max(PEAK[0], gpu()); time.sleep(1.5)


from tts.infer_cli import MegaTTS3DiTInfer

infer = MegaTTS3DiTInfer()
print(f"variant={VARIANT}  loaded={gpu():.3f} GB  g2p={infer.g2p_device}", flush=True)

with open("example/reference.wav", "rb") as f:
    audio = f.read()

TEXT = "hey i suck dick real good"

th = threading.Thread(target=sampler, daemon=True); th.start()
t0 = time.time()
try:
    with torch.no_grad():
        ctx = infer.preprocess(audio)
        ph_ref = ctx['ph_ref']; tone_ref = ctx['tone_ref']
        mel2ph_ref = ctx['mel2ph_ref']; vae_latent = ctx['vae_latent']

        from tts.frontend_function import g2p, dur_pred, prepare_inputs_for_dit
        ph_pred, tone_pred = g2p(infer, TEXT)
        mel2ph_pred = dur_pred(infer, ctx['ctx_dur_tokens'],
                               ctx['incremental_state_dur_prompt'], ph_pred, tone_pred,
                               0, 0.1, 1.0, is_first=True, is_final=True)
        inputs = prepare_inputs_for_dit(infer, mel2ph_ref, mel2ph_pred, ph_ref,
                                        tone_ref, ph_pred, tone_pred, vae_latent)
        with autocast_ctx(infer.device):
            x = infer.dit.inference(inputs, timesteps=32, seq_cfg_w=[1.4, 3.0]).float()
        after_dit = gpu()
        print(f"  after dit.inference      {after_dit:.3f} GB  x={list(x.shape)}", flush=True)

        if VARIANT == "free":
            del inputs, ctx
            gc.collect()
            print(f"  after del inputs+gc      {gpu():.3f} GB", flush=True)

        x[:, :vae_latent.size(1)] = vae_latent
        try:
            wav_pred = infer.wavvae.decode(x)[0, 0].to(torch.float32)
            where = "gpu"
        except RuntimeError as e:
            if VARIANT == "cpufb" and "not enough GPU video memory" in str(e):
                print(f"  GPU decode OOM -> CPU fallback", flush=True)
                infer.wavvae.to("cpu"); gc.collect()
                wav_pred = infer.wavvae.decode(x.cpu())[0, 0].to(torch.float32)
                infer.wavvae.to(infer.device); gc.collect()
                where = "cpu"
            else:
                raise
        dt = time.time() - t0
        print(f"  decoded on {where}       {gpu():.3f} GB   {wav_pred.numel()/24000:.2f}s audio", flush=True)
    print(f"RESULT variant={VARIANT}: OK in {dt:.1f}s  peak={PEAK[0]:.3f} GB")
except RuntimeError as e:
    dt = time.time() - t0
    print(f"RESULT variant={VARIANT}: FAIL after {dt:.1f}s  peak={PEAK[0]:.3f} GB")
    print(f"  {str(e)[:110]}")
finally:
    STOP.set(); th.join(timeout=5)
