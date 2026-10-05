"""Can we lower the DirectML plateau? Compare default vs disable_tiled_resources,
using the app's exact call path (preprocess_audio_robust + cut_wav + generate).

MODE=default | MODE=tiled_off
"""
import os, sys, subprocess, gc, io
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

MODE = os.environ.get("MODE", "default")
import torch, torch_directml

if MODE == "tiled_off":
    try:
        torch_directml.disable_tiled_resources(True)
        print("disable_tiled_resources(True) applied")
    except Exception as e:
        print("disable_tiled_resources failed:", e)

PS = ("Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -EA SilentlyContinue | "
      "Select-Object -ExpandProperty CounterSamples | "
      "Where-Object {$_.CookedValue -gt 50MB} | "
      "ForEach-Object { '{0:N3}' -f ($_.CookedValue/1GB) }")


def gpu():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=60)
    v = [float(x) for x in r.stdout.split() if x.strip()]
    return max(v) if v else -1.0


import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import wave

# --- app.py's preprocess_audio_robust, verbatim in spirit ---
def preprocess_audio_robust(audio_path, target_sr=22050, max_duration=30):
    audio = AudioSegment.from_file(audio_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if len(audio) > max_duration * 1000:
        audio = audio[:max_duration * 1000]
    audio = normalize(audio)
    audio = audio.set_frame_rate(target_sr)
    temp_path = audio_path.replace(os.path.splitext(audio_path)[1], "_processed.wav")
    audio.export(temp_path, format="wav",
                 parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr)])
    return temp_path


def cut_wav(p, max_len=28):
    a = AudioSegment.from_file(p)
    a[:int(max_len * 1000)].export(p, format="wav")


from tts.infer_cli import MegaTTS3DiTInfer

infer = MegaTTS3DiTInfer()
print(f"loaded: GPU={gpu():.3f} GB  g2p={infer.g2p_device}")

TEXTS = ["hey i suck dick real good",
         "hey i suck dick real good",
         "this is a longer sentence to push the sequence length up quite a bit more"]

print(f"\n{'request':<52} {'driver':>10}  result")
print("-" * 78)
peak = gpu()
for i, TEXT in enumerate(TEXTS, 1):
    try:
        pp = preprocess_audio_robust("example/reference.wav")
        cut_wav(pp)
        with open(pp, "rb") as f:
            content = f.read()
        ctx = infer.preprocess(content)
        out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
        with wave.open(io.BytesIO(out), "rb") as w:
            n = w.getnframes()
        del ctx, out
        gc.collect()
        g = gpu(); peak = max(peak, g)
        print(f"{TEXT[:50]:<52} {g:>8.3f} GB  OK {n/24000:.1f}s")
    except RuntimeError as e:
        gc.collect()
        g = gpu(); peak = max(peak, g)
        print(f"{TEXT[:50]:<52} {g:>8.3f} GB  FAIL {str(e)[:40]}")

print("-" * 78)
print(f"RESULT mode={MODE}: peak={peak:.3f} GB")
