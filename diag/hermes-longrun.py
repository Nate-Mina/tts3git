"""Long-lived process, app-identical path, varied text lengths.

Answers: does the DML pool creep upward across requests, or plateau?
Uses generate_speech's exact sequence, under no_grad (as infer_cli does).
"""
import os, sys, subprocess, gc, io, time, threading
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"

import torch, wave
from pydub import AudioSegment
from pydub.effects import normalize

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


def preprocess_audio_robust(audio_path, target_sr=22050, max_duration=30):
    audio = AudioSegment.from_file(audio_path)
    if audio.channels > 1:
        audio = audio.set_channels(1)
    if len(audio) > max_duration * 1000:
        audio = audio[:max_duration * 1000]
    audio = normalize(audio).set_frame_rate(target_sr)
    temp = audio_path.replace(os.path.splitext(audio_path)[1], "_processed.wav")
    audio.export(temp, format="wav",
                 parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr)])
    return temp


def cut_wav(p, max_len=28):
    AudioSegment.from_file(p)[:int(max_len * 1000)].export(p, format="wav")


from tts.infer_cli import MegaTTS3DiTInfer

infer = MegaTTS3DiTInfer()
print(f"loaded={gpu():.3f} GB  g2p={infer.g2p_device}", flush=True)

TEXTS = [
    "hey i suck dick real good",
    "hey i suck dick real good",
    "hey i suck dick real good",
    "hello this is a longer sentence to push the sequence length upward more",
    "and an even longer passage of text so that the diffusion transformer has "
    "to work with a substantially longer sequence which costs more memory",
    "short",
    "medium length sentence here for testing",
    "another short one",
    "final request of the batch to see whether memory has crept up",
]

th = threading.Thread(target=sampler, daemon=True); th.start()
print(f"\n{'#':>3} {'driver':>10}  result")
print("-" * 52)
fails = 0
try:
    for i, TEXT in enumerate(TEXTS, 1):
        try:
            pp = preprocess_audio_robust("example/reference.wav")
            cut_wav(pp)
            with open(pp, "rb") as f:
                content = f.read()
            with torch.no_grad():
                ctx = infer.preprocess(content)
                out = infer.forward(ctx, TEXT, time_step=32, p_w=1.4, t_w=3.0)
            with wave.open(io.BytesIO(out), "rb") as w:
                n = w.getnframes()
            del ctx, out
            gc.collect()
            print(f"{i:>3} {gpu():>8.3f} GB  OK {n/24000:.1f}s  [{TEXT[:34]}]", flush=True)
        except RuntimeError as e:
            fails += 1
            gc.collect()
            print(f"{i:>3} {gpu():>8.3f} GB  FAIL {str(e)[:44]}", flush=True)
finally:
    STOP.set(); th.join(timeout=5)

print("-" * 52)
print(f"peak={PEAK[0]:.3f} GB   failures={fails}/{len(TEXTS)}")
