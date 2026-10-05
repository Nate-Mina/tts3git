"""Confirm generated artifacts land INSIDE the project, not in system temp."""
import os, sys, time, glob, io
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")

import numpy as np, wave
from gradio_client import Client, handle_file

ROOT = os.getcwd()
BEFORE_OUT = set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True))
BEFORE_TMP = set(glob.glob(os.path.join(os.environ.get("TEMP", ""), "gradio", "**", "*"),
                         recursive=True))

client = Client("http://127.0.0.1:7860/", verbose=False)
res = client.predict(inp_audio=handle_file("example/reference.wav"),
                     inp_text="containment check",
                     infer_timestep=8, p_w=1.4, t_w=3.0,
                     api_name="/generate_speech")
print("returned:", type(res).__name__, str(res)[:110])

path = res if isinstance(res, str) else (res.get("path") or res.get("name"))
print("output path:", path)
inside = os.path.abspath(path).startswith(os.path.abspath(ROOT))
print()
print(f"inside project folder? {'YES' if inside else 'NO  <-- still escaping'}")
print(f"  project root: {ROOT}")

if path and os.path.exists(path):
    with wave.open(path, "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        raw = w.readframes(min(n, sr * 5))
    s = np.frombuffer(raw, dtype=np.int16)
    print(f"  audio: {n/sr:.2f}s peak={int(np.abs(s).max()) if s.size else 0}")

AFTER_OUT = set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True))
AFTER_TMP = set(glob.glob(os.path.join(os.environ.get("TEMP", ""), "gradio", "**", "*"),
                         recursive=True))
print()
print(f"new files under outputs/ : {len(AFTER_OUT - BEFORE_OUT)}")
print(f"new files in system temp : {len(AFTER_TMP - BEFORE_TMP)}")
print("VERDICT:", "CONTAINED" if inside and len(AFTER_TMP - BEFORE_TMP) == 0 else "LEAKING")
