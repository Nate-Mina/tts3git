"""Why does the app still write to system temp despite GRADIO_TEMP_DIR?

app.py sets it with os.environ.setdefault(...) BEFORE importing gradio. Test
whether that actually redirects gradio's Audio output cache.
"""
import os, sys
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
ROOT = os.path.abspath(os.getcwd())

# replicate app.py's ordering exactly
target = os.path.join(ROOT, "outputs")
os.environ.setdefault("GRADIO_TEMP_DIR", target)
print(f"GRADIO_TEMP_DIR set to: {os.environ['GRADIO_TEMP_DIR']}")

import gradio as gr
a = gr.Audio(label="x")
print(f"gr.Audio().GRADIO_CACHE = {getattr(a, 'GRADIO_CACHE', 'n/a')}")

# what does the class use at call time?
from gradio import processing_utils
print(f"processing_utils.GRADIO_CACHE = {getattr(processing_utils, 'GRADIO_CACHE', 'n/a')}")

# inspect how Component resolves GRADIO_CACHE
import inspect
from gradio.components import base as cbase
src = inspect.getsource(cbase)
for i, line in enumerate(src.splitlines(), 1):
    if "GRADIO_CACHE" in line:
        print(f"  base.py:{i}: {line.strip()[:120]}")

# Now: is the *server* actually honoring it? Compare a real round trip.
print()
print("=== direct postprocess (no HTTP) ===")
import io, numpy as np, wave
sr = 24000
buf = io.BytesIO()
wave.open(buf, "wb").setparams((1, 2, sr, 1000, "NONE", "not compressed"))
# simpler: build via soundfile
import soundfile as sf
pcm = (np.random.randn(sr) * 0.1).astype(np.float32)
b = io.BytesIO()
sf.write(b, pcm, sr, format="WAV")
raw = b.getvalue()

out = a.postprocess(raw)
print(f"postprocess -> {out}")
p = getattr(out, "path", None) or str(out)
inside = os.path.abspath(str(p)).startswith(ROOT)
print(f"inside project: {inside}")
