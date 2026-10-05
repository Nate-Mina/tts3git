"""Is the temp leak the SERVER or my CLIENT?

The server (app.py) has GRADIO_TEMP_DIR set. A gradio_client in a separate
process without that env var downloads its own copy to system temp -- so the
leak may be entirely client-side, i.e. a property of the test harness, not the
app. Test both.
"""
import os, sys, glob, io
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")

ROOT = os.getcwd()
TEMP = os.environ.get("TEMP", "")


def snap():
    return {
        "outputs": set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True)),
        "temp": set(glob.glob(os.path.join(TEMP, "gradio", "**", "*"), recursive=True)),
    }


print("=== A. SERVER side: call generate_speech() in-process (no HTTP) ===")
before = snap()
os.environ["MEGATTS3_DEVICE"] = "privateuseone:0"
import importlib
sys.path.insert(0, ROOT)
import app as appmod
with open("example/reference.wav", "rb") as f:
    raw = f.read()
out = appmod.generate_speech("example/reference.wav", "in process check", 8, 1.4, 3.0)
print("  generate_speech returned:", type(out).__name__, (len(out) if out else 0), "bytes")
after = snap()
print(f"  new under outputs/ : {len(after['outputs'] - before['outputs'])}")
print(f"  new in system temp : {len(after['temp'] - before['temp'])}")
print(f"  -> server-side leak: {'YES' if after['temp'] - before['temp'] else 'NO'}")

print()
print("=== B. CLIENT side: gradio_client without GRADIO_TEMP_DIR ===")
before2 = snap()
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860/", verbose=False)
res = c.predict(inp_audio=handle_file("example/reference.wav"),
                inp_text="client check", infer_timestep=8, p_w=1.4, t_w=3.0,
                api_name="/generate_speech")
after2 = snap()
print(f"  client got: {str(res)[:80]}")
print(f"  new under outputs/ : {len(after2['outputs'] - before2['outputs'])}")
print(f"  new in system temp : {len(after2['temp'] - before2['temp'])}")

print()
print("=== C. CLIENT with GRADIO_TEMP_DIR set to the project ===")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(ROOT, "outputs")
before3 = snap()
res3 = c.predict(inp_audio=handle_file("example/reference.wav"),
                 inp_text="client contained", infer_timestep=8, p_w=1.4, t_w=3.0,
                 api_name="/generate_speech")
after3 = snap()
print(f"  client got: {str(res3)[:80]}")
print(f"  new under outputs/ : {len(after3['outputs'] - before3['outputs'])}")
print(f"  new in system temp : {len(after3['temp'] - before3['temp'])}")
inside = os.path.abspath(str(res3)).startswith(os.path.abspath(ROOT))
print(f"  result inside project: {inside}")
