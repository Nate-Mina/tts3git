"""Prove the residual temp write is the CLIENT, not the app.

The app sets GRADIO_TEMP_DIR; a gradio_client in a separate process does not
inherit it. Ask the server to produce a file, then check where each side wrote.
"""
import os, sys, glob
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
ROOT = os.path.abspath(os.getcwd())
TEMP = os.environ.get("TEMP", "")
GDIR = os.path.join(TEMP, "gradio")


def snap():
    return (set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True)),
            set(glob.glob(os.path.join(GDIR, "**", "*"), recursive=True)))


o0, t0 = snap()
print(f"before: outputs={len(o0)} temp={len(t0)}")

# CLIENT process does NOT set GRADIO_TEMP_DIR -> observe its own behaviour
print(f"client GRADIO_TEMP_DIR = {os.environ.get('GRADIO_TEMP_DIR')!r}")

from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860/", verbose=False)
res = c.predict(inp_audio=handle_file("example/reference.wav"),
                inp_text="who wrote where", infer_timestep=8, p_w=1.4, t_w=3.0,
                api_name="/generate_speech")
o1, t1 = snap()
print(f"after : outputs={len(o1)} temp={len(t1)}")
print(f"  server wrote to outputs/ : {len(o1 - o0)} new")
print(f"  client wrote to temp/    : {len(t1 - t0)} new")
print(f"  returned path: {str(res)[:100]}")
print(f"  returned path in temp?   : {str(res).startswith(GDIR)}")

print()
print("=== now WITH GRADIO_TEMP_DIR set in the client process ===")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(ROOT, "outputs")
o2, t2 = snap()
res2 = c.predict(inp_audio=handle_file("example/reference.wav"),
                 inp_text="client contained", infer_timestep=8, p_w=1.4, t_w=3.0,
                 api_name="/generate_speech")
o3, t3 = snap()
print(f"  new in outputs/ : {len(o3 - o2)}")
print(f"  new in temp/    : {len(t3 - t2)}")
print(f"  returned path: {str(res2)[:100]}")
print(f"  inside project?: {os.path.abspath(str(res2)).startswith(ROOT)}")
