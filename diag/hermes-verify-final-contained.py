"""Final containment + function check.

1. nothing of this session's outside the project folder
2. app serves, and a real request writes only under outputs/
"""
import os, sys, glob, subprocess, time, io
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")

ROOT = os.path.abspath(os.getcwd())
TEMP = os.environ.get("TEMP", "")
HOME = os.path.expanduser("~")
results = []


def check(label, cond, detail=""):
    results.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))


print("=== 1. nothing stray outside the project ===")
probes = glob.glob(os.path.join(TEMP, "hermes-*"))
check("no hermes probes in system temp", not probes,
      f"{len(probes)}: {[os.path.basename(p) for p in probes[:5]]}" if probes else "clean")

scratch = glob.glob(os.path.join(HOME, "AppData/Local/hermes/cache/scratch", "*"))
scratch = [p for p in scratch if not p.endswith(".last_prune")]
check("hermes scratch is empty", not scratch,
      f"{len(scratch)}: {[os.path.basename(p) for p in scratch[:5]]}" if scratch else "clean")

gdir = os.path.join(TEMP, "gradio")
gn = len(glob.glob(os.path.join(gdir, "**", "*"), recursive=True))
check("system gradio temp is clear", gn == 0, f"{gn} entries" if gn else "clean")

print("")
print("=== 2. app is serving ===")
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=10) as r:
        body = r.read().decode("utf-8", "replace")
    check(f"HTTP {r.getcode()}", r.getcode() == 200)
    check("is the MegaTTS3 UI", "MegaTTS" in body)
except Exception as e:
    check("app reachable", False, str(e)[:80])

print("")
print("=== 3. a real request stays inside the project ===")
before_out = set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True))
before_tmp = set(glob.glob(os.path.join(gdir, "**", "*"), recursive=True))

from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7860/", verbose=False)
res = c.predict(inp_audio=handle_file("example/reference.wav"),
                inp_text="final containment verification", infer_timestep=8,
                p_w=1.4, t_w=3.0, api_name="/generate_speech")
path = res if isinstance(res, str) else (res.get("path") or res.get("name"))

after_out = set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True))
after_tmp = set(glob.glob(os.path.join(gdir, "**", "*"), recursive=True))

check("output lives under the project", os.path.abspath(path).startswith(ROOT), path)
check("new files in outputs/", len(after_out - before_out) > 0, f"{len(after_out - before_out)} new")
check("no new files in system temp", len(after_tmp - before_tmp) == 0,
      f"{len(after_tmp - before_tmp)} new" if after_tmp - before_tmp else "0 new")

if path and os.path.exists(path):
    import wave, numpy as np
    with wave.open(path, "rb") as w:
        n, sr = w.getnframes(), w.getframerate()
        raw = w.readframes(min(n, sr * 5))
    s = np.frombuffer(raw, dtype=np.int16)
    pk = int(np.abs(s).max()) if s.size else 0
    check("audio is real (not silent)", pk > 1000, f"{n/sr:.2f}s peak={pk}")

print("")
print("=" * 52)
print(f"RESULT: {sum(results)}/{len(results)} checks passed")
print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
sys.exit(0 if all(results) else 1)
