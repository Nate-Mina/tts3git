"""Reproduce the user's scenario: a long-lived app served many requests.

Starts app.py as a subprocess, sends N requests through the real HTTP endpoint
with varied text lengths, and records driver VRAM + success after each one.
"""
import os, sys, subprocess, time, io, json
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())

import numpy as np, wave, urllib.request

PY = os.path.join(os.getcwd(), "env-dml", "Scripts", "python.exe")
PS = ("Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -EA SilentlyContinue | "
      "Select-Object -ExpandProperty CounterSamples | "
      "Where-Object {$_.CookedValue -gt 50MB} | "
      "ForEach-Object { '{0:N3}' -f ($_.CookedValue/1GB) }")


def gpu():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=60)
    v = [float(x) for x in r.stdout.split() if x.strip()]
    return max(v) if v else -1.0


print(f"VRAM before app start: {gpu():.3f} GB")

env = {**os.environ, "PYTHONPATH": "", "MEGATTS3_DEVICE": "privateuseone:0"}
proc = subprocess.Popen([PY, "app.py"], cwd=os.getcwd(), env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

try:
    up = False
    for _ in range(48):
        time.sleep(5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=5) as r:
                if r.getcode() == 200:
                    up = True
                    break
        except Exception:
            continue
    print(f"app up: {up}   VRAM: {gpu():.3f} GB")
    if not up:
        raise SystemExit("app never came up")

    from gradio_client import Client, handle_file
    client = Client("http://127.0.0.1:7860/", verbose=False)
    REF = "example/reference.wav"

    # varied lengths, like a real user
    TEXTS = [
        "hey i suck dick real good",
        "hey i suck dick real good",
        "hello there this is a somewhat longer sentence to push things a bit",
        "short one",
        "and now a considerably longer passage of text that will produce a much "
        "longer sequence in the diffusion transformer because attention cost "
        "grows with the square of the sequence length in this architecture",
        "back to short",
        "medium length sentence right about here for testing purposes",
        "another one",
    ]

    print(f"\n{'#':>3} {'text':<44} {'driver':>10}  result")
    print("-" * 78)
    peak = gpu()
    fails = 0
    for i, TEXT in enumerate(TEXTS, 1):
        try:
            res = client.predict(inp_audio=handle_file(REF), inp_text=TEXT,
                                 infer_timestep=32, p_w=1.4, t_w=3.0,
                                 api_name="/generate_speech")
            g = gpu(); peak = max(peak, g)
            if not res:
                fails += 1
                print(f"{i:>3} {TEXT[:42]:<44} {g:>8.3f} GB  FAIL(None)")
                continue
            path = res if isinstance(res, str) else (res.get("path") or res.get("name"))
            with wave.open(path, "rb") as w:
                n, sr = w.getnframes(), w.getframerate()
                raw = w.readframes(min(n, sr * 5))
            s = np.frombuffer(raw, dtype=np.int16)
            pk = int(np.abs(s).max()) if s.size else 0
            print(f"{i:>3} {TEXT[:42]:<44} {g:>8.3f} GB  OK {n/sr:.1f}s peak={pk}")
        except Exception as e:
            fails += 1
            g = gpu(); peak = max(peak, g)
            print(f"{i:>3} {TEXT[:42]:<44} {g:>8.3f} GB  FAIL {str(e)[:38]}")

    print("-" * 78)
    print(f"peak driver: {peak:.3f} GB   failures: {fails}/{len(TEXTS)}")
finally:
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=25)
    except Exception:
        proc.kill()
        out = ""
    # surface any OOM lines from the app log
    oom = [l for l in (out or "").splitlines() if "Could not allocate" in l]
    print(f"\nOOM lines in app log: {len(oom)}")
    for l in oom[:4]:
        print("  ", l.strip()[:120])
