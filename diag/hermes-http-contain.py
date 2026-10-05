"""Confirm the APP is contained, using only the HTTP API (no gradio_client).

gradio_client always caches downloads in its own temp dir, so it is the wrong
instrument for measuring where the server writes. Use the raw queue API and
then read the file the SERVER reports.
"""
import os, sys, glob, json, time, urllib.request, uuid
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
ROOT = os.path.abspath(os.getcwd())
TEMP = os.environ.get("TEMP", "")
GDIR = os.path.join(TEMP, "gradio")
BASE = "http://127.0.0.1:7860"


def snap():
    return (set(glob.glob(os.path.join(ROOT, "outputs", "**", "*"), recursive=True)),
            set(glob.glob(os.path.join(GDIR, "**", "*"), recursive=True)))


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# upload the reference wav through the API so the server owns the file
with open("example/reference.wav", "rb") as f:
    wav_bytes = f.read()

boundary = uuid.uuid4().hex
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="files"; filename="reference.wav"\r\n'
    f"Content-Type: audio/wav\r\n\r\n"
).encode() + wav_bytes + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(BASE + "/gradio_api/upload", data=body,
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
with urllib.request.urlopen(req, timeout=60) as r:
    uploaded = json.loads(r.read().decode())
print("uploaded to server:", uploaded)
server_path = uploaded[0] if isinstance(uploaded, list) else uploaded

o0, t0 = snap()

sid = uuid.uuid4().hex
print("joining queue...")
try:
    post("/gradio_api/queue/join", {
        "data": [{"path": server_path, "meta": {"_type": "gradio.FileData"}},
                 "http api containment check", 8, 1.4, 3.0],
        "event_data": None, "fn_index": 0, "trigger_id": 0,
        "session_hash": sid,
    })
except Exception as e:
    print("join failed:", type(e).__name__, str(e)[:120])
    raise SystemExit(1)

# read the SSE result stream
result_path = None
try:
    with urllib.request.urlopen(f"{BASE}/gradio_api/queue/data?session_hash={sid}", timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:].strip())
            if ev.get("msg") == "process_completed":
                data = ev.get("output", {}).get("data") or []
                if data:
                    d = data[0]
                    result_path = d.get("path") if isinstance(d, dict) else d
                break
            if ev.get("msg") == "process_generating" and ev.get("output", {}).get("data"):
                d = ev["output"]["data"][0]
                result_path = d.get("path") if isinstance(d, dict) else d
except Exception as e:
    print("stream error:", type(e).__name__, str(e)[:120])

o1, t1 = snap()
print()
print("server-reported output:", result_path)
print(f"  new in outputs/ : {len(o1 - o0)}")
print(f"  new in temp/    : {len(t1 - t0)}")
if result_path:
    inside = os.path.abspath(str(result_path)).startswith(ROOT)
    print(f"  path inside project: {inside}")
    print()
    print("VERDICT:", "APP IS CONTAINED" if inside and len(t1 - t0) == 0 else "APP STILL WRITES OUTSIDE")
