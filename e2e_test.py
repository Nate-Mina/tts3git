"""End-to-end test of the running Gradio app via gradio_client.

Usage: python e2e_test.py [port]
"""
import sys
import time

from gradio_client import Client, handle_file

port = sys.argv[1] if len(sys.argv) > 1 else "7861"
client = Client(f"http://127.0.0.1:{port}", verbose=False)

t0 = time.time()
result = client.predict(
    handle_file("example/reference.wav"),
    "This is an end to end test through the Gradio web interface.",
    32,
    1.4,
    3.0,
    api_name="/generate_speech",
)
print(f"predict() returned after {time.time() - t0:.1f}s -> {result}")

import soundfile as sf
info = sf.info(result)
print(f"OK  {info.duration:.2f}s @ {info.samplerate}Hz")
