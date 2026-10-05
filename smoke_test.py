"""Smoke test for MegaTTS3 on local hardware (no Gradio, no HF Spaces).

Clones a voice from a reference WAV and writes the result to out.wav.
Usage:
    python smoke_test.py example/reference.wav "Text to speak." [out.wav]
"""
import sys
import time

from tts.infer_cli import MegaTTS3DiTInfer, cut_wav

CKPT_DIR = "checkpoints"


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "example/reference.wav"
    text = sys.argv[2] if len(sys.argv) > 2 else "This is a test of the voice cloning system."
    out_path = sys.argv[3] if len(sys.argv) > 3 else "out.wav"

    t0 = time.time()
    print(f"Loading model from {CKPT_DIR} ...")
    infer_pipe = MegaTTS3DiTInfer(ckpt_root=CKPT_DIR)
    print(f"Model loaded in {time.time() - t0:.1f}s on {infer_pipe.device}")

    cut_wav(ref_path, max_len=28)

    with open(ref_path, "rb") as f:
        audio = f.read()

    t1 = time.time()
    wav_bytes = infer_pipe.forward(infer_pipe.preprocess(audio), text, time_step=32, p_w=1.4, t_w=3.0)
    elapsed = time.time() - t1

    with open(out_path, "wb") as f:
        f.write(wav_bytes)

    import soundfile as sf
    info = sf.info(out_path)
    print(f"OK  wrote {out_path}  ({info.duration:.2f}s audio, generated in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
