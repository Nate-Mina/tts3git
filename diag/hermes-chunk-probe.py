"""Is chunk_text_english enforcing max_chars for a single long sentence?"""
import os, sys
os.chdir(r"D:\__dev\__TTS3\MegaTTS3-Voice-Cloning")
sys.path.insert(0, os.getcwd())

from tts.utils.text_utils.split_text import chunk_text_english

CASES = [
    "hey i suck dick real good",
    ("and an even longer passage of text so that the diffusion transformer "
     "has to work with a substantially longer sequence which costs more memory"),
    "First sentence here. Second one that is also fairly long and keeps going.",
    ("one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one"),
]

print(f"{'chars':>6} {'chunks':>6}  {'max chunk':>9}  over 130?")
print("-" * 52)
for t in CASES:
    c = chunk_text_english(t, max_chars=130)
    mx = max(len(x.encode()) for x in c) if c else 0
    print(f"{len(t):>6} {len(c):>6}  {mx:>9}  {'YES <-- bug' if mx > 130 else 'no'}")
    for i, ch in enumerate(c):
        print(f"        [{i}] {len(ch.encode()):>4}B  {ch[:70]!r}")
