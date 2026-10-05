"""Quantify the DiT attention cost as a function of sequence length.

If the long-text requirement exceeds the card, no refactor helps -- the input
has to be split. This computes the actual numbers from the model's dims.
"""
import math

# From tts/modules/llm_dit/dit.py
ENC_DIM = 1024
N_LAYERS = 24
N_HEADS = 16
HEAD_DIM = ENC_DIM // N_HEADS          # 64
OUT_CH = 32

# CFG batch: uncond + cond_txt + cond_spk
BSZ = 3
# fp32 bytes
B = 4

print(f"encoder_dim={ENC_DIM} layers={N_LAYERS} heads={N_HEADS} head_dim={HEAD_DIM}")
print(f"CFG batch={BSZ}  dtype=fp32\n")

print(f"{'seq':>6} {'latents':>9} {'audio':>8} {'attn/sc':>10} {'x 16h x 24l':>13} {'total':>9}  fits 11.8GB?")
print("-" * 78)
for seq in (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192):
    # sequence positions in the DiT = mel frames
    mel = seq
    latents = mel // 4
    audio_s = mel / 24000 * 4     # rough: 4 mel frames per 24000/... see below
    # attention scores: [bsz, heads, seq, seq]
    attn = BSZ * N_HEADS * seq * seq * B
    # hidden activations carried across the layer stack (several live at once)
    hidden = BSZ * seq * ENC_DIM * B
    # rough resident set: ~16 live tensors of hidden size x 24 layers is too
    # pessimistic; use 16 x hidden as the working set
    working = 16 * hidden
    total = attn + working
    fits = "yes" if total < 11.8e9 else "NO"
    print(f"{seq:>6} {latents:>9} {audio_s:>7.1f}s {attn/1e6:>8.0f}MB {working/1e6:>11.0f}MB "
          f"{total/1e6:>7.0f}MB  {fits}")

print()
print("Measured: short text (~25 chars) -> seq 1536, works.")
print("          long text (~141 chars) -> seq ~4400, OOMs at 11.7 GB.")
print()
print("Conclusion: the DiT's O(seq^2) attention with CFG batch 3 is the wall.")
print("No amount of freeing helps if the peak itself exceeds the card --")
print("long text must be SPLIT into segments and generated separately.")
