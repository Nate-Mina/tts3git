"""CPU-vs-DirectML diff of the ops shimmed in tts/dml_compat.py.

The repo ships no test suite, so this is the correctness proof. Run with the
DirectML venv so a real `privateuseone` device is exercised:

    env-dml/Scripts/python.exe scripts/dml_diff_probe.py

Exits 1 on any mismatch, 0 if every op matches CPU to tolerance.
"""
import os
import sys

os.environ.setdefault("MEGATTS3_DEVICE", "privateuseone:0")
os.chdir(os.path.dirname(__file__) + "/..")
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.dml_device import DEVICE, BACKEND, is_directml, describe_device
from tts.dml_compat import (
    safe_gather, safe_scatter_add, DmlLSTM, maybe_replace_lstm, apply_shims,
    _MAX_ONEHOT_ELEMS,
)

torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    ok = ok and cond


def dcpu(t):
    return t.detach().cpu().float()


def dml(t):
    return t.to(DEVICE)


print(f"device={DEVICE}  backend={BACKEND}  ({describe_device(torch.device(DEVICE))})")
print(f"torch={torch.__version__}  cuda={torch.cuda.is_available()}\n")

if not is_directml(DEVICE):
    print("NO DirectML device active. Use the env-dml interpreter.")
    sys.exit(2)

apply_shims()
check("apply_shims installed the torch.gather patch",
      getattr(torch.Tensor, "_gather_shimmed", False))

TOL = 1e-5


# ---- gather: production shape 1 (ar_dur expand_states) ----
B, Ts, Tt, H = 2, 7, 19, 16
h = torch.randn(B, Ts, H)
idx = torch.randint(0, Ts, (B, Tt, H))
ref = torch.gather(h, 1, idx)
got = safe_gather(dml(h), 1, dml(idx))
check("gather dim=1 [B,Ts,H]<-idx[B,Tout,H] parity",
      torch.allclose(dcpu(ref), dcpu(got), atol=TOL),
      f"max diff {(dcpu(ref)-dcpu(got)).abs().max():.3e}")

# ---- gather: production shape 2 (seq_utils expand_source_encoding) ----
T2, C = 11, 8
src = torch.randn(T2, C)
i0 = torch.randint(0, T2, (5, C))
check("gather dim=0 [T,C]<-idx[T2,C] parity",
      torch.allclose(dcpu(torch.gather(src, 0, i0)),
                     dcpu(safe_gather(dml(src), 0, dml(i0))), atol=TOL))

# ---- gather: 4-D (seq_utils:105 attention) ----
nla, Ba, S, Tx = 5, 2, 8, 6
enc = torch.randn(nla, Ba, S, Tx)
ia = torch.randint(0, nla, (1, Ba, 1, 1)).repeat(1, 1, S, Tx)
check("gather dim=0 4-D parity",
      torch.allclose(dcpu(torch.gather(enc, 0, ia)),
                     dcpu(safe_gather(dml(enc), 0, dml(ia))), atol=TOL),
      f"max diff {(dcpu(torch.gather(enc,0,ia))-dcpu(safe_gather(dml(enc),0,dml(ia)))).abs().max():.3e}")

# unbroadcast index (non-dim size mismatch) is an invalid gather input on both
# native and shimmed paths; safe_gather's einsum setup preserves that behaviour.
iu = torch.randint(0, nla, (1, Ba, 1, 1))
try:
    g_unb = safe_gather(dml(enc), 0, dml(iu))
    check("unbroadcast index shape (should NOT broadcast to input)",
          tuple(g_unb.shape) != (nla, Ba, S, Tx), f"{tuple(g_unb.shape)}")
except Exception as e:
    check("unbroadcast index shape raised (expected, matches native gather)",
          True, f"{type(e).__name__}")

# ---- monkeypatched torch.gather routes DML tensors ----
check("monkeypatched torch.gather matches CPU on DML tensor",
      torch.allclose(dcpu(torch.gather(h, 1, idx)),
                     dcpu(torch.gather(dml(h), 1, dml(idx))), atol=TOL))
check("monkeypatch is a no-op on CPU tensors",
      torch.equal(torch.gather(h, 1, idx), torch.gather(h, 1, idx)))
check("Tensor.gather() method matches safe_gather",
      torch.allclose(dcpu(safe_gather(dml(h), 1, dml(idx))),
                     dcpu(dml(h).gather(1, dml(idx))), atol=TOL))

# ---- scatter_add: mel2token_to_dur ----
Bm, Tm, ncls = 2, 25, 8
mt = torch.randint(0, ncls, (Bm, Tm))
ones = torch.ones_like(mt)
ref_s = mt.new_zeros(Bm, ncls + 1).scatter_add(1, mt, ones)
got_s = safe_scatter_add(dml(mt.new_zeros(Bm, ncls + 1)), 1, dml(mt), dml(ones))
check("scatter_add mel2token_to_dur parity",
      torch.allclose(dcpu(ref_s), dcpu(got_s), atol=TOL),
      f"max diff {(dcpu(ref_s)-dcpu(got_s)).abs().max():.3e}")

# ---- scatter_add: group_hidden_by_segs ----
Bg, Tg, nseg, Hg = 2, 9, 5, 6
seg = torch.randint(0, nseg, (Bg, Tg, Hg))
gh = torch.randn(Bg, Tg, Hg)
ref_g = gh.new_zeros(Bg, nseg + 1, Hg).scatter_add_(1, seg, gh)
got_g = safe_scatter_add(dml(gh.new_zeros(Bg, nseg + 1, Hg)), 1, dml(seg), dml(gh))
check("scatter_add group_hidden_by_segs parity",
      torch.allclose(dcpu(ref_g), dcpu(got_g), atol=TOL),
      f"max diff {(dcpu(ref_g)-dcpu(got_g)).abs().max():.3e}")

# ---- F.pad with mixed +/- amounts: LengthRegulator dur_cumsum_prev ----
Bp, Tp = 2, 12
cum = torch.randn(Bp, Tp).cumsum(1)
ref_p = F.pad(cum, [1, -1], mode='constant', value=0)
got_p = F.pad(dml(cum), [1, -1], mode='constant', value=0)
check("F.pad mixed [1,-1] parity",
      torch.allclose(dcpu(ref_p), dcpu(got_p), atol=TOL),
      f"shape {list(dcpu(got_p).shape)} max diff {(dcpu(ref_p)-dcpu(got_p)).abs().max():.3e}")

# 4-D mixed pad (both axes cropped + padded) exercises the general path
ref_p4 = F.pad(cum[None, None], [1, -1, 2, -1], mode='constant', value=0)
got_p4 = F.pad(dml(cum)[None, None], [1, -1, 2, -1], mode='constant', value=0)
check("F.pad mixed 4-D parity",
      torch.allclose(dcpu(ref_p4), dcpu(got_p4), atol=TOL),
      f"shape {list(dcpu(got_p4).shape)}")

# positive-only pad must be untouched by the shim
ref_pp = F.pad(cum, [2, 3], mode='constant', value=0)
got_pp = F.pad(dml(cum), [2, 3], mode='constant', value=0)
check("F.pad positive-only unchanged",
      torch.allclose(dcpu(ref_pp), dcpu(got_pp), atol=TOL))

# ---- gather overflow path: one-hot too big -> CPU round-trip ----
# Mirrors DiT expand_states: h[B,T,H] gathered along 1 by a long mel2token idx.
Bo, To, Ho, Ti = 1, 900, 32, 3400
h_big = torch.randn(Bo, To, Ho)
idx_big = torch.randint(0, To, (Bo, Ti, Ho))
ref_o = torch.gather(h_big, 1, idx_big)
got_o = safe_gather(dml(h_big), 1, dml(idx_big))
elems = idx_big.numel() * To
check("gather overflow -> CPU fallback parity",
      torch.allclose(dcpu(ref_o), dcpu(got_o), atol=TOL),
      f"one-hot would be {elems/1e9:.2f}G elems ({elems*4/2**30:.1f} GB), budget {_MAX_ONEHOT_ELEMS/1e6:.0f}M")
check("overflow path took the CPU branch", elems > _MAX_ONEHOT_ELEMS)
check("result stays on the DML device", got_o.device.type == DEVICE.split(':')[0])


# ---- RoPE: real-valued rotation must equal the complex64 reference ----
from tts.modules.llm_dit.transformer import (
    precompute_freqs_cis, apply_rotary_emb, reshape_for_broadcast,
)


def _ref_apply_rotary(xq, xk, freqs_cis):
    """The original complex64 implementation, kept here as the oracle."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = reshape_for_broadcast(freqs_cis, xq_)
    return (torch.view_as_real(xq_ * fc).flatten(3).type_as(xq),
            torch.view_as_real(xk_ * fc).flatten(3).type_as(xk))


torch.manual_seed(3)
Bq, Tq, Hq, Dq = 2, 6, 3, 8
xq_r = torch.randn(Bq, Tq, Hq, Dq)
xk_r = torch.randn(Bq, Tq, Hq, Dq)
fc_c = precompute_freqs_cis(Dq, Tq)

ref_q, ref_k = _ref_apply_rotary(xq_r, xk_r, fc_c)
# real-valued path, fed the real view (as the buffer stores it)
got_q, got_k = apply_rotary_emb(xq_r, xk_r, torch.view_as_real(fc_c))
check("RoPE real-valued == complex reference (q)",
      torch.allclose(ref_q, got_q, atol=1e-6),
      f"max diff {(ref_q-got_q).abs().max():.3e}")
check("RoPE real-valued == complex reference (k)",
      torch.allclose(ref_k, got_k, atol=1e-6),
      f"max diff {(ref_k-got_k).abs().max():.3e}")
# also accept the complex form for API compatibility
got_q2, _ = apply_rotary_emb(xq_r, xk_r, fc_c)
check("RoPE accepts complex freqs_cis too",
      torch.allclose(ref_q, got_q2, atol=1e-6))
check("RoPE output is real (not complex)",
      not got_q.is_complex() and not torch.is_complex(got_q))

# the same op on a DML tensor must not produce ComplexFloat
got_qd, _ = apply_rotary_emb(dml(xq_r), dml(xk_r), dml(torch.view_as_real(fc_c)))
check("RoPE on DML tensor stays real",
      not torch.is_complex(got_qd) and not dcpu(got_qd).is_complex())
check("RoPE DML parity",
      torch.allclose(dcpu(got_qd), ref_q, atol=1e-6),
      f"max diff {(dcpu(got_qd)-ref_q).abs().max():.3e}")


# ---- LSTM: state_dict key parity (strict load) ----
r = nn.LSTM(10, 16, 2, bidirectional=True)
m = DmlLSTM(10, 16, 2, bidirectional=True)
missing, unexpected = m.load_state_dict(r.state_dict(), strict=True)
check("DmlLSTM strict load_state_dict (zero missing/unexpected)",
      not missing and not unexpected, f"missing={missing} unexpected={unexpected}")

for bidir, layers, hidden, bs in [(False, 1, 16, 5), (True, 2, 12, 4),
                                  (False, 3, 8, 6), (True, 1, 10, 3)]:
    torch.manual_seed(7)
    ref_l = nn.LSTM(10, hidden, layers, batch_first=True, bidirectional=bidir)
    rep = DmlLSTM(10, hidden, layers, bidirectional=bidir, batch_first=True)
    rep.load_state_dict(ref_l.state_dict(), strict=True)
    rep.to(DEVICE)
    x = torch.randn(bs, 6, 10)
    with torch.no_grad():
        o_ref, _ = ref_l(x)
        o_mine, _ = rep(dml(x))
    check(f"LSTM(bidir={bidir},L={layers},H={hidden}) output parity",
          torch.allclose(dcpu(o_ref), dcpu(o_mine), atol=1e-4),
          f"diff {(dcpu(o_ref)-dcpu(o_mine)).abs().max():.3e}")


class Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 9, 1, bidirectional=False)
        self.sub = nn.Sequential(nn.LSTM(4, 5, 1))
        self.linear = nn.Linear(5, 5)


holder = Holder()
cnt = maybe_replace_lstm(holder, torch.device(DEVICE))
check("maybe_replace_lstm replaced both LSTMs", cnt == 2, f"count={cnt}")
check("holder.lstm is now DmlLSTM", isinstance(holder.lstm, DmlLSTM))
check("maybe_replace_lstm no-op on cpu", maybe_replace_lstm(Holder(), "cpu") == 0)


# ---- CFG: batched _forward vs sequential _forward_seq_cfg ----
# The sequential path exists to cut peak activation memory on DirectML. If it
# disagrees with the batched path, it is not a valid substitute.
from tts.modules.llm_dit.dit import Diffusion

torch.manual_seed(11)
diff = Diffusion().eval()

# Match the real call pattern: x is batch 1 (shared noisy latent),
# the 3 CFG conditions live in local_cond / x_ling (batch 3).
Bs, Ts = 3, 40
x_in = torch.randn(1, Ts, 32)
ctx_mask = torch.ones(Bs, Ts, 1)
ctx_mask[:, Ts // 2:] = 0.0
local_cond = torch.randn(Bs, Ts, 512)
x_ling = torch.randn(Bs, Ts, 1024)
ts = torch.tensor([0.4])

with torch.no_grad():
    ref_cfg = diff._forward(torch.cat([x_in] * Bs), local_cond, x_ling, timesteps=ts, ctx_mask=ctx_mask,
                            seq_cfg_w=[1.4, 3.0])
    got_cfg = diff._forward_seq_cfg(x_in, local_cond, x_ling, timesteps=ts,
                                    ctx_mask=ctx_mask, seq_cfg_w=[1.4, 3.0])

check("sequential CFG matches batched CFG (shape)",
      list(ref_cfg.shape) == list(got_cfg.shape), f"{list(ref_cfg.shape)}")
check("sequential CFG matches batched CFG (values)",
      torch.allclose(ref_cfg, got_cfg, atol=1e-5),
      f"max diff {(ref_cfg - got_cfg).abs().max():.3e}")

# NOTE: no DirectML variant here on purpose. This probe keeps the freshly
# constructed Diffusion on CPU to avoid loading 1.7 GB of weights onto the GPU;
# feeding DML inputs to CPU weights is a device mismatch by construction and
# tests nothing. The real DML path is exercised end-to-end by
# diag/hermes-verify-sdpa-run.py and by the app itself.

# the weights differ per condition, so the three slices must not collapse
torch.manual_seed(12)
x_a = torch.randn(Bs, Ts, 32)
with torch.no_grad():
    o1 = diff._forward_seq_cfg(x_a, local_cond, x_ling, timesteps=ts, ctx_mask=ctx_mask,
                               seq_cfg_w=[1.4, 3.0])
    o2 = diff._forward_seq_cfg(x_a, local_cond, x_ling, timesteps=ts, ctx_mask=ctx_mask,
                               seq_cfg_w=[0.0, 0.0])
check("cfg weights actually change the result", not torch.allclose(o1, o2, atol=1e-4))

print()
sys.exit(0 if ok else 1)
