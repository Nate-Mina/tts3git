"""DirectML op compatibility shims for MegaTTS3.

DirectML's failure mode is SILENT: a bad op returns a correctly shaped tensor
of plausible garbage rather than raising. Every value here is diffed against
the CPU reference in `scripts/dml_diff_probe.py` -- that diff is the
correctness proof (this repo ships neither a test framework nor a suite).

Shims:
  * gather / scatter_add -- exact one-hot + einsum replacements. DirectML
    ignores its index for `gather` and rejects partially-modified index dims for
    scatter_add_.
  * F.pad               -- DirectML rejects a mix of positive and negative pad
                           amounts; crop the negative sides, then pad the rest.
  * nn.LSTM               -- NotImplementedError on the fused GRU cell.
                           Replaced by a matmul cell loop with EXACT parameter
                           names (`_l0_reverse` etc.) so strict load_state_dict
                           round-trips.

All four are installed as monkeypatches by apply_shims(), so call sites need no
edits. Each is guarded by a `_*_shimmed` flag and falls through to native torch
off DirectML.

torch.stft is NOT shimmed. It lives only in `hifigan_modules.Audio2Mel`, which
has no construction site anywhere in the repo (verified) -- the hard-process-
abort path is unreachable.

NOT a shim: `llm_dit/transformer.apply_rotary_emb` was rewritten in-place to do
the RoPE rotation in real arithmetic instead of complex64. DirectML aborts the
whole process on ComplexFloat, which no monkeypatch can intercept, so the
complex multiply is unrolled into (ac-bd, ad+bc). scripts/dml_diff_probe.py
checks it against the original complex implementation.

Import this module (dml_device does so when a DirectML backend is selected) or
call apply_shims() to install the guarded monkeypatches.
"""
from __future__ import annotations

import os
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.dml_device import is_directml

# one_hot allocates an n-sized axis; refuse absurd sizes.
_MAX_ONEHOT = 8192
# ...and bound the one-hot's TOTAL footprint: idx.numel() * n elements. The
# one-hot is O(output * input), so a long expanded-mel sequence (the DiT
# `expand_states` gather) needs gigabytes. Past this budget, round-trip through
# CPU instead -- exact, and far cheaper than the one-hot it replaces.
_MAX_ONEHOT_ELEMS = 1 << 26          # 67M elements ~ 256 MB in fp32


def _subscript_letters(D: int):
    """Return (output_axis_letters, source_index_letter).

    D axes are labeled a..a+D-1; the contracted source-index / class axis gets
    the next free letter a+D.
    """
    out = list(string.ascii_lowercase[:D])
    return out, string.ascii_lowercase[D]


# ---------------------------------------------------------------------------
# exact one-hot + einsum replacements for gather / scatter_add
# ---------------------------------------------------------------------------
# torch.gather requires index.dim() == input.dim() == D. On axis `dim`, input
# carries size `n` (letter X); index carries size `m` (the output size).
# one_hot(idx, n) appends axis X as the LAST position, which einsum contracts
# against input's `dim` axis -- no broadcasting gymnastics, exact for any rank.
def _onehot_would_overflow(idx: torch.Tensor, n: int) -> bool:
    """True when the one-hot for `idx` would exceed the element budget."""
    return idx.numel() * n > _MAX_ONEHOT_ELEMS


def _cpu_gather(input, dim, index):
    """Exact torch.gather via CPU round-trip (used when one-hot is too big)."""
    return torch.gather(input.cpu(), dim, index.cpu()).to(input.device)


def safe_gather(input, dim, index):
    """torch.gather replacement, exact for any rank/axis.

    out[k] = input[..., index[k], ...] along `dim`.
    """
    if not is_directml(input.device):
        return torch.gather(input, dim, index)
    n = input.size(dim)
    if n > _MAX_ONEHOT:
        raise ValueError(f"safe_gather one-hot axis n={n} exceeds {_MAX_ONEHOT}")
    idx = index.clamp(0, n - 1).to(torch.long)
    if _onehot_would_overflow(idx, n):
        # One-hot would be O(output * input) -- gigabytes on long mel seqs.
        return _cpu_gather(input, dim, idx)
    # F.one_hot routes through scatter_, which DirectML rejects -- build the
    # one-hot mask with a comparison instead (no scatter): shape idx.shape+(n,).
    onehot = (idx.unsqueeze(-1) == torch.arange(n, device=idx.device))
    onehot = onehot.to(input.dtype)
    D = input.dim()
    out_letters, x = _subscript_letters(D)
    in_sub = out_letters[:dim] + [x] + out_letters[dim + 1:]    # X sits at the `dim` slot
    oh_sub = out_letters + [x]                                   # idx.shape + X
    return torch.einsum(f"{''.join(in_sub)},{''.join(oh_sub)}->{''.join(out_letters)}",
                        input, onehot)


def _cpu_scatter_add(input, dim, index, src):
    """Exact scatter_add via CPU round-trip (used when one-hot is too big)."""
    return input.cpu().scatter_add(dim, index.cpu(), src.cpu()).to(input.device)


def safe_scatter_add(input, dim, index, src):
    """input.copy().scatter_add(dim, index, src) replacement, exact.

    contrib[k along dim] = sum_{j: index[j]==k} src[j]
    """
    if not is_directml(input.device):
        return input.scatter_add(dim, index, src)
    n = input.size(dim)
    if n > _MAX_ONEHOT:
        raise ValueError(f"safe_scatter_add one-hot axis n={n} exceeds {_MAX_ONEHOT}")
    idx = index.clamp(0, n - 1).to(torch.long)
    if _onehot_would_overflow(idx, n):
        return _cpu_scatter_add(input, dim, idx, src)
    onehot = (idx.unsqueeze(-1) == torch.arange(n, device=idx.device))
    onehot = onehot.to(src.dtype)
    D = input.dim()
    out_letters, x = _subscript_letters(D)
    j = string.ascii_lowercase[D + 1]                            # scatter-source axis
    src_sub = out_letters[:dim] + [j] + out_letters[dim + 1:]
    oh_sub = out_letters[:dim] + [j] + out_letters[dim + 1:] + [x]
    out_sub = out_letters[:dim] + [x] + out_letters[dim + 1:]
    contrib = torch.einsum(f"{''.join(src_sub)},{''.join(oh_sub)}->{''.join(out_sub)}",
                           src, onehot)
    return input + contrib


# ---------------------------------------------------------------------------
# attention: query-axis chunking to cap the score matrix
# ---------------------------------------------------------------------------
# F.scaled_dot_product_attention materialises a [batch, heads, q_len, kv_len]
# score tensor. DirectML has no fused memory-efficient kernel, so that tensor is
# real memory: with a CFG batch of 3 at seq 4400 it is ~3.5 GB in ONE allocation,
# which is what "Could not allocate tensor with ... not enough GPU video memory"
# inside attention refers to.
#
# Attention rows are independent -- output[i] depends only on query[i] and all of
# key/value -- so splitting the query axis is mathematically exact. It caps the
# score matrix at (chunk * kv_len) instead of (q_len * kv_len). Only the
# reassociation of the concatenation differs, at fp32 rounding.
_ATTN_Q_CHUNK = 512


def chunked_sdpa(query, key, value, attn_mask=None, is_causal=False,
                 chunk: int = None):
    """scaled_dot_product_attention with the query axis processed in chunks.

    Falls through to the native call off DirectML, for short sequences, and for
    causal attention (where a query chunk would need an offset-aware mask that
    this helper does not build -- this repo only ever uses is_causal=False).
    """
    native = lambda q: F.scaled_dot_product_attention(
        q, key, value, attn_mask, is_causal=is_causal)

    if is_causal or not is_directml(query.device):
        return native(query)

    q_len = query.size(-2)
    chunk = chunk or _ATTN_Q_CHUNK
    if q_len <= chunk:
        return native(query)

    out = []
    for i in range(0, q_len, chunk):
        out.append(native(query[..., i:i + chunk, :]))
    return torch.cat(out, dim=-2)


# ---------------------------------------------------------------------------
# LSTM  (nn.LSTM -> DmlLSTM, exact param-name parity)
# ---------------------------------------------------------------------------
class DmlLSTM(nn.Module):
    """nn.LSTM replacement for DirectML.

    Parameter names mirror nn.LSTM exactly (`weight_ih_l0`, `bias_hh_l0_reverse`
    ...), so `load_state_dict(strict=True)` round-trips with zero missing keys.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(self.directions):
                sfx = "" if d == 0 else "_reverse"
                in_dim = input_size if layer == 0 else hidden_size * self.directions
                # Bare Parameters (NOT nn.Linear) so names/shapes match nn.LSTM:
                # weight_ih_lN [4H, in], weight_hh_lN [4H, H], biases [4H].
                setattr(self, f"weight_ih_l{layer}{sfx}",
                        nn.Parameter(torch.empty(hidden_size * 4, in_dim)))
                setattr(self, f"weight_hh_l{layer}{sfx}",
                        nn.Parameter(torch.empty(hidden_size * 4, hidden_size)))
                if bias:
                    setattr(self, f"bias_ih_l{layer}{sfx}",
                            nn.Parameter(torch.zeros(hidden_size * 4)))
                    setattr(self, f"bias_hh_l{layer}{sfx}",
                            nn.Parameter(torch.zeros(hidden_size * 4)))

    def _cell(self, x_t, h_t, c_t, layer, suffix):
        """LSTM cell. h_t feeds weight_hh; c_t is the cell-state recurrence.

        Gate order matches nn.LSTM exactly: [input, forget, cell, output].
        """
        w_ih = getattr(self, f"weight_ih_l{layer}{suffix}")
        w_hh = getattr(self, f"weight_hh_l{layer}{suffix}")
        # weights stored as [4H, in/H]; F.linear(x, W) auto-transposes W (computes x @ W.T).
        gates = torch.nn.functional.linear(x_t, w_ih) + \
            torch.nn.functional.linear(h_t, w_hh)
        if self.bias:
            gates = gates + getattr(self, f"bias_ih_l{layer}{suffix}")
            gates = gates + getattr(self, f"bias_hh_l{layer}{suffix}")
        i, f, g, o = gates.chunk(4, -1)
        c = torch.sigmoid(f) * c_t + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq, batch, _ = x.shape
        if hx is None:
            h = x.new_zeros(self.num_layers, self.directions, batch, self.hidden_size)
            c = x.new_zeros(self.num_layers, self.directions, batch, self.hidden_size)
        else:
            h, c = hx
            h = h.reshape(self.num_layers, self.directions, batch, self.hidden_size)
            c = c.reshape(self.num_layers, self.directions, batch, self.hidden_size)

        outputs, new_h, new_c = [], [], []
        inp = x
        for layer in range(self.num_layers):
            layer_outs = []
            for d in range(self.directions):
                sfx = "" if d == 0 else "_reverse"
                h_t, c_t = h[layer, d].clone(), c[layer, d].clone()
                seq_in = inp if d == 0 else inp.flip(0)
                o = []
                for t in range(seq_in.shape[0]):
                    h_t, c_t = self._cell(seq_in[t], h_t, c_t, layer, sfx)
                    o.append(h_t)
                o = torch.stack(o)
                if d == 1:
                    o = o.flip(0)
                layer_outs.append(o)
                new_h.append(h_t)
                new_c.append(c_t)
            inp = torch.cat(layer_outs, dim=-1)           # [seq, batch, H*directions]
        out = inp.transpose(0, 1) if self.batch_first else inp
        new_h = torch.stack(new_h).reshape(
            self.num_layers * self.directions, batch, self.hidden_size)
        new_c = torch.stack(new_c).reshape(
            self.num_layers * self.directions, batch, self.hidden_size)
        return out, (new_h, new_c)


def maybe_replace_lstm(module: nn.Module, device) -> int:
    """Swap every nn.LSTM under `module` for DmlLSTM. Returns count; no-op off DML."""
    if not is_directml(device):
        return 0
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LSTM):
            repl = DmlLSTM(child.input_size, child.hidden_size, child.num_layers,
                           child.bias, child.batch_first, child.dropout, child.bidirectional)
            repl.load_state_dict(child.state_dict(), strict=True)
            repl.to(next(child.parameters()).device)
            setattr(module, name, repl)
            count += 1
        else:
            count += maybe_replace_lstm(child, device)
    return count


# ---------------------------------------------------------------------------
# guarded monkeypatch: safety net for any DML gather in transformers internals.
# Falls through to native torch.gather for large one-hot axis (vocab dims),
# which we do not expect on DML in this pipeline -- see module docstring.
# ---------------------------------------------------------------------------
def apply_shims() -> None:
    try:
        import torch_directml as _d
    except Exception:
        return
    if not is_directml(_d.device()):
        return

    torch._orig_gather = getattr(torch, "_orig_gather", torch.gather)

    def dml_gather(input, dim, index, **kwargs):
        if is_directml(input.device) and input.size(dim) <= _MAX_ONEHOT:
            return safe_gather(input, dim, index, **kwargs)
        return torch._orig_gather(input, dim, index, **kwargs)

    torch.gather = dml_gather

    if not getattr(torch.Tensor, "_gather_shimmed", False):
        orig_method = torch.Tensor.gather

        def tensor_gather(self, dim, index=None, **kwargs):
            if index is None:           # tolerate the .gather(index) calling form
                index, dim = dim, 0
            if is_directml(self.device) and self.size(dim) <= _MAX_ONEHOT:
                return safe_gather(self, dim, index)
            return orig_method(self, dim, index, **kwargs)

        torch.Tensor.gather = tensor_gather
        torch.Tensor._gather_shimmed = True

    # scatter_add / scatter_add_: DirectML raises "The parameter is incorrect."
    # for partially-modified index dims. Route both the functional and in-place
    # forms through the exact einsum replacement.
    if not getattr(torch.Tensor, "_scatter_add_shimmed", False):
        orig_scatter_add = torch.Tensor.scatter_add
        orig_scatter_add_ = torch.Tensor.scatter_add_

        def tensor_scatter_add(self, dim, index, src):
            if is_directml(self.device) and self.size(dim) <= _MAX_ONEHOT:
                return safe_scatter_add(self, dim, index, src)
            return orig_scatter_add(self, dim, index, src)

        def tensor_scatter_add_(self, dim, index, src):
            if is_directml(self.device) and self.size(dim) <= _MAX_ONEHOT:
                # Returns a fresh tensor; all in-repo call sites rebind the
                # result (`x = ....scatter_add_(...)`), so this is equivalent.
                return safe_scatter_add(self, dim, index, src)
            return orig_scatter_add_(self, dim, index, src)

        torch.Tensor.scatter_add = tensor_scatter_add
        torch.Tensor.scatter_add_ = tensor_scatter_add_
        torch.Tensor._scatter_add_shimmed = True

    # F.pad with a MIX of positive and negative amounts: DirectML rejects it
    # ("currently doesn't support a mix of positive and negative padding
    # values"). Cropping the negative sides first and padding the positive ones
    # afterwards is exactly equivalent -- the axes are independent.
    if not getattr(torch.nn.functional, "_pad_shimmed", False):
        orig_pad = torch.nn.functional.pad

        def dml_pad(input, pad, mode="constant", value=None):
            if not is_directml(input.device) or all(p >= 0 for p in pad):
                return orig_pad(input, pad, mode, value)
            slices = [slice(None)] * input.dim()
            pos_pad = list(pad)
            for i in range(0, len(pad), 2):
                dim = input.dim() - 1 - (i // 2)
                left, right = pad[i], pad[i + 1]
                n = input.size(dim)
                start = -left if left < 0 else 0
                end = n + right if right < 0 else n
                if start or end != n:
                    slices[dim] = slice(start, end)
                pos_pad[i], pos_pad[i + 1] = max(left, 0), max(right, 0)
            out = input[tuple(slices)]
            if any(pos_pad):
                out = orig_pad(out, pos_pad, mode, value)
            return out

        torch.nn.functional.pad = dml_pad
        torch.nn.functional._pad_shimmed = True


if os.environ.get("MEGATTS3_SHIM_TORCH", "1") == "1":
    apply_shims()
