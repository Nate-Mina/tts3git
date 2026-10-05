"""Central device resolution for MegaTTS3.

One place decides which backend runs the model, so backend quirks are not
rediscovered at each call site. Precedence:

    1. MEGATTS3_DEVICE env var (explicit override, always wins)
    2. CUDA          (NVIDIA)
    3. DirectML      (AMD/Intel on Windows, device 'privateuseone:0')
    4. CPU           (always works)

DirectML note: `torch_directml.device()` takes a *string* only -- passing a
torch.device raises TypeError. Normalize to str at every boundary.
"""
import os

import torch

_DIRECTML_STR = "privateuseone:0"


def _directml_available():
    try:
        import torch_directml
        return True
    except Exception:
        return False


def _dml_has_memory() -> bool:
    """Probe whether DirectML has enough free VRAM (>= ~4 GB) to load the
    full MegaTTS3 model set.  Returns False (falling back to CPU) if the
    probe allocation fails."""
    try:
        import torch_directml as _tdml
        dev = _tdml.device()
        # Probe a 2 GB tensor; if this succeeds we have enough headroom
        # for the ~3.4 GB checkpoint pool (loaded in fp32, ~3.4 GB plus
        # activation overhead).  Smaller allocations may still succeed,
        # but the full pipeline won't fit.
        probe = torch.zeros(512 * 1024 * 1024, device=dev)
        del probe
        return True
    except Exception:
        return False


def resolve_device():
    """Return (device_str, backend_name).

    When DirectML is available, probe for enough free VRAM to hold the large
    sub-models. If the probe fails (OOM), fall back to CPU so the model loads
    on host RAM instead of crashing during ``builder.to(device)``.
    """
    override = os.environ.get("MEGATTS3_DEVICE")
    if override:
        return override, "override"

    if torch.cuda.is_available():
        return "cuda", "cuda"

    if _directml_available():
        if _dml_has_memory():
            return _DIRECTML_STR, "directml"
        print("[dml_device] DirectML available but insufficient VRAM "
              "(~4 GB needed); falling back to CPU.")

    return "cpu", "cpu"


def is_directml(device) -> bool:
    """True when device is a DirectML private-use-one device."""
    return str(device).startswith("privateuseone")


def describe_device(device) -> str:
    """Human-readable backend name for logging."""
    if is_directml(device):
        return "DirectML (AMD/Intel via torch-directml)"
    if str(device).startswith("cuda"):
        try:
            return f"CUDA: {torch.cuda.get_device_name()}"
        except Exception:
            return "CUDA"
    return "CPU"


DEVICE, BACKEND = resolve_device()


def can_allocate(device, mb: int = 2048) -> bool:
    """True when a single `mb`-sized tensor can be allocated on `device`.

    DirectML exposes no reliable free-memory query -- `torch_directml.gpu_memory()`
    returns all zeros on this driver -- so the only honest test is to try.

    Pass the device through `torch_directml.device()` rather than the raw
    'privateuseone:0' string: allocating by string raises
    `ModuleNotFoundError: No module named 'torch.privateuseone'` unless
    torch_directml has already registered the backend, which would make this
    silently report "no memory" on any cold path.
    """
    if not is_directml(device):
        return True
    try:
        import torch_directml as _tdml
        dev = _tdml.device()
        probe = torch.zeros(int(mb * 1e6 / 4), dtype=torch.float32, device=dev)
        del probe
        return True
    except Exception:
        return False


# Where the G2P (Qwen2) LM runs. Env override wins; otherwise the pipeline
# decides at load time via `resolve_g2p_device` below. G2P is invoked once per
# text segment, outside the diffusion loop, so at ~1.9 GB in fp32 it is the
# cheapest large model to evict when VRAM is scarce.
G2P_DEVICE = os.environ.get("MEGATTS3_G2P_DEVICE") or DEVICE

# Keep G2P on the GPU only if this much is still allocatable after the core
# models are resident. The generation peak runs ~3.5 GB above the resident set
# (8.18 -> 11.69 GB measured), so a smaller probe passes and still OOMs later.
#
# DirectML's allocator is tile-based and cannot hand out one large contiguous
# block -- measured on a 12 GB RX 6700 XT: 3072 MB ok, 4096 MB refused, even on
# a nearly empty card. Asking for more than the tile ceiling therefore evicts
# G2P unconditionally, which is the conservative direction we want: a needless
# CPU round-trip costs seconds, whereas guessing wrong costs the whole request.
_G2P_KEEP_GPU_MB = 4096


def resolve_g2p_device(device) -> str:
    """Pick the device for G2P once the core models are already loaded.

    Measured on a 12 GB RX 6700 XT (DirectML, fp32):

        core models resident ......... 8.18 GB   (with G2P on GPU)
        generation peak .............. 11.69 GB  -> only ~0.3 GB spare
        same, G2P moved to CPU ....... 5.04 / 9.08 GB  -> ~2.9 GB spare

    So the budget that matters is the *activation peak* (~3.5 GB over the
    resident set), not the free space at load time -- a 2 GB probe passes
    happily at 8.18 GB used and still OOMs a request later. Require enough
    headroom for the peak with margin; otherwise move G2P to CPU, where the
    cost is a few seconds per text segment (it is off the diffusion hot path).
    """
    if os.environ.get("MEGATTS3_G2P_DEVICE"):
        return os.environ["MEGATTS3_G2P_DEVICE"]      # explicit choice wins
    if not is_directml(device):
        return device                                  # CPU/CUDA: no contention
    if can_allocate(device, _G2P_KEEP_GPU_MB):
        return device
    print(f"[dml_device] <{_G2P_KEEP_GPU_MB} MB free on DirectML after the core "
          f"models; placing G2P on CPU (off the diffusion hot path).")
    return "cpu"


def autocast_ctx(device):
    """Device-correct autocast. DirectML and CPU get fp32 -- DirectML has no
    usable fp16 autocast, and the repo's `precision=torch.float16` default would
    otherwise be silently applied to the wrong backend."""
    enabled = str(device).startswith("cuda")
    return torch.amp.autocast(
        device_type="cuda" if enabled else "cpu",
        dtype=torch.float16,
        enabled=enabled,
    )
