"""Local stand-in for the `spaces` package (HF Spaces only).

app.py does `import spaces` and decorates the inference function with
`@spaces.GPU`. Running locally there is no Spaces runtime to talk to, so this
provides the same no-op surface. Replace with the real `spaces` package if you
ever deploy to a ZeroGPU Space.
"""


def GPU(func=None, **kwargs):
    """No-op stand-in for spaces.GPU: run the function inline."""
    if func is None:
        def _decorator(f):
            return f
        return _decorator
    return func


def GradIO(*args, **kwargs):
    """No-op stand-in for spaces.GradIO."""
    raise NotImplementedError(
        "spaces.GradIO requires the real `spaces` package (HF Spaces only)."
    )


__version__ = "0.0.0-local-shim"
