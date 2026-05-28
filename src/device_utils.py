"""Shared torch device selection + a one-line startup banner.

Used by main.py, train.py, vision/vision_main.py, and dashboards so that
every entry point prints which compute device is actually being used.
This prevents silent CPU-only runs (e.g. when the wrong torch wheel is
installed) from going unnoticed.
"""

from __future__ import annotations

import os

try:
    import torch
except Exception:
    torch = None


def get_device():
    """Return the best available torch.device: cuda > mps > cpu."""
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_cuda() -> bool:
    """True iff CUDA is available (i.e. pin_memory + GPU inference are worth it)."""
    return torch is not None and torch.cuda.is_available()


def device_summary() -> str:
    """One-line human-readable summary of the active compute device."""
    if torch is None:
        return "torch not installed"
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        cuda_v = getattr(torch.version, "cuda", "?")
        return (
            f"CUDA ({name}, {vram:.1f} GB VRAM, "
            f"torch {torch.__version__}, cuda {cuda_v})"
        )
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return f"MPS (Apple Silicon, torch {torch.__version__})"
    return (
        f"CPU only (torch {torch.__version__}) -- no GPU detected. "
        "If you have an NVIDIA GPU, install the CUDA build of torch."
    )


def print_device_banner(prefix: str = "") -> None:
    """Print a one-line device banner; safe to call multiple times.

    Honors the env var PSY_QUIET_DEVICE=1 to suppress (e.g. in repeated
    subprocess invocations).
    """
    if os.environ.get("PSY_QUIET_DEVICE", "").strip() in {"1", "true", "True"}:
        return
    line = f"{prefix}Compute device: {device_summary()}"
    print(line, flush=True)


# ── Compute settings (AMP, DataLoader workers) ────────────────

_DEFAULT_COMPUTE = {
    "use_amp": True,
    "dataloader_workers": 8,
    "prefetch_factor": 4,
    "persistent_workers": True,
    "cudnn_benchmark": True,
    # channels_last default OFF: measured slower than NCHW on our
    # 4090 + torch 2.10 + cu126 setup. Opt in if you've verified a win.
    "channels_last": False,
}


def _load_compute_settings() -> dict:
    """Read configs/config.yaml -> compute: block. Falls back to defaults.

    Honors env-var overrides: PSY_USE_AMP, PSY_DATALOADER_WORKERS.
    Cached after first call.
    """
    global _COMPUTE_CACHE
    if "_COMPUTE_CACHE" in globals() and _COMPUTE_CACHE is not None:
        return _COMPUTE_CACHE

    settings = dict(_DEFAULT_COMPUTE)

    try:
        import yaml
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(os.path.dirname(here), "configs", "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            compute = cfg.get("compute") or {}
            for k in settings:
                if k in compute:
                    settings[k] = compute[k]
    except Exception:
        pass

    env_amp = os.environ.get("PSY_USE_AMP")
    if env_amp is not None:
        settings["use_amp"] = env_amp.strip().lower() in {"1", "true", "yes"}
    env_workers = os.environ.get("PSY_DATALOADER_WORKERS")
    if env_workers is not None:
        try:
            settings["dataloader_workers"] = max(0, int(env_workers))
        except ValueError:
            pass

    _COMPUTE_CACHE = settings
    return settings


_COMPUTE_CACHE = None


def use_amp() -> bool:
    """True iff AMP should be enabled (CUDA available AND config opted in)."""
    return use_cuda() and bool(_load_compute_settings().get("use_amp", True))


def _running_under_streamlit() -> bool:
    """True iff we appear to be running inside a Streamlit script.

    Streamlit re-imports the entire script in each spawned multiprocessing
    worker on Windows, which is incompatible with DataLoader workers > 0
    (the worker boots, re-runs Streamlit setup, often crashes on missing
    files or sidebar state). When detected we force num_workers=0.
    """
    if os.environ.get("STREAMLIT_SERVER_PORT"):
        return True
    if os.environ.get("STREAMLIT_RUNTIME") == "1":
        return True
    try:
        import sys
        return "streamlit" in sys.modules and "streamlit.runtime" in sys.modules
    except Exception:
        return False


def _windows_commit_budget_workers(default_n: int) -> int:
    """Cap worker count so we don't blow Windows page-file commit charge.

    Each spawned DataLoader worker re-imports torch/scipy/sklearn from
    scratch — that's roughly 2-3 GB of address space per worker. On
    Windows with a default-sized page file we've seen
    'paging file is too small' ImportErrors at >= ~8 workers when the
    parent process also has CUDA loaded.
    """
    if os.name != "nt":
        return default_n
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return min(default_n, 6)
    per_worker_gb = 2.5
    parent_overhead_gb = 4.0
    safe_n = max(0, int((avail_gb - parent_overhead_gb) / per_worker_gb))
    # 2–4 workers: keeps GPU fed without 8× torch re-import on spawn.
    capped = min(default_n, max(2, safe_n), 4)
    return capped if capped >= 1 else 0


def dataloader_workers() -> int:
    """Number of DataLoader worker processes from config (default 8).

    Forced to 0 when running under Streamlit on any OS: Streamlit's
    script-rerun model + Windows spawn re-imports the whole app script
    inside each worker, which is fundamentally unsafe.

    On Windows, also capped by available RAM (each spawn-worker re-imports
    the full torch stack, ~2.5 GB committed memory per worker).
    """
    if _running_under_streamlit():
        return 0
    n = int(_load_compute_settings().get("dataloader_workers", 8))
    return _windows_commit_budget_workers(n)


def persistent_workers() -> bool:
    """Whether DataLoader should keep workers alive between epochs."""
    return bool(_load_compute_settings().get("persistent_workers", True))


def prefetch_factor() -> int:
    """Number of batches each DataLoader worker prefetches."""
    return int(_load_compute_settings().get("prefetch_factor", 4))


def use_channels_last() -> bool:
    """Whether to use torch.channels_last memory format for conv nets."""
    return use_cuda() and bool(_load_compute_settings().get("channels_last", True))


def enable_cudnn_benchmark() -> None:
    """Turn on cuDNN's kernel autotuner. Safe to call multiple times.

    Should be called once at startup for any training script that has
    fixed-shape conv inputs (ResNet, EEGNet, etc.).
    """
    if torch is None or not torch.cuda.is_available():
        return
    if not _load_compute_settings().get("cudnn_benchmark", True):
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def make_autocast(device, enabled=None):
    """Return a torch.amp.autocast context manager (or nullcontext if disabled).

    Usage:
        with make_autocast(device):
            logits = model(x)
            loss = loss_fn(logits, y)
    """
    import contextlib
    if enabled is None:
        enabled = use_amp()
    if not enabled or device is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(enabled=None):
    """Return torch.amp.GradScaler('cuda') or a no-op stand-in if disabled."""
    if enabled is None:
        enabled = use_amp()

    class _NoopScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def unscale_(self, optimizer):
            pass

    if not enabled or torch is None or not torch.cuda.is_available():
        return _NoopScaler()
    return torch.amp.GradScaler("cuda")
