"""
PSY197B — Model Interpretability
================================
Spatial/temporal filter visualization and saliency maps for EEGNet.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch


def plot_spatial_filters(model, epoch_fif_path, save_dir):
    """Plot depthwise conv spatial filters as scalp topomaps.

    Extracts block1[2] (depthwise conv) weights → (F1*D, n_channels).
    Each row is a spatial filter plotted via mne.viz.plot_topomap.
    """
    w = model.block1[2].weight.detach().cpu().numpy()
    # shape: (F1*D, 1, n_channels, 1) → squeeze to (F1*D, n_channels)
    w = w.squeeze()
    n_filters = w.shape[0]

    epochs = mne.read_epochs(epoch_fif_path, preload=False, verbose=False)
    info = epochs.info

    os.makedirs(save_dir, exist_ok=True)

    cols = min(n_filters, 4)
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n_filters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_filters):
        mne.viz.plot_topomap(w[i], info, axes=axes[i], show=False)
        axes[i].set_title(f"Filter {i+1}")
    for i in range(n_filters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("EEGNet Spatial Filters (Depthwise Conv)", y=1.02)
    fig.tight_layout()
    out = os.path.join(save_dir, "spatial_filters_topomap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_temporal_filters(model, sfreq, save_dir):
    """Plot first-layer temporal conv filters as 1D waveforms + FFT insets.

    Extracts block1[0] weights → (F1, 1, 1, kernel_length).
    """
    w = model.block1[0].weight.detach().cpu().numpy()
    # shape: (F1, 1, 1, kernel_length) → (F1, kernel_length)
    w = w.squeeze()
    if w.ndim == 1:
        w = w[np.newaxis, :]
    n_filters, kernel_len = w.shape

    os.makedirs(save_dir, exist_ok=True)

    time_ms = np.arange(kernel_len) / sfreq * 1000

    cols = min(n_filters, 4)
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_filters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_filters):
        ax = axes[i]
        ax.plot(time_ms, w[i], "b-", linewidth=1.2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Weight")
        ax.set_title(f"Temporal Filter {i+1}")

        # FFT inset
        fft_vals = np.abs(np.fft.rfft(w[i]))
        freqs = np.fft.rfftfreq(kernel_len, d=1.0 / sfreq)
        inset = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        inset.plot(freqs, fft_vals, "r-", linewidth=0.8)
        inset.set_xlim(0, sfreq / 2)
        inset.set_xlabel("Hz", fontsize=7)
        inset.set_ylabel("|FFT|", fontsize=7)
        inset.tick_params(labelsize=6)
        peak_freq = freqs[np.argmax(fft_vals[1:]) + 1]
        inset.axvline(peak_freq, color="gray", ls="--", lw=0.5)
        inset.set_title(f"Peak: {peak_freq:.1f} Hz", fontsize=7)

    for i in range(n_filters, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("EEGNet Temporal Filters (1st Conv Layer)", y=1.02)
    fig.tight_layout()
    out = os.path.join(save_dir, "temporal_filters.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_saliency_topomap(model, X_batch, epoch_fif_path, save_dir,
                          target_class=1):
    """Compute input-gradient saliency and plot as a scalp topomap.

    Averages |∂L/∂x| across trials and time → (n_channels,) importance.
    """
    device = next(model.parameters()).device
    model.eval()

    x = torch.from_numpy(X_batch).float().to(device)
    x.requires_grad_(True)

    logits = model(x)
    score = logits[:, target_class].sum()
    score.backward()

    grad = x.grad.detach().cpu().numpy()  # (B, 1, C, T) or (B, C, T)
    if grad.ndim == 4:
        grad = grad.squeeze(1)  # (B, C, T)
    importance = np.abs(grad).mean(axis=(0, 2))  # (C,)

    epochs = mne.read_epochs(epoch_fif_path, preload=False, verbose=False)
    info = epochs.info

    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im, _ = mne.viz.plot_topomap(importance, info, axes=ax, show=False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Mean |gradient|")
    class_name = "CR" if target_class == 1 else "FA"
    ax.set_title(f"Input Saliency (target: {class_name})")

    out = os.path.join(save_dir, f"saliency_topomap_class{target_class}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out
