"""GEDAI wrapper for the PSY197B EEG pipeline.

Uses the ``gedai`` Python port for leadfield-aware, multiresolution
artifact identification with SENSAI auto-thresholding.

Slot-in replacement for ICA when ``use_gedai: true`` in run_config.yaml.
Operates on the continuous MNE Raw object *after* filtering, just like
ICA does — everything downstream (epoching, ERPs, features) is unchanged.
"""

import os

import mne
from gedai import Gedai


STRENGTH_PRESETS = {
    "auto": 3.0,
    "auto+": 1.0,
    "auto-": 6.0,
}


def apply_gedai(
    raw: mne.io.BaseRaw,
    *,
    wavelet_type: str = "haar",
    wavelet_level: int = 0,
    reference_cov: str = "leadfield",
    sensai_method: str = "optimize",
    denoising_strength: str | float = "auto-",
    output_plot_dir: str | None = None,
    label: str = "",
) -> tuple[mne.io.BaseRaw, None]:
    """Clean continuous EEG with the GEDAI Python port.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG (preloaded, already filtered / re-referenced).
    wavelet_type : str
        Wavelet for multiresolution decomposition (default ``"haar"``).
    wavelet_level : int
        Decomposition level (0 = auto).
    reference_cov : str
        Reference covariance method — ``"leadfield"`` uses montage
        geometry, ``"identity"`` for a simple reference.
    sensai_method : str
        SENSAI thresholding strategy (``"optimize"`` recommended).
    denoising_strength : str or float
        Controls artifact suppression aggressiveness:
        - ``"auto"`` (default): balanced, noise_multiplier=3.0
        - ``"auto+"``: conservative (preserves signal), noise_multiplier=1.0
        - ``"auto-"``: aggressive (removes more noise), noise_multiplier=6.0
        - float: manual noise_multiplier value
    output_plot_dir : str or None
        If given, save diagnostic plots here.
    label : str
        Subject/condition tag for log messages and filenames.

    Returns
    -------
    raw_clean : mne.io.BaseRaw
        Cleaned copy of the input.
    sensai_score : None
        Placeholder for API consistency (score is logged, not returned
        by the library).
    """
    if isinstance(denoising_strength, str):
        if denoising_strength not in STRENGTH_PRESETS:
            raise ValueError(
                f"denoising_strength must be 'auto', 'auto+', 'auto-', or a float, "
                f"got '{denoising_strength}'"
            )
        noise_multiplier = STRENGTH_PRESETS[denoising_strength]
    else:
        noise_multiplier = float(denoising_strength)

    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude="bads")
    n_ch = len(picks_eeg)

    if n_ch < 3:
        print(f"    GEDAI [{label}]: too few EEG channels ({n_ch}), skipping")
        return raw, None

    if raw.get_montage() is None:
        print(f"    GEDAI [{label}]: no montage set — applying standard_1020")
        raw.set_montage("standard_1020", on_missing="warn")

    print(f"    GEDAI [{label}]: fitting on {n_ch} EEG channels "
          f"(wavelet={wavelet_type}, level={wavelet_level}, "
          f"ref_cov={reference_cov}, strength={denoising_strength})")

    gedai = Gedai(
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level,
    )

    gedai.fit_raw(
        raw,
        reference_cov=reference_cov,
        sensai_method=sensai_method,
        noise_multiplier=noise_multiplier,
    )

    raw_clean = gedai.transform_raw(raw)

    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude="bads")
    raw._data[picks_eeg] = raw_clean.get_data(picks=picks_eeg)

    print(f"    GEDAI [{label}]: transform complete")

    if output_plot_dir is not None:
        _save_diagnostics(gedai, output_plot_dir, label)

    return raw, None


def _save_diagnostics(gedai: Gedai, plot_dir: str, label: str) -> None:
    """Save the built-in GEDAI diagnostic plot(s)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        result = gedai.plot_fit()
        if result is None:
            print(f"    GEDAI [{label}]: plot_fit() returned None")
            return

        figs = result if isinstance(result, (list, tuple)) else [result]
        for i, fig in enumerate(figs):
            out = os.path.join(plot_dir, f"{label}_gedai_fit_{i}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"    GEDAI [{label}]: {len(figs)} diagnostic plot(s) → {plot_dir}")
    except Exception as exc:
        print(f"    GEDAI [{label}]: plot export skipped — {exc}")
