"""Publication-quality visualizations for the gaze-contingent vision pipeline."""

import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from .config import CATEGORIES, CATEGORY_COLORS, CROP_SIZE


def _conf_color_rgb(conf):
    """Confidence → (R, G, B) tuple for matplotlib (0-1 floats)."""
    if conf > 0.45:
        return (0, 0.8, 0)
    elif conf > 0.25:
        return (0.9, 0.85, 0)
    return (0.9, 0, 0)


def _conf_color_bgr(conf):
    """Confidence → (B, G, R) tuple for OpenCV (0-255 ints)."""
    if conf > 0.45:
        return (0, 200, 0)
    elif conf > 0.25:
        return (0, 220, 220)
    return (0, 0, 220)


# ── Visualization A: Labeled Frame Grid ──────────────────────

def plot_labeled_frame_grid(results_df, frames_dict, gaze_df, out_path,
                            n=12, crops_dir=None):
    """Grid of world frames + gaze crops with top-3 classification scores."""
    if results_df.empty:
        print("    No results to plot for frame grid")
        return

    available_ts = set(frames_dict.keys())
    df = results_df[results_df["timestamp_ns"].isin(available_ts)].copy()
    if df.empty:
        print("    No frames matched results — skipping grid")
        return

    # Category-diverse sampling
    sampled = []
    n_cats = df["gaze_target_category"].nunique()
    per_cat = max(1, n // n_cats) if n_cats > 0 else n
    for cat in df["gaze_target_category"].unique():
        rows = df[df["gaze_target_category"] == cat]
        sampled.append(rows.sample(min(per_cat, len(rows)), random_state=42))
    sampled = pd.concat(sampled).drop_duplicates(subset="fixation_id")
    if len(sampled) < n:
        remaining = df[~df["fixation_id"].isin(sampled["fixation_id"])]
        if not remaining.empty:
            extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=42)
            sampled = pd.concat([sampled, extra])
    sampled = sampled.head(n)
    n_actual = len(sampled)

    pairs_per_row = 4
    n_rows = math.ceil(n_actual / pairs_per_row)
    fig = plt.figure(figsize=(22, 5 * n_rows))
    gs = GridSpec(n_rows, pairs_per_row * 2, figure=fig, wspace=0.05, hspace=0.4)

    score_cols = [c for c in results_df.columns if c.startswith("score_")]

    for idx, (_, row) in enumerate(sampled.iterrows()):
        grid_row = idx // pairs_per_row
        grid_col = (idx % pairs_per_row) * 2

        ts_ns = int(row["timestamp_ns"])
        frame = frames_dict.get(ts_ns)
        if frame is None:
            continue

        display_frame = frame[:, :, ::-1].copy()
        gx, gy = int(round(row["gaze_x_px"])), int(round(row["gaze_y_px"]))
        conf = row["confidence"]
        cat = row["gaze_target_category"]

        r, g, b = _conf_color_rgb(conf)
        bgr = (int(b * 255), int(g * 255), int(r * 255))

        half = CROP_SIZE // 2
        h, w = display_frame.shape[:2]
        x1 = max(gx - half, 0)
        y1 = max(gy - half, 0)
        x2 = min(gx + half, w)
        y2 = min(gy + half, h)

        annotated = display_frame.copy()
        cv2.circle(annotated, (gx, gy), 20, bgr, 3)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

        # Left panel: full frame
        ax_frame = fig.add_subplot(gs[grid_row, grid_col])
        ax_frame.imshow(annotated)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])

        # Title: top-3 scores
        title_l1 = f"{cat} ({conf:.0%})"
        if score_cols:
            scores = {c.replace("score_", ""): row[c] for c in score_cols}
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            if len(ranked) >= 3:
                title_l2 = f"2nd: {ranked[1][0]} {ranked[1][1]:.0%}  |  3rd: {ranked[2][0]} {ranked[2][1]:.0%}"
            else:
                title_l2 = ""
            ax_frame.set_title(f"{title_l1}\n{title_l2}", fontsize=8, loc="left")
        else:
            ax_frame.set_title(title_l1, fontsize=8, loc="left")

        # Right panel: crop
        ax_crop = fig.add_subplot(gs[grid_row, grid_col + 1])
        crop_img = None
        if crops_dir:
            fid = int(row["fixation_id"])
            crop_path = os.path.join(crops_dir, f"{fid}_{ts_ns}.png")
            if os.path.exists(crop_path):
                crop_bgr = cv2.imread(crop_path)
                if crop_bgr is not None:
                    crop_img = crop_bgr[:, :, ::-1]

        if crop_img is not None:
            border_w = 4
            bordered = np.full(
                (crop_img.shape[0] + 2 * border_w,
                 crop_img.shape[1] + 2 * border_w, 3),
                fill_value=int(r * 255), dtype=np.uint8,
            )
            bordered[:, :, 0] = int(r * 255)
            bordered[:, :, 1] = int(g * 255)
            bordered[:, :, 2] = int(b * 255)
            bordered[border_w:-border_w, border_w:-border_w] = crop_img
            ax_crop.imshow(bordered)
        else:
            ax_crop.text(0.5, 0.5, "crop missing", ha="center", va="center",
                         fontsize=8, color="grey")
            ax_crop.set_facecolor("#f0f0f0")
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])

    fig.suptitle("Labeled Fixation Frames", fontsize=18, y=1.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── Visualization B: Category Timeline ───────────────────────

def plot_category_timeline(results_df, out_path, min_confidence=0.25):
    """Scatter timeline, stacked area, and confidence histogram."""
    if results_df.empty:
        print("    No results to plot for timeline")
        return

    cat_list = list(CATEGORIES.keys())
    colors_list = [CATEGORY_COLORS[c] for c in cat_list]

    n_total = len(results_df)
    df = results_df[results_df["confidence"] >= min_confidence].copy()
    n_filtered = n_total - len(df)
    if n_filtered > 0:
        print(f"    Timeline: filtered {n_filtered} low-confidence fixations (<{min_confidence:.0%})")

    # If no fixations pass the filter, use all of them so the plots
    # are still generated (the histogram is especially useful here).
    if df.empty:
        print("    All fixations below confidence threshold — showing all unfiltered")
        df = results_df.copy()
        n_filtered = 0

    t_min = df["timestamp_s"].min()
    df["rel_time_s"] = df["timestamp_s"] - t_min

    cat_to_y = {c: i for i, c in enumerate(cat_list)}

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": [1, 1, 0.6]},
    )
    subtitle = (f"(showing {len(df)} fixations, confidence ≥ {min_confidence:.0%}, "
                f"{n_filtered} low-conf removed)")

    # Top: scatter
    y_vals = df["gaze_target_category"].map(cat_to_y)
    valid = y_vals.notna()
    point_colors = [CATEGORY_COLORS.get(c, "#999999") for c in df.loc[valid, "gaze_target_category"]]
    ax1.scatter(
        df.loc[valid, "rel_time_s"],
        y_vals[valid],
        c=point_colors,
        s=df.loc[valid, "duration_ms"].clip(upper=500) * 0.1,
        alpha=0.6,
        edgecolors="none",
    )
    ax1.set_yticks(range(len(cat_list)))
    ax1.set_yticklabels(cat_list)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title(f"Gaze Target Category Over Session\n{subtitle}", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Middle: stacked area in 30-second bins
    bin_size = 30
    max_t = df["rel_time_s"].max()
    bins = np.arange(0, max_t + bin_size, bin_size)
    df["time_bin"] = pd.cut(df["rel_time_s"], bins=bins, labels=bins[:-1], include_lowest=True)

    proportions = pd.DataFrame(index=bins[:-1], columns=cat_list, data=0.0)
    for b in bins[:-1]:
        subset = df[df["time_bin"] == b]
        if len(subset) == 0:
            continue
        counts = subset["gaze_target_category"].value_counts()
        for c in cat_list:
            proportions.loc[b, c] = counts.get(c, 0) / len(subset)

    ax2.stackplot(
        proportions.index.astype(float),
        *[proportions[c].values.astype(float) for c in cat_list],
        labels=cat_list,
        colors=colors_list,
        alpha=0.8,
    )
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Proportion", fontsize=12)
    ax2.set_title("Category Distribution Over Time (30s bins)", fontsize=14)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Bottom: confidence histogram (uses ALL fixations, not filtered)
    all_conf = results_df["confidence"].values
    bin_edges = np.linspace(0, 1, 21)
    counts_hist, _ = np.histogram(all_conf, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bar_colors = []
    for bc in bin_centers:
        if bc < 0.25:
            bar_colors.append("#FF4444")
        elif bc < 0.45:
            bar_colors.append("#FFD700")
        else:
            bar_colors.append("#22AA22")

    ax3.bar(bin_centers, counts_hist, width=0.045, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax3.axvline(x=0.25, color="black", linestyle="--", linewidth=1.2)
    ax3.axvline(x=0.45, color="black", linestyle="--", linewidth=1.2)
    ax3.text(0.25, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 1,
             " chance", fontsize=8, va="top")
    ax3.text(0.45, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 1,
             " reliable", fontsize=8, va="top")
    ax3.set_xlabel("Confidence", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Confidence Distribution", fontsize=14)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── Debug: Annotated Sample Frames ───────────────────────────

def save_debug_frames(results_df, frames_dict, out_dir, n=10):
    """Save individual full frames annotated with gaze crop box and predicted label."""
    os.makedirs(out_dir, exist_ok=True)

    if results_df.empty:
        print("    No results for debug frames")
        return

    sample = results_df.sample(min(n, len(results_df)), random_state=42)

    saved = 0
    for _, row in sample.iterrows():
        ts_ns = int(row["timestamp_ns"])
        frame = frames_dict.get(ts_ns)
        if frame is None:
            continue

        img = frame.copy()
        gx, gy = int(round(row["gaze_x_px"])), int(round(row["gaze_y_px"]))
        cat = row["gaze_target_category"]
        conf = row["confidence"]

        half = CROP_SIZE // 2
        h, w = img.shape[:2]
        x1 = max(gx - half, 0)
        y1 = max(gy - half, 0)
        x2 = min(gx + half, w)
        y2 = min(gy + half, h)

        color = _conf_color_bgr(conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.circle(img, (gx, gy), 8, color, -1)

        label_text = f"{cat} ({conf:.0%})"
        text_x = min(x2 + 10, w - 300)
        text_y = max(y1 + 30, 30)

        cv2.putText(img, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        fid = int(row["fixation_id"])
        out_path_dbg = os.path.join(out_dir, f"debug_{fid}_{cat}.jpg")
        cv2.imwrite(out_path_dbg, img)
        saved += 1

    print(f"    Saved {saved} debug frames to {out_dir}")
