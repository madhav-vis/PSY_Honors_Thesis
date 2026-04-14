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


# ── Visualization C: CLIP vs Human Agreement ─────────────────

def plot_clip_vs_human(results_df, human_labels_df, out_path):
    """Confusion matrix and per-category accuracy for CLIP vs hand labels."""
    if human_labels_df is None:
        print("    No human labels found — skipping CLIP vs Human plot")
        return

    merged = results_df.merge(
        human_labels_df[["fixation_id", "human_label"]],
        on="fixation_id",
        how="inner",
    )
    if merged.empty:
        print("    No matching labels for CLIP vs Human comparison")
        return

    cat_list = list(CATEGORIES.keys())
    cat_to_idx = {c: i for i, c in enumerate(cat_list)}
    n_cats = len(cat_list)

    conf_matrix = np.zeros((n_cats, n_cats), dtype=int)
    for _, row in merged.iterrows():
        human = row["human_label"]
        clip_pred = row["gaze_target_category"]
        if human in cat_to_idx and clip_pred in cat_to_idx:
            conf_matrix[cat_to_idx[human], cat_to_idx[clip_pred]] += 1

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(
        conf_matrix.astype(float), row_sums,
        where=row_sums > 0, out=np.zeros_like(conf_matrix, dtype=float),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    im = ax1.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(n_cats):
        for j in range(n_cats):
            ax1.text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                     fontsize=10, color="white" if norm_matrix[i, j] > 0.5 else "black")
    ax1.set_xticks(range(n_cats))
    ax1.set_xticklabels(cat_list, rotation=45, ha="right")
    ax1.set_yticks(range(n_cats))
    ax1.set_yticklabels(cat_list)
    ax1.set_xlabel("CLIP Prediction")
    ax1.set_ylabel("Human Label")
    ax1.set_title("CLIP vs Human Labels")
    plt.colorbar(im, ax=ax1, label="Recall")

    per_cat_acc = []
    for c in cat_list:
        mask = merged["human_label"] == c
        if mask.sum() == 0:
            per_cat_acc.append(0.0)
        else:
            per_cat_acc.append((merged.loc[mask, "gaze_target_category"] == c).mean())

    overall = (merged["gaze_target_category"] == merged["human_label"]).mean()
    print(f"    Overall CLIP accuracy: {overall:.1%}")

    bar_colors = ["red" if a < 0.5 else "gold" if a < 0.75 else "green" for a in per_cat_acc]
    ax2.bar(cat_list, per_cat_acc, color=bar_colors, edgecolor="black")
    ax2.axhline(y=overall, color="black", linestyle="--", linewidth=1.5,
                label=f"Overall: {overall:.1%}")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Per-Category Accuracy (Overall: {overall:.1%})")
    ax2.legend()
    ax2.set_xticklabels(cat_list, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── Debug: Annotated Sample Frames ───────────────────────────

def save_debug_frames(results_df, frames_dict, out_dir, n=10):
    """Save individual full frames annotated with gaze crop box and CLIP label.

    Intended for quick visual sanity checking, not routine use.
    """
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


# ── Visualization D: Embedding Cluster Map + Examples ─────────

def plot_embedding_clusters(embeddings, cluster_labels, results_df,
                            crops_dir, out_path, n_examples=5):
    """UMAP/PCA projection of embeddings colored by cluster, plus example crops."""
    from sklearn.decomposition import PCA as _PCA

    n_clusters = int(cluster_labels.max()) + 1
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    # 2D projection
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                            random_state=42)
        coords = reducer.fit_transform(embeddings)
        proj_label = "UMAP"
    except ImportError:
        print("    Warning: umap-learn not installed — falling back to PCA 2D")
        pca2 = _PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(embeddings)
        proj_label = "PCA"

    fig_height = 6 + 3 * n_clusters
    fig = plt.figure(figsize=(20, fig_height))
    gs = GridSpec(1 + n_clusters, n_examples, figure=fig,
                  height_ratios=[3] + [1] * n_clusters,
                  hspace=0.35, wspace=0.05)

    # Top: scatter
    ax_scatter = fig.add_subplot(gs[0, :])
    for k in range(n_clusters):
        mask = cluster_labels == k
        ax_scatter.scatter(coords[mask, 0], coords[mask, 1],
                           c=[cmap(k)], s=3, alpha=0.6, label=f"Cluster {k}")
    ax_scatter.legend(fontsize=8, markerscale=4, loc="upper right")
    ax_scatter.set_title(
        f"CLIP Embedding Space — {proj_label} Projection ({len(embeddings)} fixations)\n"
        "Each point = one fixation. Nearby points = similar visual content.",
        fontsize=14,
    )
    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])

    # Bottom: example crops per cluster
    fix_ids = results_df["fixation_id"].values
    ts_ns_arr = results_df["timestamp_ns"].values

    for k in range(n_clusters):
        cluster_mask = np.where(cluster_labels == k)[0]
        count = len(cluster_mask)
        rng = np.random.RandomState(42 + k)
        sample_idx = rng.choice(cluster_mask,
                                min(n_examples, count), replace=False)

        for col_i in range(n_examples):
            ax = fig.add_subplot(gs[1 + k, col_i])
            if col_i < len(sample_idx):
                idx = sample_idx[col_i]
                fid = int(fix_ids[idx])
                ts = int(ts_ns_arr[idx])
                crop_path = os.path.join(crops_dir, f"{fid}_{ts}.png")
                if os.path.exists(crop_path):
                    img = cv2.imread(crop_path)
                    if img is not None:
                        ax.imshow(img[:, :, ::-1])
                    else:
                        ax.set_facecolor("#e0e0e0")
                else:
                    ax.set_facecolor("#e0e0e0")
            else:
                ax.set_facecolor("#e0e0e0")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_i == 0:
                ax.set_ylabel(f"C{k} (n={count})", fontsize=9, rotation=0,
                              labelpad=45, va="center")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── Visualization E: Optimal K Analysis ──────────────────────

def plot_optimal_k(k_analysis_dict, out_path):
    """Elbow curve and silhouette scores for choosing n_clusters."""
    ks = k_analysis_dict["k_values"]
    inertias = k_analysis_dict["inertias"]
    sils = k_analysis_dict["silhouettes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, inertias, "o-", linewidth=2)
    if len(inertias) >= 3:
        second_deriv = np.diff(inertias, n=2)
        elbow_idx = int(np.argmax(np.abs(second_deriv))) + 1
        ax1.axvline(x=ks[elbow_idx], color="red", linestyle="--",
                    label=f"Elbow at k={ks[elbow_idx]}")
        ax1.legend()
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Inertia vs K (Elbow Method)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks, sils, "o-", linewidth=2, color="green")
    best_idx = int(np.argmax(sils))
    ax2.axvline(x=ks[best_idx], color="red", linestyle="--",
                label=f"Best k={ks[best_idx]} (sil={sils[best_idx]:.3f})")
    ax2.legend()
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs K")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Optimal Number of Clusters Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


# ── Visualization F: Cluster Timeline ────────────────────────

def plot_cluster_timeline(results_df, out_path):
    """Cluster sequence over time and proportion in 30s bins."""
    if "cluster_id" not in results_df.columns:
        print("    No cluster_id column — skipping cluster timeline")
        return

    df = results_df.copy()
    n_clusters = int(df["cluster_id"].max()) + 1
    cmap = plt.cm.get_cmap("tab10", n_clusters)

    t_min = df["timestamp_s"].min()
    df["rel_time_s"] = df["timestamp_s"] - t_min

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [1, 1]})

    # Top: scatter
    for k in range(n_clusters):
        mask = df["cluster_id"] == k
        ax1.scatter(df.loc[mask, "rel_time_s"], df.loc[mask, "cluster_id"],
                    c=[cmap(k)], s=4, alpha=0.7, label=f"Cluster {k}")
    ax1.set_yticks(range(n_clusters))
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_title("Visual Cluster Sequence Over Session", fontsize=14)
    ax1.legend(fontsize=8, markerscale=3, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: stacked area
    bin_size = 30
    max_t = df["rel_time_s"].max()
    bins = np.arange(0, max_t + bin_size, bin_size)
    df["time_bin"] = pd.cut(df["rel_time_s"], bins=bins,
                            labels=bins[:-1], include_lowest=True)

    proportions = pd.DataFrame(index=bins[:-1],
                               columns=list(range(n_clusters)), data=0.0)
    for b in bins[:-1]:
        subset = df[df["time_bin"] == b]
        if len(subset) == 0:
            continue
        counts = subset["cluster_id"].value_counts()
        for c in range(n_clusters):
            proportions.loc[b, c] = counts.get(c, 0) / len(subset)

    ax2.stackplot(
        proportions.index.astype(float),
        *[proportions[c].values.astype(float) for c in range(n_clusters)],
        labels=[f"Cluster {c}" for c in range(n_clusters)],
        colors=[cmap(c) for c in range(n_clusters)],
        alpha=0.8,
    )
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Proportion", fontsize=12)
    ax2.set_title("Cluster Distribution Over Time (30s bins)", fontsize=14)
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")
