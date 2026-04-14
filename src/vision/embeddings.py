"""Clustering and analysis of CLIP image embeddings."""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances


def cluster_embeddings(embeddings, n_clusters=7, random_state=42):
    """K-Means clustering on PCA-reduced CLIP embeddings.

    Returns (cluster_labels, kmeans_model).
    """
    n = len(embeddings)
    print(f"    K-Means clustering: {n} fixations → {n_clusters} clusters")

    pca = PCA(n_components=min(50, n, embeddings.shape[1]),
              random_state=random_state)
    embs_pca = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    top10 = explained[:10]
    print(f"    PCA top-10 explained variance: "
          f"{', '.join(f'{v:.3f}' for v in top10)}  "
          f"(cumulative {sum(top10):.3f})")

    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=20, max_iter=500)
    labels = km.fit_predict(embs_pca)

    counts = pd.Series(labels).value_counts().sort_index()
    print("    Cluster sizes:")
    for cid, cnt in counts.items():
        print(f"      cluster {cid}: {cnt}")

    return labels, km


def find_optimal_k(embeddings, k_range=range(3, 12), random_state=42):
    """Run K-Means for each k, compute inertia and silhouette scores."""
    pca = PCA(n_components=min(50, len(embeddings), embeddings.shape[1]),
              random_state=random_state)
    embs_pca = pca.fit_transform(embeddings)

    n_sample = min(2000, len(embs_pca))
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(len(embs_pca), n_sample, replace=False)
    embs_sample = embs_pca[sample_idx]

    k_values, inertias, silhouettes = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state,
                    n_init=20, max_iter=500)
        km.fit(embs_pca)
        labels_sample = km.predict(embs_sample)
        sil = silhouette_score(embs_sample, labels_sample)

        k_values.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:12.1f}  silhouette={sil:.4f}")

    best_k = k_values[int(np.argmax(silhouettes))]
    print(f"    Best k by silhouette: {best_k}")

    return {"k_values": k_values, "inertias": inertias,
            "silhouettes": silhouettes}


def compute_trial_embedding_features(results_df, et_prepro_df,
                                      embeddings_array):
    """Aggregate embedding/cluster features to trial level for EEG fusion."""
    fid_to_row = {int(fid): i for i, fid in
                  enumerate(results_df["fixation_id"].values)}

    n_clusters = int(results_df["cluster_id"].max()) + 1
    cluster_cols = [f"vis_cluster_{c}" for c in range(n_clusters)]

    records = []
    n_empty = 0
    for _, trial in et_prepro_df.iterrows():
        t_idx = trial["trialIdx"]
        t_time = trial["trigger_time"]

        window = results_df[
            (results_df["timestamp_s"] >= t_time - 1.0)
            & (results_df["timestamp_s"] <= t_time + 1.0)
        ]

        rec = {"trialIdx": t_idx}

        if window.empty:
            n_empty += 1
            rec["dominant_cluster"] = np.nan
            rec["cluster_entropy"] = np.nan
            rec["n_fixations_in_window"] = 0
            rec["emb_spread"] = np.nan
            rec["mean_embedding"] = np.nan
            for cc in cluster_cols:
                rec[cc] = np.nan
        else:
            counts = window["cluster_id"].value_counts()
            rec["dominant_cluster"] = int(counts.index[0])
            rec["n_fixations_in_window"] = len(window)

            count_arr = np.array(
                [counts.get(c, 0) for c in range(n_clusters)], dtype=float
            )
            total = count_arr.sum()
            probs = count_arr / total
            rec["cluster_entropy"] = float(sp_entropy(probs, base=2))

            for c in range(n_clusters):
                rec[f"vis_cluster_{c}"] = counts.get(c, 0) / total

            row_indices = [fid_to_row[int(fid)]
                           for fid in window["fixation_id"]
                           if int(fid) in fid_to_row]
            if row_indices:
                embs_win = embeddings_array[row_indices]
                mean_emb = embs_win.mean(axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb = mean_emb / norm
                rec["mean_embedding"] = json.dumps(mean_emb.tolist())

                if len(row_indices) > 1:
                    dists = cosine_distances(embs_win)
                    triu_idx = np.triu_indices(len(embs_win), k=1)
                    rec["emb_spread"] = float(dists[triu_idx].mean())
                else:
                    rec["emb_spread"] = 0.0
            else:
                rec["mean_embedding"] = np.nan
                rec["emb_spread"] = np.nan

        records.append(rec)

    df = pd.DataFrame(records)

    n_with = (df["n_fixations_in_window"] > 0).sum()
    print(f"    Built embedding features for {n_with} / {len(df)} trials")
    if n_empty > 0:
        print(f"    {n_empty} trials had 0 fixations in ±1s window")

    return df


def save_embeddings(embeddings, fixation_ids, out_path):
    """Save embeddings as .npy and companion fixation IDs as .csv."""
    np.save(f"{out_path}.npy", embeddings)
    pd.DataFrame({"fixation_id": fixation_ids}).to_csv(
        f"{out_path}_ids.csv", index=False
    )
    print(f"    Saved {len(embeddings)} embeddings to {out_path}.npy")


def load_embeddings(out_path):
    """Load embeddings .npy and companion fixation IDs .csv."""
    embs = np.load(f"{out_path}.npy")
    ids_df = pd.read_csv(f"{out_path}_ids.csv")
    fixation_ids = ids_df["fixation_id"].tolist()
    print(f"    Loaded {len(embs)} embeddings from {out_path}.npy")
    return embs, fixation_ids
