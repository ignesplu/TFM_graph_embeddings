from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan
import umap


def _get_embedding_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Extract embedding matrix and municipality IDs from DataFrame.
    
    Validates input data and extracts embedding columns for downstream processing.
    
    Args:
        df: Input DataFrame with embeddings
        
    Returns:
        Tuple of (embedding matrix, municipality IDs)
    """
    if "cc" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'cc'.")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError("No se encontraron columnas de embedding (prefijo 'emb_').")

    ids = df["cc"].astype(str)
    X = df[emb_cols].to_numpy(dtype=np.float32, copy=True)
    if np.isnan(X).any():
        raise ValueError("Los embeddings contienen NaNs. Limpia o imputa antes de continuar.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Hay IDs de municipio duplicados en 'cc'.")
    if X.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 municipios para comparar.")

    return X, ids


def top_cosine_pairs(
    df: pd.DataFrame,
    top_k: Optional[int] = None,
    min_sim: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate cosine similarities between all municipality pairs and return top matches.
    
    Computes pairwise cosine similarities between embeddings and returns
    the most similar pairs, optionally filtered by threshold or limited by count.
    
    Args:
        df: DataFrame with municipality IDs and embeddings
        top_k: Maximum number of pairs to return
        min_sim: Minimum similarity threshold
        
    Returns:
        DataFrame of top similar pairs with cosine similarity scores
    """
    X, ids = _get_embedding_matrix(df)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    S = Xn @ Xn.T
    n = S.shape[0]

    iu, ju = np.triu_indices(n, k=1)
    sims = S[iu, ju]

    if min_sim is not None:
        mask = sims >= float(min_sim)
        iu, ju, sims = iu[mask], ju[mask], sims[mask]

    order = np.argsort(-sims)
    if top_k is not None:
        order = order[:int(top_k)]

    cc_i = ids.to_numpy()[iu[order]]
    cc_j = ids.to_numpy()[ju[order]]
    sim_vals = sims[order]

    return pd.DataFrame({"cc_i": cc_i, "cc_j": cc_j, "cosine_sim": sim_vals})


def cluster_embeddings(
    df: pd.DataFrame,
    method: Literal["kmeans", "hdbscan", "hsdbscan"] = "kmeans",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    scale_for_kmeans: bool = True,
) -> pd.DataFrame:
    """
    Cluster municipalities using embedding similarity with KMeans or HDBSCAN.
    
    Performs clustering analysis on municipality embeddings using either
    centroid-based (KMeans) or density-based (HDBSCAN) clustering algorithms.
    
    Args:
        df: DataFrame with municipality IDs and embeddings
        method: Clustering algorithm ('kmeans' or 'hdbscan')
        n_clusters: Number of clusters for KMeans
        random_state: Random seed for reproducibility
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        metric: Distance metric for clustering
        scale_for_kmeans: Whether to scale data for KMeans
        
    Returns:
        DataFrame with cluster assignments for each municipality
    """
    X, ids = _get_embedding_matrix(df)

    method = method.lower()
    if method == "hsdbscan":
        method = "hdbscan"

    if method == "kmeans":
        if n_clusters is None or n_clusters < 1:
            raise ValueError("Para KMeans debes especificar n_clusters >= 1.")
        X_use = X
        if scale_for_kmeans:
            X_use = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(X_use)

    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=None if min_samples is None else int(min_samples),
            metric=metric,
            core_dist_n_jobs=0,  # 0 = auto
        )
        labels = clusterer.fit_predict(X)
    else:
        raise ValueError("method debe ser 'kmeans' o 'hdbscan'.")

    return pd.DataFrame({"cc": ids.to_numpy(), "cluster": labels})


def umap_2d_plot(
    df: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    annotate: bool = True,
    figsize: Tuple[float, float] = (8.0, 6.0),
):
    """
    Create 2D UMAP projection of embeddings with optional annotations.
    
    Reduces high-dimensional embeddings to 2D using UMAP for visualization
    and creates a scatter plot with municipality labels.
    
    Args:
        df: DataFrame with municipality IDs and embeddings
        n_neighbors: UMAP neighborhood size parameter
        min_dist: UMAP minimum distance parameter
        metric: Distance metric for UMAP
        random_state: Random seed for reproducibility
        annotate: Whether to label points with municipality IDs
        figsize: Figure size for the plot
        
    Returns:
        Tuple of (2D coordinates DataFrame, matplotlib figure, matplotlib axes)
    """
    X, ids = _get_embedding_matrix(df)

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        n_components=2,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    Z = reducer.fit_transform(X)

    df_2d = pd.DataFrame({"cc": ids.to_numpy(), "x": Z[:, 0], "y": Z[:, 1]})

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df_2d["x"], df_2d["y"])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Municipios en 2D (UMAP)")

    if annotate:
        for _, row in df_2d.iterrows():
            ax.annotate(str(row["cc"]), (row["x"], row["y"]), xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    return df_2d, fig, ax


def cook_distance_latent(
    df_full: pd.DataFrame,
    df_drop: pd.DataFrame,
    removed_cc: int,
    emb_prefix: str = "emb_",
    align: bool = True,
    metric_cosine_matrix: bool = True,
) -> Dict[str, Any]:
    """Calculate Cook's distance in latent space to measure node removal impact.
    
    Quantifies how much the embeddings of remaining nodes change when a specific
    municipality is removed from the training data. Provides both vector displacement
    and cosine similarity changes with optional Procrustes alignment.
    
    Args:
        df_full: DataFrame with embeddings trained on ALL municipalities
        df_drop: DataFrame with embeddings trained WITHOUT the removed municipality
        removed_cc: Identifier of the municipality removed in df_drop
        emb_prefix: Prefix for embedding columns (default: 'emb_')
        align: Whether to apply orthogonal alignment (Procrustes) to account for rotation differences
        metric_cosine_matrix: Whether to compute changes in cosine similarity matrix
        
    Returns:
        Dictionary containing:
            - meta: Process information (common nodes, dimensions, alignment details)
            - global: Global metrics (mean vector displacement, cosine changes)
            - per_node: DataFrame with deltas per node (L2 shifts, cosine changes)
            - top_affected: Top nodes most affected by removal
            - interpretation: Brief textual interpretation of impact magnitude
    """
    emb_cols = [c for c in df_full.columns if c.startswith(emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No se encontraron columnas de embedding con prefijo '{emb_prefix}'.")

    if removed_cc in set(df_drop["cc"]):
        raise ValueError(f"'{removed_cc}' aparece en df_drop; debería estar eliminado en ese DataFrame.")

    df_full_cmp = df_full[df_full["cc"] != removed_cc].copy()

    common = sorted(set(df_full_cmp["cc"]).intersection(set(df_drop["cc"])))
    if len(common) < 3:
        raise ValueError("Muy pocos cc comunes para comparar (se requieren >= 3).")

    F = df_full_cmp.set_index("cc").loc[common, emb_cols].to_numpy()
    D = df_drop.set_index("cc").loc[common, emb_cols].to_numpy()
    n, d = F.shape

    R = None
    aligned = False
    cond_number = None
    if align:
        Fc = F - F.mean(axis=0, keepdims=True)
        Dc = D - D.mean(axis=0, keepdims=True)

        M = Dc.T @ Fc
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
        F_aligned = (F - F.mean(axis=0, keepdims=True)) @ R + D.mean(axis=0, keepdims=True)

        cond_number = (S.max() / S.min()) if S.min() > 0 else np.inf
        F = F_aligned
        aligned = True

    vec_delta = F - D
    l2_shift = np.linalg.norm(vec_delta, axis=1)
    mean_l2 = float(l2_shift.mean())
    median_l2 = float(np.median(l2_shift))

    center = D.mean(axis=0, keepdims=True)
    typical_scale = float(np.mean(np.linalg.norm(D - center, axis=1))) + 1e-12
    rel_mean_l2 = float(mean_l2 / typical_scale)

    def _cosine_matrix(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return Xn @ Xn.T

    mean_abs_delta_cos = np.nan
    median_abs_delta_cos = np.nan
    mean_abs_delta_cos_per_node = np.full(n, np.nan)

    if metric_cosine_matrix:
        S_full = _cosine_matrix(F)
        S_drop = _cosine_matrix(D)
        triu_idx = np.triu_indices(n, k=1)
        abs_diff = np.abs(S_full[triu_idx] - S_drop[triu_idx])
        mean_abs_delta_cos = float(abs_diff.mean())
        median_abs_delta_cos = float(np.median(abs_diff))
        mean_abs_delta_cos_per_node = np.mean(np.abs(S_full - S_drop), axis=1)

    per_node = pd.DataFrame({
        "cc": common,
        "l2_shift": l2_shift,
        "mean_abs_delta_cosine": mean_abs_delta_cos_per_node,
    }).sort_values("l2_shift", ascending=False).reset_index(drop=True)

    top_by_l2 = per_node.nlargest(10, "l2_shift")[["cc", "l2_shift"]].values.tolist()
    if metric_cosine_matrix:
        top_by_cos = per_node.nlargest(10, "mean_abs_delta_cosine")[["cc", "mean_abs_delta_cosine"]].values.tolist()
    else:
        top_by_cos = []

    if rel_mean_l2 < 0.05:
        lvl_vec = "muy bajo"
    elif rel_mean_l2 < 0.15:
        lvl_vec = "bajo-moderado"
    elif rel_mean_l2 < 0.30:
        lvl_vec = "moderado"
    else:
        lvl_vec = "alto"

    if metric_cosine_matrix and not np.isnan(mean_abs_delta_cos):
        if mean_abs_delta_cos < 0.02:
            lvl_cos = "bajo"
        elif mean_abs_delta_cos < 0.05:
            lvl_cos = "moderado"
        else:
            lvl_cos = "alto"
        cos_text = f"Cambio medio de similitud coseno {mean_abs_delta_cos:.3f} ({lvl_cos}). "
    else:
        cos_text = ""

    interp = (
        f"Impacto global del eliminado {removed_cc}: desplazamiento medio relativo {rel_mean_l2:.3f} ({lvl_vec}). "
        + cos_text
        + "Nodos en 'top_affected' indican dónde se concentra el efecto."
    )

    return {
        "meta": {
            "removed_cc": removed_cc,
            "n_common": n,
            "dim": d,
            "aligned": aligned,
            "alignment_cond_number": cond_number,
            "emb_cols": emb_cols[:5] + (["..."] if len(emb_cols) > 5 else []),
        },
        "global": {
            "mean_l2_shift": mean_l2,
            "median_l2_shift": median_l2,
            "rel_mean_l2_shift": rel_mean_l2,
            "mean_abs_delta_cosine": mean_abs_delta_cos,
            "median_abs_delta_cosine": median_abs_delta_cos,
        },
        "per_node": per_node,
        "top_affected": {
            "by_l2_shift": top_by_l2,
            "by_cosine_change": top_by_cos
        },
        "interpretation": interp,
    }
