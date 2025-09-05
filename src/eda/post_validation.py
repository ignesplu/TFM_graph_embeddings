from __future__ import annotations
from typing import Literal, Optional, Tuple
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
