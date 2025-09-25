import re
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict, Any
from sklearn.neighbors import NearestNeighbors
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import scikit_posthocs as sp


def discover_models(df, mean_suffix="_mean", low_suffix="_ci_low", high_suffix="_ci_high"):
    """
    Discover model names from performance metric columns with suffix patterns.

    Identifies model names by finding columns ending with mean suffix and validates
    the existence of corresponding confidence interval columns.

    Args:
        df: Input DataFrame with performance metrics
        mean_suffix: Suffix for mean performance columns
        low_suffix: Suffix for lower confidence interval columns
        high_suffix: Suffix for upper confidence interval columns

    Returns:
        List of discovered model names
    """
    mean_cols = [c for c in df.columns if c.endswith(mean_suffix) and c != "target__"]
    models = [re.sub(f"{re.escape(mean_suffix)}$", "", c) for c in mean_cols]

    missing = []
    for m in models:
        for s in (low_suffix, high_suffix):
            col = f"{m}{s}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        print("[AVISO] Faltan columnas de IC para:", missing)
    return models


def build_matrix(df, models, mean_suffix="_mean"):
    """
    Build Nxk performance matrix from validation results.

    Creates a matrix with validation tasks as rows and models as columns,
    using mean performance metrics and handling missing values.

    Args:
        df: Input DataFrame with validation results
        models: List of model names to include
        mean_suffix: Suffix for mean performance columns

    Returns:
        Performance matrix with validation tasks as index
    """
    cols = [f"{m}{mean_suffix}" for m in models]
    mat = df[["target__"] + cols].copy()
    mat = mat.set_index("target__")
    mat.columns = models
    mat = mat.dropna(how="any")
    return mat


def friedman_with_ranks(mat):
    """
    Perform Friedman test and calculate performance ranks.

    Applies Friedman non-parametric test for multiple comparisons and computes
    performance ranks for each validation task and mean ranks across tasks.

    Args:
        mat: Performance matrix (validations × models)

    Returns:
        Tuple of (Friedman statistic, p-value, per-task ranks, mean ranks)
    """
    arrays_por_modelo = [mat[col].values for col in mat.columns]
    stat, p = friedmanchisquare(*arrays_por_modelo)

    ranks = mat.apply(
        lambda row: rankdata(-row.values, method="average"), axis=1, result_type="expand"
    )
    ranks.columns = mat.columns
    mean_ranks = ranks.mean(axis=0).sort_values()

    return stat, p, ranks, mean_ranks


def critical_difference(k, N, q_alpha=3.314):
    """
    Calculate critical difference for Nemenyi post-hoc test.

    Computes the minimum rank difference required for statistical significance
    in the Nemenyi post-hoc test following Friedman analysis.

    Args:
        k: Number of models being compared
        N: Number of validation tasks/datasets
        q_alpha: Critical value for significance level

    Returns:
        Critical difference value
    """
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))


def nemenyi_posthoc(mat):
    """
    Perform Nemenyi post-hoc test after Friedman analysis.

    Computes pairwise p-values using the Nemenyi test to identify
    statistically significant differences between model pairs.

    Args:
        mat: Performance matrix (validations x models)

    Returns:
        DataFrame of pairwise p-values
    """
    ph = sp.posthoc_nemenyi_friedman(mat.values)
    ph.index = mat.columns
    ph.columns = mat.columns
    return ph


def pairwise_wilcoxon_holm(mat, alpha=0.05, zero_method="wilcox", correction="holm"):
    """
    Perform pairwise Wilcoxon signed-rank tests with Holm correction.

    Conducts pairwise comparisons between models using Wilcoxon test
    and applies Holm-Bonferroni correction for multiple comparisons.

    Args:
        mat: Performance matrix (validations x models)
        alpha: Significance level
        zero_method: Method for handling zero differences
        correction: Multiple comparison correction method

    Returns:
        DataFrame of adjusted pairwise p-values
    """
    models = list(mat.columns)
    k = len(models)
    raw = pd.DataFrame(np.ones((k, k)), index=models, columns=models)

    for i in range(k):
        for j in range(i + 1, k):
            a = mat.iloc[:, i].values
            b = mat.iloc[:, j].values
            try:
                _, p = wilcoxon(
                    a, b, zero_method=zero_method, alternative="two-sided", method="auto"
                )
            except ValueError:
                p = 1.0
            raw.iloc[i, j] = p
            raw.iloc[j, i] = p

    if correction.lower() == "holm":
        tril = raw.where(np.triu(np.ones_like(raw, dtype=bool), 1))
        pvals = tril.stack()
        m = len(pvals)
        adj = pvals.copy()
        ranks = pvals.rank(method="first")
        for idx, p in pvals.sort_values().items():
            r = ranks.loc[idx]
            adj_val = (m - r + 1) * p
            adj.loc[idx] = adj_val

        adj = adj.groupby(level=0).cummax()

        adj_mat = pd.DataFrame(np.ones((len(models), len(models))), index=models, columns=models)
        for (i, j), p in adj.items():
            adj_mat.loc[i, j] = p
            adj_mat.loc[j, i] = p
        np.fill_diagonal(adj_mat.values, 0.0)
        return adj_mat.clip(upper=1.0)
    else:
        return raw


def bootstrap_rank_probabilities(
    df,
    models,
    B=2000,
    mean_suffix="_mean",
    low_suffix="_ci_low",
    high_suffix="_ci_high",
    clip01=True,
    random_state=42,
):
    """
    Estimate rank probabilities using bootstrap sampling with confidence intervals.

    Uses performance metric confidence intervals to simulate bootstrap samples
    and estimate the probability of each model achieving the best rank.

    Args:
        df: Input DataFrame with performance metrics and confidence intervals
        models: List of model names to analyze
        B: Number of bootstrap samples
        mean_suffix: Suffix for mean performance columns
        low_suffix: Suffix for lower confidence interval columns
        high_suffix: Suffix for upper confidence interval columns
        clip01: Whether to clip performance values to [0,1] range
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (rank summary statistics, probability of being best)
    """
    rng = np.random.default_rng(random_state)
    M, S = {}, {}
    for m in models:
        mu = df[f"{m}{mean_suffix}"].values.astype(float)
        lo = df.get(f"{m}{low_suffix}", pd.Series([np.nan] * len(df))).values.astype(float)
        hi = df.get(f"{m}{high_suffix}", pd.Series([np.nan] * len(df))).values.astype(float)
        se = (hi - lo) / 3.92  # IC95% ≈ ±1.96*SE  => ancho ≈ 3.92*SE
        M[m] = mu
        S[m] = se

    valid_mask = np.ones(len(df), dtype=bool)
    for m in models:
        valid_mask &= np.isfinite(M[m]) & np.isfinite(S[m])
    idx = np.where(valid_mask)[0]
    if len(idx) == 0:
        raise ValueError("No hay filas válidas con CI completos para bootstrap.")

    mat_ranks_mean = np.zeros((B, len(models)))
    winners = np.zeros(B, dtype=int)

    for b in range(B):
        sims = []
        for m in models:
            mu = M[m][idx]
            se = S[m][idx]
            se = np.where(se <= 0, 1e-6, se)
            draw = rng.normal(mu, se)
            if clip01:
                draw = np.clip(draw, 0.0, 1.0)
            sims.append(draw)
        sim_mat = np.column_stack(sims)  # N×k
        ranks = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, sim_mat)
        mean_ranks = ranks.mean(axis=0)
        mat_ranks_mean[b, :] = mean_ranks
        winners[b] = np.argmin(mean_ranks)

    mean_ranks_boot = pd.DataFrame(mat_ranks_mean, columns=models)
    summary = pd.DataFrame(
        {
            "rank_mean": mean_ranks_boot.mean(0),
            "rank_p2.5": mean_ranks_boot.quantile(0.025, 0),
            "rank_p97.5": mean_ranks_boot.quantile(0.975, 0),
        }
    ).sort_values("rank_mean")

    prob_best = pd.Series(
        np.bincount(winners, minlength=len(models)) / B, index=models
    ).sort_values(ascending=False)
    return summary, prob_best


def statistical_analysis(df, usar_wilcoxon=True, hacer_bootstrap=False, B=2000, alpha=0.05):
    """
    Comprehensive statistical analysis of model performance results.

    Performs complete statistical comparison pipeline including Friedman test,
    Nemenyi post-hoc analysis, optional Wilcoxon tests, and bootstrap ranking.

    Args:
        df: DataFrame with model performance results
        usar_wilcoxon: Whether to perform Wilcoxon pairwise tests
        hacer_bootstrap: Whether to perform bootstrap analysis
        B: Number of bootstrap samples
        alpha: Significance level

    Returns:
        Dictionary containing all analysis results
    """
    # 1) Descubrir modelos y construir matriz
    models = discover_models(df)
    mat = build_matrix(df, models)

    print(f"Modelos detectados ({len(models)}): {models}")
    print(f"Validaciones usadas (filas tras dropna): {mat.shape[0]}")

    # 2) Friedman + rangos
    stat, p, ranks, mean_ranks = friedman_with_ranks(mat)
    print("\n=== Test de Friedman ===")
    print(f"Estadístico: {stat:.4f} | p-value: {p:.6f}")

    print("\n=== Ranks medios (menor es mejor) ===")
    print(mean_ranks)

    # 3) Nemenyi
    nemenyi = nemenyi_posthoc(mat)
    print("\n=== Post-hoc Nemenyi (p-values) ===")
    print(nemenyi)

    k = mat.shape[1]
    N = mat.shape[0]
    CD = critical_difference(k, N, q_alpha=3.314)
    print(f"\nDiferencia Crítica (aprox, α=0.05): {CD:.3f}")
    print(
        "Sugerencia: si la diferencia de ranks medios entre dos modelos supera la CD, se consideran diferentes."
    )

    # 4) Wilcoxon + Holm
    if usar_wilcoxon:
        wilcoxon_holm = pairwise_wilcoxon_holm(mat, alpha=alpha, correction="holm")
        print("\n=== Wilcoxon pareado + Holm (p-values ajustados) ===")
        print(wilcoxon_holm)

    # 5) Bootstrap con IC
    if hacer_bootstrap:
        summary, prob_best = bootstrap_rank_probabilities(df, models, B=B)
        print("\n=== Bootstrap de ranks medios (con IC95% por modelo) ===")
        print(summary)
        print("\nProbabilidad de ser el mejor (por rank medio):")
        print(prob_best)

    return {
        "mat": mat,
        "friedman_p": p,
        "mean_ranks": mean_ranks,
        "nemenyi_pvals": nemenyi,
        "CD": CD,
        "wilcoxon_holm": wilcoxon_holm if usar_wilcoxon else None,
    }


def knn_category_agreement(
    df: pd.DataFrame,
    label_cols: List[str],
    *,
    id_col: str = "cc",
    emb_prefix: str = "emb_",
    k: int = 10,
    metric: str = "cosine",
    quantiles: Optional[Union[int, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Calculate K-nearest neighbors category agreement for embedding validation.

    For each node and each target variable, finds K nearest neighbors by embedding
    similarity and measures how many neighbors share the same category/quantile.
    Useful for validating that similar embeddings correspond to similar attributes.

    Args:
        df: DataFrame containing embeddings and target variables
        label_cols: List of target variable names to evaluate
        id_col: Node identifier column name
        emb_prefix: Prefix for embedding columns
        k: Number of nearest neighbors to consider
        metric: Distance metric for neighbor search
        quantiles: Number of quantiles for discretizing numeric variables.
                  Can be integer (for all variables) or dict (per-variable)

    Returns:
        Dictionary containing:
            - per_cc_col: Detailed results per node and variable
            - by_column: Summary statistics per variable
            - by_node: Summary statistics per node
            - details: Metadata about the analysis configuration
    """
    # 1) Embeddings y validaciones
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]
    if not emb_cols:
        raise ValueError(f"No se encontraron columnas que empiecen por '{emb_prefix}'.")
    if id_col not in df.columns:
        raise ValueError(f"No se encuentra id_col='{id_col}' en df.")
    for col in label_cols:
        if col not in df.columns:
            raise ValueError(f"La columna objetivo '{col}' no existe en df.")

    df = df.copy()
    X = df[emb_cols].to_numpy()
    ids = df[id_col].to_numpy()
    n_samples = len(df)
    n_neighbors = min(k + 1, n_samples)

    # 2) KNN
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(X)
    knn_idx = nn.kneighbors(return_distance=False)

    # 3) Preparar variables categorizadas
    def _q_for(col: str) -> Optional[int]:
        if quantiles is None:
            return None
        if isinstance(quantiles, int):
            return quantiles
        if isinstance(quantiles, dict):
            return quantiles.get(col, None)
        return None

    cat_data: Dict[str, pd.Series] = {}
    bin_info: Dict[str, Any] = {}

    for col in label_cols:
        s = df[col]
        q = _q_for(col)
        if q is not None and q >= 2 and pd.api.types.is_numeric_dtype(s):
            labels = [f"Q{i+1}" for i in range(q)]
            s_cat = pd.qcut(s, q=q, labels=labels, duplicates="drop")

            try:
                _, edges = pd.qcut(s, q=q, retbins=True, duplicates="drop")
                edges = list(map(float, edges))
            except Exception:
                edges = None
            cat_data[col] = s_cat.astype("object")
            bin_info[col] = {
                "type": "quantile",
                "requested_q": q,
                "effective_labels": list(s_cat.cat.categories) if hasattr(s_cat, "cat") else None,
                "edges": edges,
            }
        else:
            cat_data[col] = s.astype("object")
            bin_info[col] = {
                "type": "as_is",
                "requested_q": None,
                "effective_labels": None,
                "edges": None,
            }

    # 4) Scoring por cc y columna
    rows = []
    for i in range(n_samples):
        neigh = [j for j in knn_idx[i] if j != i][:k]

        for col in label_cols:
            target = cat_data[col].iloc[i]
            if pd.isna(target):
                rows.append(
                    {
                        id_col: ids[i],
                        "column": col,
                        "category": np.nan,
                        "matches": np.nan,
                        "neighbors_used": 0,
                        "proportion": np.nan,
                    }
                )
                continue

            neigh_labels = cat_data[col].iloc[neigh]
            valid_mask = ~neigh_labels.isna()
            valid_labels = neigh_labels[valid_mask]

            matches = int((valid_labels == target).sum())
            neighbors_used = int(valid_mask.sum())
            proportion = (matches / neighbors_used) if neighbors_used > 0 else np.nan

            rows.append(
                {
                    id_col: ids[i],
                    "column": col,
                    "category": target,
                    "matches": matches,
                    "neighbors_used": neighbors_used,
                    "proportion": proportion,
                }
            )

    per_cc_col = pd.DataFrame(rows)
    per_cc_col_valid = per_cc_col.dropna(subset=["proportion"]).copy()
    if per_cc_col_valid.empty:
        raise ValueError("No hay proporciones válidas (todas las etiquetas objetivo son NaN).")

    # 5) Resúmenes
    by_column = (
        per_cc_col_valid.groupby("column")
        .agg(
            mean_proportion=("proportion", "mean"),
            mean_matches=("matches", "mean"),
            n=("proportion", "size"),
        )
        .reset_index()
        .sort_values("mean_proportion", ascending=False)
        .reset_index(drop=True)
    )

    by_node = (
        per_cc_col_valid.groupby(id_col)["proportion"]
        .mean()
        .rename("mean_proportion_across_columns")
        .reset_index()
        .sort_values("mean_proportion_across_columns", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "per_cc_col": per_cc_col_valid.sort_values(["column", id_col]).reset_index(drop=True),
        "by_column": by_column,
        "by_node": by_node,
        "details": {
            "k": k,
            "metric": metric,
            "emb_cols": emb_cols[:5] + (["..."] if len(emb_cols) > 5 else []),
            "quantiles": quantiles,
            "binning_info": bin_info,
            "note": "La proporción es #coincidencias / #vecinos con etiqueta válida para esa columna.",
        },
    }
