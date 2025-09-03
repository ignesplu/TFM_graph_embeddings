import itertools
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import torch

from ...models.HGT_TE import (
    PreproHGTTE,
    HGTTemporalEncoder,
)


def set_seed(seed: int = 1337):
    """
    Set random seeds for reproducibility across libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def tensorize_targets_from_df(
    y_df: pd.DataFrame,
    node_index: Dict[str, int],
    id_col: str = "cc",
    target_cols: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Convert target DataFrame to tensor format aligned with node indices.

    Args:
        y_df: DataFrame containing target values
        node_index: Mapping from node IDs to indices
        id_col: Node identifier column name
        target_cols: Specific target columns to extract

    Returns:
        Tuple of (target tensor, target column names)
    """
    if target_cols is None:
        target_cols = [c for c in y_df.columns if c != id_col]
    N = len(node_index)
    K = len(target_cols)
    Y = torch.zeros((N, K), dtype=torch.float32)

    if id_col in y_df.columns:
        idx_map = y_df.set_index(id_col)
        for cc, nidx in node_index.items():
            if cc in idx_map.index:
                vals = idx_map.loc[cc][target_cols].to_numpy(dtype=np.float32)
                Y[nidx] = torch.tensor(vals)
    else:
        ordered_cc = sorted(node_index.keys())
        assert len(ordered_cc) == len(y_df), "y_df no coincide en longitud con node_index"
        for i, cc in enumerate(ordered_cc):
            Y[i] = torch.tensor(y_df.iloc[i][target_cols].to_numpy(dtype=np.float32))
    return Y, target_cols


def eval_no_train(Z: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate embeddings without training (sanity check metrics).

    Computes basic statistical properties of embeddings for quality assessment.

    Args:
        Z: Embedding tensor

    Returns:
        Dictionary of evaluation metrics
    """
    Z_np = Z.detach().cpu().numpy()
    svals = np.linalg.svd(Z_np, compute_uv=False)
    return {
        "score_main": float(np.mean(np.var(Z_np, axis=0))),
        "z_frob": float(np.linalg.norm(Z_np)),
        "z_rank_est": float(np.sum(svals > 1e-6)),
    }


def eval_linear_probe(
    Z: torch.Tensor,
    y: torch.Tensor,
    cv_folds: int = 5,
    regressor: str = "ridge",
) -> Dict[str, float]:
    """
    Evaluate embeddings using linear probing with cross-validation.

    Trains linear models to predict target variables from embeddings
    and returns RMSE scores.

    Args:
        Z: Embedding tensor
        y: Target tensor
        cv_folds: Number of cross-validation folds
        regressor: Type of linear regressor ('ridge' or 'linreg')

    Returns:
        Dictionary of RMSE metrics
    """

    X = Z.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    N, K = Y.shape

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=123)
    rmses_cols = []

    for k in range(K):
        yk = Y[:, k]
        fold_rmses = []
        for tr, te in kf.split(X):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = yk[tr], yk[te]
            if regressor == "ridge":
                model = RidgeCV(alphas=np.logspace(-3, 3, 13))
            elif regressor == "linreg":
                model = LinearRegression()
            else:
                raise ValueError("regressor debe ser 'ridge' o 'linreg'")
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            rmse = math.sqrt(mean_squared_error(yte, pred))
            fold_rmses.append(rmse)
        rmses_cols.append(np.mean(fold_rmses))

    return {
        "score_main": float(np.mean(rmses_cols)),
        **{f"rmse_{k}": float(v) for k, v in enumerate(rmses_cols)},
    }


@dataclass
class HGTTEGridResult:
    """
    Container for HGT-TE grid search results.

    Attributes:
        results_df: DataFrame with all grid search results
        best_config: Best hyperparameter configuration
        best_metrics: Best evaluation metrics
        best_Z: Best embeddings tensor [N, D]
    """

    results_df: pd.DataFrame
    best_config: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_Z: torch.Tensor


def hgtte_gridsearch(
    *,
    ginputs,
    metadata,
    in_dim: int,
    device: str = "cpu",
    param_grid: Dict[str, List[Any]],
    mode: str = "no_train",
    y_df: Optional[pd.DataFrame] = None,
    y_id_col: str = "cc",
    y_cols: Optional[List[str]] = None,
    eval_fn: Optional[Callable[..., Dict[str, float]]] = None,
    seed: int = 1337,
) -> HGTTEGridResult:
    """
    Perform comprehensive grid search for HGT-TE hyperparameters.

    Evaluates multiple hyperparameter combinations for HGT-TE model
    using specified evaluation mode.

    Args:
        ginputs: Graph inputs container
        metadata: Graph metadata for heterogeneous convolution
        in_dim: Input dimension per time step
        device: Computation device
        param_grid: Hyperparameter grid to search
        mode: Evaluation mode ('no_train', 'linear_probe', or 'custom')
        y_df: Target DataFrame (required for linear_probe mode)
        y_id_col: Node identifier column in target DataFrame
        y_cols: Target columns to evaluate
        eval_fn: Custom evaluation function (required for custom mode)
        seed: Random seed for reproducibility

    Returns:
        HGTTEGridResult container with all results
    """
    assert mode in ("no_train", "linear_probe", "custom")
    if mode == "linear_probe":
        assert y_df is not None, "Debes proporcionar y_df para linear_probe."

    set_seed(seed)

    y_t = None
    if mode == "linear_probe":
        y_t, _ = tensorize_targets_from_df(
            y_df, ginputs.node_index, id_col=y_id_col, target_cols=y_cols
        )

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    rows = []
    best = {"score_main": float("inf")}
    best_Z = None
    best_cfg = None

    i = 1
    for combo in combos:
        cfg = dict(zip(keys, combo))
        torch.cuda.empty_cache()

        # Model
        model = HGTTemporalEncoder(
            input_dim_per_year=in_dim,
            spatial_hidden=cfg.get("spatial_hidden", 128),
            spatial_out=cfg.get("spatial_out", 128),
            heads=cfg.get("heads", 2),
            dropout=cfg.get("dropout", 0.1),
            temporal_layers=cfg.get("temporal_layers", 2),
            temporal_heads=cfg.get("temporal_heads", 4),
            temporal_ff=cfg.get("temporal_ff", 512),
            target_year=cfg.get("target_year", 2022),
            years_sorted=ginputs.years_sorted,
            metadata=metadata,
            temporal_pe_dim=cfg.get("temporal_pe_dim", None),
            lambda_focus=cfg.get("lambda_focus", 0.25),
        ).to(device)
        model.eval()

        t0 = time.time()
        with torch.no_grad():
            Z = model(ginputs.data_per_year)  # [N, D]
        elapsed = time.time() - t0

        # Evaluation
        if mode == "no_train":
            metrics = eval_no_train(Z)
        elif mode == "linear_probe":
            metrics = eval_linear_probe(Z, y=y_t)
        else:
            assert eval_fn is not None, "Debes pasar eval_fn si mode='custom'."
            metrics = eval_fn(Z=Z, ginputs=ginputs, config=cfg)

        row = {**cfg, **metrics, "time_sec": elapsed, "Z_dim": int(Z.size(1))}
        rows.append(row)
        print(f"[{i}/{len(combos)}] {row}")
        i += 1

        # Best model
        if metrics["score_main"] < best["score_main"]:
            best = metrics
            best_Z = Z.detach().cpu().clone()
            best_cfg = cfg

    results_df = pd.DataFrame(rows).sort_values("score_main", ascending=True).reset_index(drop=True)
    return HGTTEGridResult(
        results_df=results_df, best_config=best_cfg, best_metrics=best, best_Z=best_Z
    )


def default_hgtte_param_grid():
    """
    Define default hyperparameter grid for HGT-TE model.

    Returns:
        Dictionary of parameter ranges for grid search
    """
    grid = {
        "spatial_hidden": [64, 128],
        "spatial_out": [64, 128],
        "heads": [2, 4],
        "dropout": [0.1, 0.3],
        "temporal_layers": [1, 2],
        "temporal_heads": [2, 4],
        "temporal_ff": [128, 256],
        "temporal_pe_dim": [None, 64],  # if None => uses spatial_out
        "lambda_focus": [0.1, 0.25, 0.5],
        "target_year": [2022],
    }
    return grid


def build_proxies_df(
    *,
    tabu_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    node_id_col: str = "cc",
    year_col: str = "year",
    target_year: int = 2022,
    static_proxy_cols: Optional[List[str]] = None,
    temp_proxy_cols: Optional[List[str]] = None,
    drop_rows_with_any_na: bool = True,
    fillna_with: Optional[float] = None,
    node_index: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Build proxy target DataFrame for linear probing evaluation.

    Combines static and temporal features into a single DataFrame
    for use as targets in linear probing evaluation.

    Args:
        tabu_df: Tabular data DataFrame
        temp_df: Temporal data DataFrame
        node_id_col: Node identifier column
        year_col: Year column
        target_year: Target year for temporal features
        static_proxy_cols: Static feature columns to include
        temp_proxy_cols: Temporal feature columns to include
        drop_rows_with_any_na: Whether to drop rows with missing values
        fillna_with: Value to fill missing values with
        node_index: Node index mapping for alignment

    Returns:
        DataFrame with proxy targets for evaluation
    """
    base_cc = sorted(map(str, tabu_df[node_id_col].unique()))
    df = pd.DataFrame({node_id_col: base_cc})

    # 1) Static proxies from tabular data
    if static_proxy_cols:
        cols_exist = [c for c in static_proxy_cols if c in tabu_df.columns]
        if len(cols_exist) < len(static_proxy_cols):
            missing = set(static_proxy_cols) - set(cols_exist)
            print(f"[WARN] TABU no tiene columnas estáticas: {missing}")
        df = df.merge(tabu_df[[node_id_col] + cols_exist].copy(), on=node_id_col, how="left")

    # 2) Temporal proxies from tabular data
    if temp_proxy_cols:
        temp_y = temp_df[temp_df[year_col] == target_year].copy()
        cols_exist = [c for c in temp_proxy_cols if c in temp_y.columns]
        if len(cols_exist) < len(temp_proxy_cols):
            missing = set(temp_proxy_cols) - set(cols_exist)
            print(f"[WARN] TEMP({target_year}) no tiene columnas temporales: {missing}")
        temp_y = temp_y[[node_id_col] + cols_exist]
        df = df.merge(temp_y, on=node_id_col, how="left")

    # Drop if NAs
    if drop_rows_with_any_na and fillna_with is None:
        df = df.dropna(axis=0, how="any")
    elif fillna_with is not None:
        df = df.fillna(fillna_with)

    # Order
    if node_index is not None:
        df["_order"] = df[node_id_col].map(node_index)
        df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return df


def run_hgtte_gridsearch(
    device,
    tabu: pd.DataFrame,
    temp: pd.DataFrame,
    mdir: pd.DataFrame,
    mndi: pd.DataFrame,
    add_idea_emb: bool,
    no_mad: bool,
    node_id_col: str = "cc",
    src_col: str = "src",
    dst_col: str = "dst",
    year_col: str = "year",
    static_cols: Optional[List[str]] = None,
    temp_cols: Optional[List[str]] = None,
    edge_attr_cols_dir: Optional[List[str]] = None,
    edge_attr_cols_undir: Optional[List[str]] = None,
    target_year: int = 2022,
    static_proxy_cols: list = ["idea_price_mean", "geo_distancia_capital"],
    temp_proxy_cols: list = ["n_ss_general_por_hab", "n_nacimientos_por_hab"],
    drop_proxy_rows_with_na: bool = True,
    param_grid: dict = None,
    gs_mode: str = "linear_probe",
):
    """
    Complete HGT-TE grid search pipeline with preprocessing and evaluation.

    End-to-end function that handles data preprocessing, proxy target construction,
    and grid search execution for HGT-TE model optimization.

    Args:
        device: Computation device
        tabu: Tabular data
        temp: Temporal data
        mdir: Directed edge data
        mndi: Undirected edge data
        add_idea_emb: Whether to add idea embeddings
        no_mad: Whether to exclude Madrid from data
        node_id_col: Node identifier column
        src_col: Source column for edges
        dst_col: Destination column for edges
        year_col: Year column
        static_cols: Static feature columns
        temp_cols: Temporal feature columns
        edge_attr_cols_dir: Directed edge attribute columns
        edge_attr_cols_undir: Undirected edge attribute columns
        target_year: Target year for prediction
        static_proxy_cols: Static proxy target columns
        temp_proxy_cols: Temporal proxy target columns
        drop_proxy_rows_with_na: Whether to drop rows with missing proxy values
        param_grid: Custom parameter grid
        gs_mode: Grid search evaluation mode

    Returns:
        Tuple of (results DataFrame, best configuration, best metrics, best embeddings)
    """

    p = PreproHGTTE(add_idea_emb=add_idea_emb, no_mad=no_mad)
    ginputs, tabu_hgtte, temp_hgtte, meta, in_dim = p.run(
        tabu=tabu,
        temp=temp,
        mdir=mdir,
        mndi=mndi,
        node_id_col=node_id_col,
        src_col=src_col,
        dst_col=dst_col,
        year_col=year_col,
        static_cols=static_cols,
        temp_cols=temp_cols,
        edge_attr_cols_dir=edge_attr_cols_dir,
        edge_attr_cols_undir=edge_attr_cols_undir,
        device=device,
    )

    proxies_df = build_proxies_df(
        tabu_df=tabu_hgtte,
        temp_df=temp_hgtte,
        node_id_col=node_id_col,
        year_col=year_col,
        target_year=target_year,
        static_proxy_cols=static_proxy_cols,
        temp_proxy_cols=temp_proxy_cols,
        drop_rows_with_any_na=drop_proxy_rows_with_na,
        node_index=ginputs.node_index,
    )

    grid = param_grid if param_grid is not None else default_hgtte_param_grid()
    res = hgtte_gridsearch(
        ginputs=ginputs,
        metadata=meta,
        in_dim=in_dim,
        device=device,
        param_grid=grid,
        mode=gs_mode,
        y_df=proxies_df,
        y_id_col=node_id_col,
        y_cols=[c for c in proxies_df.columns if c != node_id_col],
    )

    return res.results_df, res.best_config, res.best_metrics, res.best_Z
