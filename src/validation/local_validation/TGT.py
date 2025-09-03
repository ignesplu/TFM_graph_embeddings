import itertools
import gc
import json
import os
import numpy as np
import pandas as pd

import torch

from ...models.TGT import PreproTGT, TGTConfig, compute_tgt_embeddings


def _grid_keys_order(grid: dict):
    """
    Get ordered list of grid parameter keys.

    Args:
        grid: Parameter grid dictionary

    Returns:
        List of parameter keys in consistent order
    """
    return list(grid.keys())


def _params_key(params: dict, keys_order: list):
    """
    Create unique key for parameter combination.

    Args:
        params: Parameter dictionary
        keys_order: Ordered list of parameter keys

    Returns:
        Tuple representing unique parameter combination
    """
    return tuple(params[k] for k in keys_order)


def _load_completed_param_keys(csv_path: str, keys_order: list):
    """
    Load already completed parameter combinations from CSV.

    Args:
        csv_path: Path to results CSV file
        keys_order: Ordered list of parameter keys

    Returns:
        Set of completed parameter combinations
    """
    if not (csv_path and os.path.exists(csv_path)):
        return set()
    df = pd.read_csv(
        csv_path,
        usecols=[c for c in keys_order if os.path.exists(csv_path)] + ["status"],
    )
    if not all(k in df.columns for k in keys_order) or "status" not in df.columns:
        return set()
    done = df[df["status"] == "ok"]
    return set(tuple(done[k].iloc[i] for k in keys_order) for i in range(len(done)))


def _append_row_to_csv(row_dict: dict, csv_path: str):
    """
    Append a row of results to CSV file.

    Args:
        row_dict: Dictionary of results to append
        csv_path: Path to CSV file
    """
    df_row = pd.DataFrame([row_dict])
    header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, index=False, mode="a", header=header)


def _extract_targets_from_temp(
    temp_df: pd.DataFrame,
    node_index: dict,
    target_year: int,
    target_cols=None,
    cc_col: str = "cc",
    year_col: str = "year",
):
    """
    Extract target variables from temporal data for specific year.

    Args:
        temp_df: Temporal data DataFrame
        node_index: Node ID to index mapping
        target_year: Target year for extraction
        target_cols: Specific target columns to extract
        cc_col: Node identifier column name
        year_col: Year column name

    Returns:
        Dictionary of target arrays indexed by column name
    """
    df = temp_df.copy()
    df[cc_col] = df[cc_col].astype(str)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if year_col not in num_cols:
        num_cols = [c for c in num_cols if c != year_col]
    feature_cols = [c for c in num_cols if c != year_col]
    if target_cols is not None:
        feature_cols = [c for c in feature_cols if c in set(target_cols)]
        if len(feature_cols) == 0:
            raise ValueError("None of the requested target_cols are numeric or present in temp_df.")
    yr = df[df[year_col] == target_year]
    out = {}
    for col in feature_cols:
        sub = yr[[cc_col, col]].dropna()
        if sub.empty:
            continue
        idx, y = [], []
        for _, r in sub.iterrows():
            cc = str(r[cc_col])
            if cc in node_index:
                idx.append(node_index[cc])
                y.append(float(r[col]))
        if len(idx) >= 10:
            out[col] = (
                np.asarray(idx, dtype=np.int64),
                np.asarray(y, dtype=np.float32),
            )
    if not out:
        raise ValueError("No valid targets found for the given year/columns.")
    return out


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """
    Compute ridge regression solution in closed form.

    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength

    Returns:
        Tuple of (weights, RMSE)
    """
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    Xb = np.hstack([X, ones])
    d = Xb.shape[1]
    I = np.eye(d, dtype=X.dtype)
    A = Xb.T @ Xb + alpha * I
    w = np.linalg.pinv(A) @ (Xb.T @ y)
    yhat = Xb @ w
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return w, rmse


def _kfold_rmse(
    Z: np.ndarray,
    idx: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    k: int = 5,
    seed: int = 42,
) -> float:
    """
    Compute k-fold cross-validated RMSE using ridge regression.

    Args:
        Z: Embedding matrix
        idx: Node indices for target values
        y: Target values
        alpha: Ridge regularization parameter
        k: Number of folds
        seed: Random seed for reproducibility

    Returns:
        Mean RMSE across k folds
    """
    n = len(idx)
    k = min(k, n) if n > 1 else 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    rmses = []
    for fi in range(k):
        test_ix = folds[fi]
        train_ix = np.concatenate([folds[j] for j in range(k) if j != fi]) if k > 1 else test_ix
        Xtr, ytr = Z[idx[train_ix]], y[train_ix]
        Xte, yte = Z[idx[test_ix]], y[test_ix]
        w, _ = _ridge_closed_form(Xtr, ytr, alpha=alpha)
        ones_te = np.ones((Xte.shape[0], 1), dtype=Xte.dtype)
        Xteb = np.hstack([Xte, ones_te])
        yhat = Xteb @ w
        rmse = float(np.sqrt(np.mean((yte - yhat) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("inf")


def default_tgt_param_grid():
    """
    Define default hyperparameter grid for Temporal Graph Transformer.

    Returns:
        Dictionary of parameter ranges for grid search
    """
    return {
        "hidden": [64, 96],
        "heads": [2, 4],
        "tf_layers": [1, 2],
        "tf_ff": [128, 256],
        "dropout": [0.1],
        "time_enc_dim": [16, 32],
        "decay": [0.25, 0.35, 0.5],
    }


def _expand_grid(grid: dict):
    """
    Generate all combinations of grid parameters.

    Args:
        grid: Parameter grid dictionary

    Returns:
        List of all parameter combinations
    """
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    out = []
    for c in combos:
        out.append({k: v for k, v in zip(keys, c)})
    return out


def run_tgt_gridsearch(
    tabu: pd.DataFrame,
    temp: pd.DataFrame,
    mdir: pd.DataFrame,
    mndi: pd.DataFrame,
    add_idea_emb: bool = True,
    no_mad: bool = False,
    target_year: int = 2022,
    target_cols=None,
    param_grid=None,
    alpha_ridge: float = 1.0,
    k_folds: int = 5,
    device=None,
    verbose: bool = True,
    csv_path: str = "path/to/tgt/grid/results.csv",
    resume: bool = True,
    save_weights_json: bool = False,
):
    """
    Run comprehensive grid search for Temporal Graph Transformer hyperparameters.

    Performs memory-efficient grid search with resume capability, evaluating
    each parameter combination using k-fold cross-validation with ridge regression.

    Args:
        tabu: Tabular data DataFrame
        temp: Temporal data DataFrame
        mdir: Directed edge data DataFrame
        mndi: Undirected edge data DataFrame
        add_idea_emb: Whether to add idea embeddings
        no_mad: Whether to exclude Madrid from data
        target_year: Target year for prediction
        target_cols: Specific target columns to evaluate
        param_grid: Custom parameter grid (uses default if None)
        alpha_ridge: Ridge regularization parameter
        k_folds: Number of cross-validation folds
        device: Computation device (CPU/GPU)
        verbose: Whether to print progress
        csv_path: Path to save results CSV
        resume: Whether to resume from existing results
        save_weights_json: Whether to save temporal weights (memory intensive)

    Returns:
        DataFrame with grid search results sorted by average RMSE
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = default_tgt_param_grid() if param_grid is None else param_grid
    combos = _expand_grid(grid)
    keys_order = _grid_keys_order(grid)

    completed = _load_completed_param_keys(csv_path, keys_order) if resume else set()
    if verbose and completed:
        print(f"[resume] Saltando {len(completed)} combinaciones ya completadas.")

    # Prepro data
    p = PreproTGT(add_idea_emb=add_idea_emb, no_mad=no_mad)
    prep = p.run(tabu, temp, mdir, mndi)

    # Targets
    targets = _extract_targets_from_temp(temp, prep.node_index, target_year, target_cols)

    total = len(combos)
    for i, params in enumerate(combos, start=1):
        key = _params_key(params, keys_order)
        if key in completed:
            if verbose:
                print(f"[{i}/{total}] SKIP {params}")
            continue

        cfg = TGTConfig(
            hidden=params["hidden"],
            heads=params["heads"],
            tf_layers=params["tf_layers"],
            tf_ff=params["tf_ff"],
            dropout=params["dropout"],
            time_enc_dim=params["time_enc_dim"],
            tf_heads=max(1, min(8, params["heads"])),
        )
        if verbose:
            print(f"[{i}/{total}] Probando: {params}")

        row = {**params}
        try:
            with torch.inference_mode():
                Z, years, w = compute_tgt_embeddings(
                    prep,
                    target_year=target_year,
                    device=device,
                    cfg=cfg,
                    decay=params["decay"],
                )

            Znp = Z.numpy().astype(np.float32, copy=False)

            # RMSE x target
            rmses = []
            for tname, (idx, y) in targets.items():
                y = y.astype(np.float32, copy=False)
                idx = idx.astype(np.int32, copy=False)
                rmse = _kfold_rmse(Znp, idx, y, alpha=alpha_ridge, k=k_folds)
                row[f"rmse_{tname}"] = float(rmse)
                rmses.append(float(rmse))

            row["avg_rmse"] = float(np.mean(rmses)) if rmses else float("inf")
            row["years_json"] = (
                json.dumps(list(map(int, years))) if hasattr(years, "__iter__") else "[]"
            )
            if save_weights_json and hasattr(w, "numpy"):
                row["weights_json"] = json.dumps(w.numpy().astype(np.float32).tolist())
            row["status"] = "ok"
            if verbose:
                print(f"  -> avg_rmse={row['avg_rmse']:.6f}")

        except RuntimeError as e:
            row["avg_rmse"] = float("inf")
            row["error"] = str(e)
            row["status"] = "error"
            if verbose:
                print("  -> Error:", e)

        _append_row_to_csv(row, csv_path)

        # Memory
        for var in ("Z", "Znp", "years", "w", "rmses"):
            if var in locals():
                try:
                    del locals()[var]
                except Exception:
                    pass
        gc.collect()

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "avg_rmse" in df.columns:
            df = df.sort_values("avg_rmse", ascending=True, na_position="last")
        return df
    else:
        return pd.DataFrame()
