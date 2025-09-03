import itertools
import gc
import json
import os
import numpy as np
import pandas as pd

import torch

from ...models.TGT import PreproTGT, TGTConfig, compute_tgt_embeddings


def _grid_keys_order(grid: dict):
    return list(grid.keys())


def _params_key(params: dict, keys_order: list):
    return tuple(params[k] for k in keys_order)


def _load_completed_param_keys(csv_path: str, keys_order: list):
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
    df = temp_df.copy()
    df[cc_col] = df[cc_col].astype(str)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if year_col not in num_cols:
        num_cols = [c for c in num_cols if c != year_col]
    feature_cols = [c for c in num_cols if c != year_col]
    if target_cols is not None:
        feature_cols = [c for c in feature_cols if c in set(target_cols)]
        if len(feature_cols) == 0:
            raise ValueError(
                "None of the requested target_cols are numeric or present in temp_df."
            )
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
    n = len(idx)
    k = min(k, n) if n > 1 else 1
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    rmses = []
    for fi in range(k):
        test_ix = folds[fi]
        train_ix = (
            np.concatenate([folds[j] for j in range(k) if j != fi])
            if k > 1
            else test_ix
        )
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
    Igual que antes pero:
      - No acumula resultados en RAM.
      - Reanuda solo con las claves de hiperparámetros ya 'ok'.
      - Limpia memoria tras cada iteración.
    Devuelve un DataFrame *ligero* leído del CSV ya ordenado (si existe).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = default_tgt_param_grid() if param_grid is None else param_grid
    combos = _expand_grid(grid)
    keys_order = _grid_keys_order(grid)

    completed = _load_completed_param_keys(csv_path, keys_order) if resume else set()
    if verbose and completed:
        print(f"[resume] Saltando {len(completed)} combinaciones ya completadas.")

    # Preprocesado una vez
    p = PreproTGT(add_idea_emb=add_idea_emb, no_mad=no_mad)
    prep = p.run(tabu, temp, mdir, mndi)

    # Targets una vez (usa dtypes ligeros)
    targets = _extract_targets_from_temp(
        temp, prep.node_index, target_year, target_cols
    )

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
            # asegúrate de no construir grafos de gradiente
            with torch.inference_mode():
                Z, years, w = compute_tgt_embeddings(
                    prep,
                    target_year=target_year,
                    device=device,
                    cfg=cfg,
                    decay=params["decay"],
                )

            # baja a float32 para ahorrar RAM
            Znp = Z.numpy().astype(np.float32, copy=False)

            # RMSE por target
            rmses = []
            for tname, (idx, y) in targets.items():
                # dtypes ligeros
                y = y.astype(np.float32, copy=False)
                idx = idx.astype(np.int32, copy=False)
                rmse = _kfold_rmse(Znp, idx, y, alpha=alpha_ridge, k=k_folds)
                row[f"rmse_{tname}"] = float(rmse)
                rmses.append(float(rmse))

            row["avg_rmse"] = float(np.mean(rmses)) if rmses else float("inf")
            row["years_json"] = (
                json.dumps(list(map(int, years)))
                if hasattr(years, "__iter__")
                else "[]"
            )
            if save_weights_json and hasattr(w, "numpy"):
                # OJO: puede ser grande; por eso lo desactivamos por defecto
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

        # --- liberar memoria agresivamente ---
        for var in ("Z", "Znp", "years", "w", "rmses"):
            if var in locals():
                try:
                    del locals()[var]
                except Exception:
                    pass
        gc.collect()

    # devolver algo ligero (si quieres)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "avg_rmse" in df.columns:
            df = df.sort_values("avg_rmse", ascending=True, na_position="last")
        return df
    else:
        return pd.DataFrame()
