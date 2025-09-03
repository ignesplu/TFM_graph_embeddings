import os
import random
import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

import torch

from ...models.LINE import (
    PreproLINE,
    full_train_line,
)


def validation_emb(
    raw_df: pd.DataFrame,
    target_col: str,
    emb_df: pd.DataFrame,
    pref_X: str = "emb_",
    node_id_col: str = "cc",
):
    """
    Validate embeddings by training multiple regression models and evaluating their performance.

    This function performs a comprehensive validation of node embeddings by training
    four different regression models (Linear Regression, XGBoost, Random Forest, and SVR)
    and calculating their RMSE scores on a test set.

    Args:
        raw_df: Original DataFrame containing target values and node identifiers
        target_col: Name of the target column to predict
        emb_df: DataFrame containing the node embeddings
        pref_X: Prefix for embedding column names (default: "emb_")
        node_id_col: Name of the node identifier column (default: "cc")

    Returns:
        tuple: RMSE scores for (Linear Regression, XGBoost, Random Forest, SVR)
    """
    SEED = 33
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)

    data = emb_df.merge(raw_df[[node_id_col, target_col]], on=node_id_col, how="inner")
    data = data.sort_values(by=node_id_col).reset_index(drop=True)

    X = data[[c for c in data.columns if c.startswith(pref_X)]].copy()
    y = data[target_col].copy()

    X = X.reindex(sorted(X.columns), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True, stratify=None
    )

    # 1) LineaRegression
    linreg = LinearRegression(n_jobs=None)
    linreg.fit(X_train, y_train)
    y_pred_lr = linreg.predict(X_test)
    lr_rmse = mean_squared_error(y_test, y_pred_lr)

    # 2) XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_rmse = mean_squared_error(y_test, y_pred_xgb)

    # 3) RandomForest
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=SEED,
        n_jobs=1,
        bootstrap=True,
        oob_score=False,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_rmse = mean_squared_error(y_test, y_pred_rf)

    # 4) SVR
    svr = SVR(
        kernel="rbf",
        C=10,
        gamma="scale",
    )
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    svr_rmse = mean_squared_error(y_test, y_pred_svr)

    return lr_rmse, xgb_rmse, rf_rmse, svr_rmse


def expand_grid(param_dict: dict):
    """
    Generate all possible combinations of hyperparameters from a parameter dictionary.

    Creates a comprehensive grid of hyperparameter combinations for grid search
    by computing the Cartesian product of all parameter values.

    Args:
        param_dict: Dictionary where keys are parameter names and values are lists of possible values

    Returns:
        list: List of dictionaries, each representing a unique parameter combination
    """
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in product(*values)]
    return combos


def gridsearch_params(
    device,
    tabu: pd.DataFrame,
    temp: pd.DataFrame,
    mdir: pd.DataFrame,
    mndi: pd.DataFrame,
    gs_dict: dict = None,
    validation_cols: list = [
        "eur_gastos_mean",
        "idea_price_mean",
        "idea_size_mean",
        "n_migr_inter_mean",
    ],
    n_loops: int = 5,
    add_idea_emb: bool = True,
    no_mad: bool = False,
) -> pd.DataFrame:
    """
    Perform comprehensive grid search for LINE model hyperparameter optimization.

    This function executes a full grid search across multiple hyperparameter combinations,
    trains LINE models with different configurations, and evaluates their performance
    using multiple regression models and validation metrics. It aggregates results
    and calculates weighted rankings to identify optimal parameter settings.

    Note:
        In weighted rankings, XGB is given slightly more weight as it is the best predictor
        in most cases. On the other hand, RF is given more weight than SVR as RF tends to
        generalise better in graph/tabular embeddings and is more consistent. Finally, LR
        is kept at 0.15 because it makes sense to keep it as a signal of linearity, but
        without distorting it too much.

    Args:
        device: Torch device (CPU/GPU) for model training
        tabu: Tabular data DataFrame
        temp: Temporal data DataFrame
        mdir: Direct migration data DataFrame
        mndi: Indirect migration data DataFrame
        gs_dict: Dictionary of hyperparameters to search (uses default if None)
        validation_cols: List of target columns for validation
        n_loops: Number of training loops for each configuration
        add_idea_emb: Whether to add idea embeddings
        no_mad: Whether to exclude Madrid from data

    Returns:
        pd.DataFrame: Comprehensive results DataFrame sorted by overall ranking,
                      containing RMSE scores and rankings for all configurations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input Data
    prepro = PreproLINE(add_idea_emb=add_idea_emb, no_mad=no_mad, year=2022)
    df_line_mms_mean, df_line_mms_sum, df_line_r_mean, df_line_r_sum = prepro.run(
        tabu, temp, mdir, mndi
    )
    model_values = {
        "LINE-MMS-Mean": df_line_mms_mean,
        "LINE-MMS-Sum": df_line_mms_sum,
        "LINE-R-Mean": df_line_r_mean,
        "LINE-R-Sum": df_line_r_sum,
    }

    # Params
    if not gs_dict:
        gs_dict = {
            "emb_dim": [128, 256],
            "n_epochs": [100],
            "batch_size": [10000],
            "neg": [5, 10],
            "lr": [0.005, 0.01, 0.025],
        }
    all_combos = expand_grid(gs_dict)

    val_list = []
    for i in range(n_loops):
        for name, df_m in model_values.items():
            for output_col in validation_cols:
                for params in all_combos:
                    print(i, "-", name, "-", output_col, "-", params)

                    # Train
                    emb_df = full_train_line(device=device, df=df_m.copy()).rename(
                        columns={"node_id": "cc"}
                    )
                    # Validation Metrics
                    lr_rmse, xgb_rmse, rf_rmse, svr_rmse = validation_emb(
                        tabu,
                        output_col,
                        emb_df,
                        pref_X="emb_",
                        node_id_col="cc",
                    )

                    val_list.append(
                        [
                            name,
                            output_col,
                            params["emb_dim"],
                            params["n_epochs"],
                            params["batch_size"],
                            params["neg"],
                            params["lr"],
                            lr_rmse,
                            xgb_rmse,
                            rf_rmse,
                            svr_rmse,
                        ]
                    )

    df_val = pd.DataFrame(
        val_list,
        columns=[
            "name",
            "output_col",
            "emb_dim",
            "n_epochs",
            "batch_size",
            "neg",
            "lr",
            "lr_rmse",
            "xgb_rmse",
            "rf_rmse",
            "svr_rmse",
        ],
    )
    df_val = (
        df_val.groupby(
            [
                "name",
                "output_col",
                "emb_dim",
                "n_epochs",
                "batch_size",
                "neg",
                "lr",
            ]
        )
        .mean()
        .reset_index()
    )

    eur_df = (
        df_val[df_val.output_col == "eur_gastos_mean"]
        .rename(
            columns={
                "lr_rmse": "eur_lr",
                "xgb_rmse": "eur_xgb",
                "rf_rmse": "eur_rf",
                "svr_rmse": "eur_svr",
            }
        )
        .drop(columns=["output_col"], axis=1)
    )
    price_df = (
        df_val[df_val.output_col == "idea_price_mean"]
        .rename(
            columns={
                "lr_rmse": "price_lr",
                "xgb_rmse": "price_xgb",
                "rf_rmse": "price_rf",
                "svr_rmse": "price_svr",
            }
        )
        .drop(columns=["output_col"], axis=1)
    )
    size_df = (
        df_val[df_val.output_col == "idea_size_mean"]
        .rename(
            columns={
                "lr_rmse": "size_lr",
                "xgb_rmse": "size_xgb",
                "rf_rmse": "size_rf",
                "svr_rmse": "size_svr",
            }
        )
        .drop(columns=["output_col"], axis=1)
    )
    migr_df = (
        df_val[df_val.output_col == "n_migr_inter_mean"]
        .rename(
            columns={
                "lr_rmse": "migr_lr",
                "xgb_rmse": "migr_xgb",
                "rf_rmse": "migr_rf",
                "svr_rmse": "migr_svr",
            }
        )
        .drop(columns=["output_col"], axis=1)
    )

    idx_cols = ["name", "emb_dim", "n_epochs", "batch_size", "neg", "lr"]
    val_final = (
        eur_df.set_index(idx_cols)
        .join(price_df.set_index(idx_cols))
        .join(size_df.set_index(idx_cols))
        .join(migr_df.set_index(idx_cols))
        .reset_index()
    )

    list_rmse = [
        "eur_lr",
        "eur_xgb",
        "eur_rf",
        "eur_svr",
        "price_lr",
        "price_xgb",
        "price_rf",
        "price_svr",
        "size_lr",
        "size_xgb",
        "size_rf",
        "size_svr",
        "migr_lr",
        "migr_xgb",
        "migr_rf",
        "migr_svr",
    ]
    # Ranking Metrics
    for rmse in list_rmse:
        val_final[f"rank_{rmse}"] = val_final[rmse].rank(method="average", ascending=True)

    # Rank Mean
    val_final["rank_mean"] = val_final[
        [col for col in val_final.columns if col.startswith("rank_")]
    ].mean(axis=1)

    weights = {"_xgb": 0.40, "_rf": 0.25, "_svr": 0.20, "_lr": 0.15}

    def weighted_mean(row):
        total = 0
        for suf, w in weights.items():
            cols = [c for c in row.index if c.startswith("rank_") and c.endswith(suf)]
            if len(cols) > 0:
                total += row[cols].mean() * w
        return total

    # Rank Mean Weighted
    val_final["rank_mean_weighted"] = val_final.apply(weighted_mean, axis=1)
    # Sum RMSE
    val_final["sum_RMSE"] = val_final[list_rmse].sum(axis=1)
    # Global Rank
    val_final["rank"] = val_final[["rank_mean", "rank_mean_weighted"]].mean(axis=1)

    return val_final.sort_values(by=["rank"])
