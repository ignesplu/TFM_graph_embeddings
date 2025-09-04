import pandas as pd
import time

import torch

from .GTMAE import _cartesian_product, composite_score, evaluate_gtmae
from ...models.E2A_SAGE_MAE import train_edge_node_multitask_sage


def default_sage_param_grid():
    """
    Define default hyperparameter grid for E2A-SAGE model optimization.

    Returns a comprehensive parameter grid specifically tuned for the
    Edge-Attribute Aware SAGE architecture, covering architectural,
    regularization, and training hyperparameters for grid search.

    Returns:
        Dictionary of hyperparameter lists for grid search
    """
    grid = {
        "hid": [96, 128],
        "out": [64, 128],
        "use_batchnorm": [True],
        "l2_norm_layers": [True, False],
        "dropout": [0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-5, 1e-4],
        "edge_drop_prob": [0.0, 0.2],
        "edge_loss_type": ["huber"],
        "edge_huber_delta": [1.0],
        "node_loss_type": ["huber"],
        "node_huber_delta": [1.0],
        "lambda_node": [0.25, 0.5, 1.0],
        "node_mask_rate": [0.0, 0.2],
        "add_ranking": [False, True],
        "lambda_rank": [0.3],
        "margin": [0.1],
        "monitor": ["val_edge_rmse"],
        "patience": [30],
        "min_delta": [0.0],
        "val_ratio": [0.2],
        "test_ratio": [0.2],
        "seed": [33],
        "pair_mode": ["cosine_l2_absdiff"],
        "use_pair_feats": [True],
        "print_every": [50],
        "epochs": [250],
    }
    return grid


def run_sage_gridsearch(
    data,
    node_feat_names,
    edge_attr_names,
    device,
    param_grid=None,
    limit=None,
    sort_weights=(0.5, 0.5),
):
    """
    Execute comprehensive grid search for E2A-SAGE hyperparameter optimization.

    Performs systematic hyperparameter search across specified parameter grid
    for the Edge-Attribute Aware SAGE model. Trains models with different
    configurations and evaluates performance using composite scoring.
    Tracks best performing model and embeddings.

    Args:
        data: PyG Data object with graph structure
        node_feat_names: List of node feature column names
        edge_attr_names: List of edge attribute column names
        device: Training device (CPU/GPU)
        param_grid: Custom parameter grid (uses default if None)
        limit: Maximum number of combinations to test (for debugging)
        sort_weights: Tuple of (edge_weight, node_weight) for composite scoring

    Returns:
        Tuple of (results DataFrame, best model, best embeddings)
    """
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Targets
    target_cols = list(edge_attr_names)
    node_target_cols = list(node_feat_names)

    # Grid
    grid = param_grid or default_sage_param_grid()
    combos = list(_cartesian_product(grid))
    if limit is not None:
        combos = combos[:limit]

    results = []
    best_row, best_model, best_Z = None, None, None
    t0 = time.time()

    for i, hp in enumerate(combos, 1):
        desc = f"{i}/{len(combos)}"
        try:
            model, Z, _ = train_edge_node_multitask_sage(
                data=data,
                target_cols=target_cols,
                node_feat_names=node_feat_names,
                node_target_cols=node_target_cols,
                hid=hp["hid"],
                out=hp["out"],
                dropout=hp["dropout"],
                use_batchnorm=hp["use_batchnorm"],
                l2_norm_layers=hp["l2_norm_layers"],
                lr=hp["lr"],
                weight_decay=hp["weight_decay"],
                edge_drop_prob=hp["edge_drop_prob"],
                edge_loss_type=hp["edge_loss_type"],
                edge_huber_delta=hp["edge_huber_delta"],
                node_loss_type=hp["node_loss_type"],
                node_huber_delta=hp["node_huber_delta"],
                lambda_node=hp["lambda_node"],
                node_mask_rate=hp["node_mask_rate"],
                add_ranking=hp["add_ranking"],
                lambda_rank=hp["lambda_rank"],
                margin=hp["margin"],
                monitor=hp["monitor"],
                patience=hp["patience"],
                min_delta=hp["min_delta"],
                val_ratio=hp["val_ratio"],
                test_ratio=hp["test_ratio"],
                seed=hp["seed"],
                use_pair_feats=hp["use_pair_feats"],
                pair_mode=hp["pair_mode"],
                print_every=hp["print_every"],
                epochs=hp["epochs"],
                device=device,
                dbg_print=False,
            )

            metrics = evaluate_gtmae(
                model=model,
                data=data,
                node_feat_names=node_feat_names,
                edge_attr_names=edge_attr_names,
                target_cols=target_cols,
                node_target_cols=node_target_cols,
                seed=hp["seed"],
                val_ratio=hp["val_ratio"],
                test_ratio=hp["test_ratio"],
                device=device,
                use_pair_feats=hp["use_pair_feats"],
                pair_mode=hp["pair_mode"],
            )

            row = {**hp, **metrics}
            row["combo_idx"] = i
            row["seconds"] = round(time.time() - t0, 2)

            # Grouped score
            row["score"] = composite_score(row, *sort_weights)
            results.append(row)

            # Track best
            if (best_row is None) or (row["score"] < best_row["score"]):
                best_row, best_model, best_Z = row, model, Z

            print(
                f"[{desc}] score={row['score']:.4f}  edge_RMSE={row['val_edge_rmse']:.4f}  node_RMSE={row['val_node_rmse']:.4f}"
            )

        except Exception as e:
            print(f"[{desc}] ERROR: {e}")
            continue

    df = pd.DataFrame(results).sort_values("score", ascending=True).reset_index(drop=True)
    return df, best_model, best_Z
