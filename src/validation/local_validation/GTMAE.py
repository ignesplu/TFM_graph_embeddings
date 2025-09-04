import pandas as pd
import itertools
import time

import torch

from ...models.GTMAE import (
    grouped_undirected_split,
    make_supervised_edge_splits,
    pair_features_from_x,
    _rmse,
    _mae,
    _spearman,
    _r2,
    train_edge_node_multitask,
)


@torch.no_grad()
def evaluate_gtmae(
    model,
    data,
    node_feat_names,
    edge_attr_names,
    target_cols,
    node_target_cols,
    seed=33,
    val_ratio=0.2,
    test_ratio=0.2,
    device="cpu",
    use_pair_feats=True,
    pair_mode="cosine_l2_absdiff",
):
    """
    Evaluate GTMAE model performance on validation and test sets.

    Reconstructs the same data splits used during training and computes
    comprehensive evaluation metrics for both edge and node prediction tasks.

    Args:
        model: Trained GTMAE model
        data: PyG Data object with graph structure
        node_feat_names: List of node feature column names
        edge_attr_names: List of edge attribute column names
        target_cols: Edge target columns to evaluate
        node_target_cols: Node target columns to evaluate
        seed: Random seed for reproducible splits
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        device: Device for evaluation (CPU/GPU)
        use_pair_feats: Whether to use pair features
        pair_mode: Pair feature computation mode

    Returns:
        Dictionary containing validation and test metrics for both edge and node tasks
    """
    model.eval()

    train_mask, val_mask, test_mask = grouped_undirected_split(
        data.edge_index,
        data.edge_is_undirected,
        num_nodes=data.num_nodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    _, val_data, test_data = make_supervised_edge_splits(data, train_mask, val_mask, test_mask)

    all_edge_cols = list(data.edge_continuous_cols) + ["edge_type"]
    edge_target_idx = [all_edge_cols.index(c) for c in target_cols]
    node_target_idx = [node_feat_names.index(c) for c in node_target_cols]

    def _prepare(split):
        x = split.x.to(device)
        ei = split.edge_index.to(device)
        ea = split.edge_attr.to(device)
        pos_ei = split.pos_edge_label_index.to(device)
        y_edge = split.pos_edge_attr.to(device)[:, edge_target_idx]  # [E, Te]
        y_node = x[:, node_target_idx]  # [N, Tn]
        return x, ei, ea, pos_ei, y_edge, y_node

    # VAL
    x_v, ei_v, ea_v, pos_ei_v, y_edge_v, y_node_v = _prepare(val_data)
    z_v = model.encoder(x_v, ei_v, ea_v, drop_prob=0.0)
    pf_v = pair_features_from_x(x_v, pos_ei_v, mode=pair_mode) if use_pair_feats else None
    y_edge_hat_v = model.edge_dec(z_v, pos_ei_v, pf_v)
    y_node_hat_v = model.node_dec(z_v)

    # TEST
    x_t, ei_t, ea_t, pos_ei_t, y_edge_t, y_node_t = _prepare(test_data)
    z_t = model.encoder(x_t, ei_t, ea_t, drop_prob=0.0)
    pf_t = pair_features_from_x(x_t, pos_ei_t, mode=pair_mode) if use_pair_feats else None
    y_edge_hat_t = model.edge_dec(z_t, pos_ei_t, pf_t)
    y_node_hat_t = model.node_dec(z_t)

    # Metrics
    def _np(t):
        return t.detach().cpu().numpy()

    # VAL
    e_true_v, e_pred_v = _np(y_edge_v), _np(y_edge_hat_v)
    n_true_v, n_pred_v = _np(y_node_v), _np(y_node_hat_v)

    edge_val_rmse = _rmse(e_true_v, e_pred_v)
    edge_val_mae = _mae(e_true_v, e_pred_v)
    try:
        edge_val_r2 = float(_r2(e_true_v, e_pred_v))
    except Exception:
        edge_val_r2 = float("nan")
    edge_val_sp = _spearman(e_true_v.ravel(), e_pred_v.ravel())

    node_val_rmse = _rmse(n_true_v, n_pred_v)
    node_val_mae = _mae(n_true_v, n_pred_v)
    try:
        node_val_r2 = float(_r2(n_true_v, n_pred_v))
    except Exception:
        node_val_r2 = float("nan")
    node_val_sp = _spearman(n_true_v.ravel(), n_pred_v.ravel())

    # TEST
    e_true_t, e_pred_t = _np(y_edge_t), _np(y_edge_hat_t)
    n_true_t, n_pred_t = _np(y_node_t), _np(y_node_hat_t)

    edge_test_rmse = _rmse(e_true_t, e_pred_t)
    edge_test_mae = _mae(e_true_t, e_pred_t)
    try:
        edge_test_r2 = float(_r2(e_true_t, e_pred_t))
    except Exception:
        edge_test_r2 = float("nan")
    edge_test_sp = _spearman(e_true_t.ravel(), e_pred_t.ravel())

    node_test_rmse = _rmse(n_true_t, n_pred_t)
    node_test_mae = _mae(n_true_t, n_pred_t)
    try:
        node_test_r2 = float(_r2(n_true_t, n_pred_t))
    except Exception:
        node_test_r2 = float("nan")
    node_test_sp = _spearman(n_true_t.ravel(), n_pred_t.ravel())

    return {
        # VAL
        "val_edge_rmse": edge_val_rmse,
        "val_edge_mae": edge_val_mae,
        "val_edge_r2": edge_val_r2,
        "val_edge_spearman": edge_val_sp,
        "val_node_rmse": node_val_rmse,
        "val_node_mae": node_val_mae,
        "val_node_r2": node_val_r2,
        "val_node_spearman": node_val_sp,
        # TEST
        "test_edge_rmse": edge_test_rmse,
        "test_edge_mae": edge_test_mae,
        "test_edge_r2": edge_test_r2,
        "test_edge_spearman": edge_test_sp,
        "test_node_rmse": node_test_rmse,
        "test_node_mae": node_test_mae,
        "test_node_r2": node_test_r2,
        "test_node_spearman": node_test_sp,
    }


def default_gtmae_param_grid():
    """
    Define default hyperparameter grid for GTMAE model optimization.

    Returns a comprehensive parameter grid covering architectural,
    regularization, and training hyperparameters for grid search.

    Returns:
        Dictionary of hyperparameter lists for grid search
    """
    grid = {
        "hid": [96, 128],
        "out": [64, 128],
        "heads": [2, 4],
        "dropout": [0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-5, 1e-4],
        "edge_drop_prob": [0.0, 0.2],
        "edge_loss_type": ["huber"],  # ["huber", "mse"]
        "edge_huber_delta": [1.0],
        "node_loss_type": ["huber"],
        "node_huber_delta": [1.0],
        "lambda_node": [0.25, 0.5, 1.0],
        "node_mask_rate": [0.0, 0.2],
        "add_ranking": [False, True],
        "lambda_rank": [0.3],  # used only if add_ranking=True
        "margin": [0.1],
        "monitor": ["val_edge_rmse"],  # ["val_node_rmse", "val_edge_spearman", ...]
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


def _cartesian_product(param_grid: dict):
    """
    Generate Cartesian product of hyperparameter values.

    Args:
        param_grid: Dictionary where keys are parameter names and values are lists

    Yields:
        Dictionary representing a unique hyperparameter combination
    """
    keys = list(param_grid.keys())
    for values in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))


def composite_score(row, w_edge=0.5, w_node=0.5):
    """
    Compute composite score for model ranking.
    Calculates a weighted combination of edge and node RMSE scores
    for comparing model performance across multiple objectives.

    Args:
        row: Dictionary containing validation metrics
        w_edge: Weight for edge RMSE component
        w_node: Weight for node RMSE component

    Returns:
        Composite score (lower is better)
    """

    return w_edge * row["val_edge_rmse"] + w_node * row["val_node_rmse"]


def run_gtmae_gridsearch(
    data,
    node_feat_names,
    edge_attr_names,
    device,
    param_grid=None,
    limit=None,
    sort_weights=(0.5, 0.5),
):
    """
    Execute comprehensive grid search for GTMAE hyperparameter optimization.

    Performs systematic hyperparameter search across specified parameter grid,
    trains models with different configurations, and evaluates performance
    using composite scoring. Tracks best performing model and embeddings.

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
    grid = param_grid or default_gtmae_param_grid()
    combos = list(_cartesian_product(grid))
    if limit is not None:
        combos = combos[:limit]

    results = []
    best_row, best_model, best_Z = None, None, None
    t0 = time.time()

    for i, hp in enumerate(combos, 1):
        desc = f"{i}/{len(combos)}"
        try:
            model, Z, _ = train_edge_node_multitask(
                data=data,
                target_cols=target_cols,
                node_feat_names=node_feat_names,
                node_target_cols=node_target_cols,
                hid=hp["hid"],
                out=hp["out"],
                heads=hp["heads"],
                dropout=hp["dropout"],
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
