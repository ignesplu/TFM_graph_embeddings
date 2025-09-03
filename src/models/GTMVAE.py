import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dropout_edge

from .GTMAE import (
    PreproGTMAE,  # Se utiliza el mismo preprocesado que en GTMAE
    grouped_undirected_split,
    make_supervised_edge_splits,
    pair_features_from_x,
    EarlyStopper,
    EdgeRegressor,
    NodeRegressor,
    _rmse,
    _mae,
    _spearman,
    _r2,
)


# +------------------+
# |  TRAINING UTILS  |
# +------------------+


def _kl_normal(
    mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between normal distribution and standard normal.
    Calculates KL(q(z|x)=N(mu,diag(sigma^2)) || p(z)=N(0,I)).

    Args:
        mu: Mean of the variational distribution
        logvar: Log variance of the variational distribution
        reduction: Reduction method - 'mean', 'sum', or None for per-sample

    Returns:
        KL divergence value(s)
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    else:
        return kl.sum(dim=-1)  # por muestra


def _reparameterize(
    mu: torch.Tensor, logvar: torch.Tensor, training: bool
) -> torch.Tensor:
    """
    Reparameterization trick for sampling from normal distribution.

    Args:
        mu: Mean of the distribution
        logvar: Log variance of the distribution
        training: Whether in training mode (stochastic) or eval mode (deterministic)

    Returns:
        Sampled latent vector (stochastic in training, deterministic in eval)
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    else:
        return mu  # modo determinista en eval


# +---------+
# |  MODEL  |
# +---------+


class EdgeAwareGCNVEncoder(nn.Module):
    """
    Variational encoder with pre-bottleneck MLP and GNN layers.

    Encoder that compresses features through a pre-bottleneck MLP,
    processes through TransformerConv layers, and outputs variational
    parameters (mu, logvar) for latent representation.
    """

    def __init__(
        self,
        in_ch,
        edge_dim,
        hid=128,
        out=64,
        heads=2,
        dropout=0.2,
        pre_dim=256,
        pre_drop=0.1,
    ):
        """
        Initialize variational encoder.

        Args:
            in_ch: Input feature dimension
            edge_dim: Edge attribute dimension
            hid: Hidden dimension for GNN
            out: Output latent dimension
            heads: Number of attention heads
            dropout: Dropout rate
            pre_dim: Pre-bottleneck dimension
            pre_drop: Pre-bottleneck dropout rate
        """
        super().__init__()

        # 1) Pre-bottleneck: comprime feats en pre_dim
        self.pre = nn.Sequential(
            nn.Linear(in_ch, pre_dim), nn.ReLU(), nn.Dropout(pre_drop)
        )

        # 2) GNN
        self.conv1 = TransformerConv(
            pre_dim, hid, heads=heads, edge_dim=edge_dim, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hid * heads)

        self.conv2 = TransformerConv(
            hid * heads, out, heads=1, edge_dim=edge_dim, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(out)

        self.dropout = dropout

        # 3) Cabezas variacionales
        self.lin_mu = nn.Linear(out, out)
        self.lin_logvar = nn.Linear(out, out)

    def forward(self, x, edge_index, edge_attr, drop_prob=0.2):
        """
        Forward pass with variational encoding.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            drop_prob: Edge dropout probability

        Returns:
            Tuple of (mu, logvar) for variational distribution
        """
        # Pre-bottleneck
        x = self.pre(x)

        # Edge dropout (solo índice/atributos de arista)
        if drop_prob > 0 and self.training:
            edge_index, edge_mask = dropout_edge(
                edge_index, p=drop_prob, training=self.training
            )
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        # GNN 1
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = self.norm1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GNN 2
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)

        # Var heads
        mu = self.lin_mu(h)
        logvar = self.lin_logvar(h)
        return mu, logvar


# +---------+
# |  TRAIN  |
# +---------+


def train_edge_node_multitask_v(
    data: Data,
    device,
    target_cols: list,
    node_feat_names: list,
    node_target_cols: list,
    hid=128,
    out=64,
    lr=1e-3,
    epochs=150,
    weight_decay=1e-4,
    dropout=0.2,
    heads=2,
    print_every=20,
    patience=30,
    min_delta=0.0,
    monitor="val_edge_rmse",
    restore_best=True,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=33,
    use_pair_feats=True,
    pair_mode="cosine_l2_absdiff",
    edge_loss_type="huber",
    edge_huber_delta=1.0,
    node_loss_type="huber",
    node_huber_delta=1.0,
    add_ranking=False,
    lambda_rank=0.5,
    margin=0.1,
    lambda_node=1.0,
    edge_drop_prob=0.2,
    node_mask_rate=0.15,
    beta_kl=1e-3,
    kl_warmup=10,
    dbg_print=True,
):
    """
    Train variational multitask model for edge and node attribute prediction.

    Variational version of the GTMAE training function that incorporates
    KL divergence regularization and stochastic latent representations.

    Args:
        data: PyG Data object with graph structure
        device: Training device (CPU/GPU)
        target_cols: Edge target columns to predict
        node_feat_names: Node feature column names
        node_target_cols: Node target columns to predict
        hid: Hidden dimension for encoder
        out: Output dimension for encoder
        lr: Learning rate
        epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
        dropout: Dropout rate
        heads: Number of attention heads
        print_every: Print frequency
        patience: Early stopping patience
        min_delta: Early stopping minimum delta
        monitor: Metric to monitor for early stopping
        restore_best: Whether to restore best model
        val_ratio: Validation ratio
        test_ratio: Test ratio
        seed: Random seed
        use_pair_feats: Whether to use pair features
        pair_mode: Pair feature computation mode
        edge_loss_type: Loss type for edge prediction
        edge_huber_delta: Delta for Huber loss (edge)
        node_loss_type: Loss type for node prediction
        node_huber_delta: Delta for Huber loss (node)
        add_ranking: Whether to add ranking loss
        lambda_rank: Weight for ranking loss
        margin: Margin for ranking loss
        lambda_node: Weight for node loss
        edge_drop_prob: Edge dropout probability
        node_mask_rate: Node feature masking rate
        beta_kl: Weight for KL divergence term
        kl_warmup: Number of epochs for KL warmup
        dbg_print: Whether to print debug information

    Returns:
        Tuple of (trained model, deterministic node embeddings)
    """
    # ---------- split de aristas ----------
    train_mask, val_mask, test_mask = grouped_undirected_split(
        data.edge_index,
        data.edge_is_undirected,
        num_nodes=data.num_nodes,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_data, val_data, test_data = make_supervised_edge_splits(
        data, train_mask, val_mask, test_mask
    )

    # Mapear objetivos de ARISTA
    all_edge_cols = list(data.edge_continuous_cols) + ["edge_type"]
    edge_target_idx = [all_edge_cols.index(c) for c in target_cols]

    # Mapear objetivos de NODO
    node_target_idx = [node_feat_names.index(c) for c in node_target_cols]
    node_out_dim = len(node_target_idx)

    if dbg_print:
        print(
            f"[SPLIT] train={train_mask.sum().item()}  val={val_mask.sum().item()}  test={test_mask.sum().item()}"
        )
        print(f"[EDGE TARGETS] {target_cols} -> idx {edge_target_idx}")
        print(f"[NODE TARGETS] {node_target_cols} -> idx {node_target_idx}")

    # ---------- modelo ----------
    class GTMultiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = EdgeAwareGCNVEncoder(
                data.num_features,
                edge_dim=data.edge_attr.size(1),
                hid=hid,
                out=out,
                dropout=dropout,
                heads=heads,
            )
            self.edge_dec = EdgeRegressor(
                z_dim=out,
                out_dim=len(edge_target_idx),
                hid=hid,
                use_pair_feats=use_pair_feats,
            )
            self.node_dec = NodeRegressor(z_dim=out, out_dim=node_out_dim, hid=hid)

    model = GTMultiModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Pérdidas de reconstrucción
    def _make_reg_loss(kind, delta):
        if kind == "huber":
            return nn.SmoothL1Loss(beta=delta)
        if kind == "mse":
            return nn.MSELoss()
        raise ValueError("loss_type debe ser 'huber' o 'mse'.")

    edge_loss_fn = _make_reg_loss(edge_loss_type, edge_huber_delta)
    node_loss_fn = _make_reg_loss(node_loss_type, node_huber_delta)

    # Tensors base (edge)
    def _to_dev(split):
        x = split.x.to(device)
        ei = split.edge_index.to(device)
        ea = split.edge_attr.to(device)
        pos_ei = split.pos_edge_label_index.to(device)
        y_edge = split.pos_edge_attr.to(device)[:, edge_target_idx]
        return x, ei, ea, pos_ei, y_edge

    x_tr, ei_tr, ea_tr, pos_ei_tr, y_edge_tr = _to_dev(train_data)
    x_va, ei_va, ea_va, pos_ei_va, y_edge_va = _to_dev(val_data)
    x_te, ei_te, ea_te, pos_ei_te, y_edge_te = _to_dev(test_data)

    # Targets nodales
    def _node_targets(x_tensor):
        return x_tensor[:, node_target_idx]

    y_node_tr = _node_targets(x_tr).detach()
    y_node_va = _node_targets(x_va).detach()
    y_node_te = _node_targets(x_te).detach()

    # Early stopping
    es = EarlyStopper(
        patience=patience,
        min_delta=min_delta,
        mode=("min" if "rmse" in monitor or "mae" in monitor else "max"),
        restore_best=restore_best,
    )

    # Ranking (edge)
    def _ranking_loss(z, pos_ei, y_true, margin=0.1):
        with torch.no_grad():
            u = pos_ei[0].cpu().numpy()
            v = pos_ei[1].cpu().numpy()
            y = y_true.detach().cpu().numpy()
            if y.ndim == 2:
                y_scalar = y.mean(axis=1)
            else:
                y_scalar = y
            df = pd.DataFrame({"u": u, "v": v, "y": y_scalar})
            pos_pairs = []
            for uu, grp in df.groupby("u"):
                if len(grp) < 2:
                    continue
                v_pos = grp.sort_values("y", ascending=False).iloc[0]["v"]
                v_neg = grp.sort_values("y", ascending=True).iloc[0]["v"]
                if v_pos == v_neg:
                    continue
                pos_pairs.append((int(uu), int(v_pos), int(v_neg)))
            if not pos_pairs:
                return torch.tensor(0.0, device=z.device)
        uu = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=z.device)
        vp = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=z.device)
        vn = torch.tensor([p[2] for p in pos_pairs], dtype=torch.long, device=z.device)

        zu = F.normalize(z[uu], p=2, dim=-1)
        zvp = F.normalize(z[vp], p=2, dim=-1)
        zvn = F.normalize(z[vn], p=2, dim=-1)
        s_pos = (zu * zvp).sum(dim=-1)
        s_neg = (zu * zvn).sum(dim=-1)
        return torch.clamp(margin - s_pos + s_neg, min=0.0).mean()

    def _mask_node_inputs(x_tensor, rate):
        if rate <= 0:
            return x_tensor
        x_masked = x_tensor.clone()
        if len(node_target_idx) > 0:
            m = torch.bernoulli(
                torch.full(
                    (x_tensor.size(0), len(node_target_idx)),
                    1.0 - rate,
                    device=x_tensor.device,
                )
            )
            x_masked[:, node_target_idx] = x_masked[:, node_target_idx] * m
        return x_masked

    # --------------- TRAIN LOOP ---------------
    print("[TRAIN]")
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # Warmup KL
        kl_coeff = float(beta_kl) * min(1.0, ep / max(1, kl_warmup))

        # Enmascara SOLO columnas target nodales en la entrada de TRAIN
        x_tr_in = _mask_node_inputs(x_tr, node_mask_rate)

        # Encoder variacional (TRAIN): mu, logvar -> z
        mu_tr, logvar_tr = model.encoder(
            x_tr_in, ei_tr, ea_tr, drop_prob=edge_drop_prob
        )
        z = _reparameterize(mu_tr, logvar_tr, training=True)

        # ----- Edge head -----
        pf_tr = (
            pair_features_from_x(x_tr_in, pos_ei_tr, mode=pair_mode)
            if use_pair_feats
            else None
        )
        y_edge_hat = model.edge_dec(z, pos_ei_tr, pf_tr)
        edge_reg_loss = edge_loss_fn(y_edge_hat, y_edge_tr)

        # ----- Node head -----
        y_node_hat = model.node_dec(z)
        node_reg_loss = (
            node_loss_fn(y_node_hat, y_node_tr)
            if node_out_dim > 0
            else torch.tensor(0.0, device=z.device)
        )

        # ----- KL -----
        kl_loss = _kl_normal(mu_tr, logvar_tr, reduction="mean")

        # ----- Ranking opcional (edge) -----
        if add_ranking:
            rank_loss = _ranking_loss(z, pos_ei_tr, y_edge_tr, margin=margin)
        else:
            rank_loss = torch.tensor(0.0, device=z.device)

        loss = (
            edge_reg_loss
            + lambda_node * node_reg_loss
            + (lambda_rank * rank_loss if add_ranking else 0.0)
            + kl_coeff * kl_loss
        )
        loss.backward()
        opt.step()

        # ---- VALIDACIÓN (determinista: usar mu) ----
        model.eval()
        with torch.no_grad():
            mu_va, logvar_va = model.encoder(x_va, ei_va, ea_va, drop_prob=0.0)
            z_va = mu_va

            # Edge val
            pf_va = (
                pair_features_from_x(x_va, pos_ei_va, mode=pair_mode)
                if use_pair_feats
                else None
            )
            y_edge_hat_va = model.edge_dec(z_va, pos_ei_va, pf_va)
            # Node val
            y_node_hat_va = model.node_dec(z_va) if node_out_dim > 0 else None

            # Métricas EDGE
            y_edge_va_np = y_edge_va.detach().cpu().numpy()
            y_edge_hat_va_np = y_edge_hat_va.detach().cpu().numpy()
            edge_val_rmse = _rmse(y_edge_va_np, y_edge_hat_va_np)
            edge_val_mae = _mae(y_edge_va_np, y_edge_hat_va_np)
            edge_val_spr = _spearman(y_edge_va_np.ravel(), y_edge_hat_va_np.ravel())
            edge_val_r2 = _r2(y_edge_va_np, y_edge_hat_va_np)

            # Métricas NODE
            if node_out_dim > 0:
                y_node_va_np = y_node_va.detach().cpu().numpy()
                y_node_hat_va_np = y_node_hat_va.detach().cpu().numpy()
                node_val_rmse = _rmse(y_node_va_np, y_node_hat_va_np)
                node_val_mae = _mae(y_node_va_np, y_node_hat_va_np)
                node_val_spr = _spearman(y_node_va_np.ravel(), y_node_hat_va_np.ravel())
                node_val_r2 = _r2(y_node_va_np, y_node_hat_va_np)
            else:
                node_val_rmse = node_val_mae = node_val_spr = node_val_r2 = float("nan")

        if (ep % print_every == 0) or (ep == 1):
            log = (
                f" [{ep:03d}] total={loss.item():.4f} | edge={edge_reg_loss.item():.4f} "
                f"| node={node_reg_loss.item():.4f} | rank={rank_loss.item():.4f} | kl({kl_coeff:.5f})={kl_loss.item():.4f} || "
                f"VAL edge(RMSE={edge_val_rmse:.4f}, MAE={edge_val_mae:.4f}, Sp={edge_val_spr:.4f}, R2={edge_val_r2:.4f})"
                f"| node(RMSE={node_val_rmse:.4f}, MAE={node_val_mae:.4f}, Sp={node_val_spr:.4f}, R2={node_val_r2:.4f})"
            )
            print(log)

        # ---- early stopping según 'monitor' ----
        if monitor == "val_edge_rmse":
            score, es.mode = edge_val_rmse, "min"
        elif monitor == "val_edge_mae":
            score, es.mode = edge_val_mae, "min"
        elif monitor == "val_edge_spr":
            score, es.mode = (
                edge_val_spr if not (edge_val_spr != edge_val_spr) else -1e9
            ), "max"
        elif monitor == "val_node_rmse":
            score, es.mode = node_val_rmse, "min"
        elif monitor == "val_node_mae":
            score, es.mode = node_val_mae, "min"
        else:
            score, es.mode = (
                node_val_spr if not (node_val_spr != node_val_spr) else -1e9
            ), "max"

        if es.step(score, model):
            if print_every:
                print(
                    f"Early stopping en epoch {ep} (mejor {monitor}={es.best_score:.4f})."
                )
            break

    es.maybe_restore(model)

    # ---- Embeddings finales en el grafo COMPLETO (determinista: mu) ----
    model.eval()
    with torch.no_grad():
        mu_full, _ = model.encoder(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        )
        Z = mu_full.cpu()

    # ---- TEST ----
    with torch.no_grad():
        mu_te, _ = model.encoder(x_te, ei_te, ea_te, drop_prob=0.0)
        z_te = mu_te
        pf_te = (
            pair_features_from_x(x_te, pos_ei_te, mode=pair_mode)
            if use_pair_feats
            else None
        )
        y_edge_hat_te = model.edge_dec(z_te, pos_ei_te, pf_te)
        edge_test_rmse = _rmse(
            y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy()
        )
        edge_test_mae = _mae(
            y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy()
        )
        edge_test_spr = _spearman(
            y_edge_te.detach().cpu().numpy().ravel(),
            y_edge_hat_te.detach().cpu().numpy().ravel(),
        )
        edge_test_r2 = _r2(
            y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy()
        )

        y_node_hat_te = model.node_dec(z_te) if node_out_dim > 0 else None
        if node_out_dim > 0:
            node_test_rmse = _rmse(
                y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy()
            )
            node_test_mae = _mae(
                y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy()
            )
            node_test_spr = _spearman(
                y_node_te.detach().cpu().numpy().ravel(),
                y_node_hat_te.detach().cpu().numpy().ravel(),
            )
            node_test_sr2 = _r2(
                y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy()
            )
        else:
            node_test_rmse = node_test_mae = node_test_spr = node_test_sr2 = float(
                "nan"
            )

    if print_every:
        print("[TEST]")
        print(
            f" [EDGE] RMSE={edge_test_rmse:.6f} | MAE={edge_test_mae:.6f} | Sp={edge_test_spr:.6f} | R2={edge_test_r2:.6f}"
        )
        print(
            f" [NODE] RMSE={node_test_rmse:.6f} | MAE={node_test_mae:.6f} | Sp={node_test_spr:.6f} | R2={node_test_sr2:.6f}"
        )

    return model, Z
