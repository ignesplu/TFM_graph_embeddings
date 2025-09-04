import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import MessagePassing

from .GTMAE import (
    PreproGTMAE,  # Se utiliza el mismo preprocesado que en GTMAE
    grouped_undirected_split,
    make_supervised_edge_splits,
    pair_features_from_x,
    EarlyStopper,
    _rmse,
    _mae,
    _spearman,
    _r2,
    EdgeRegressor,
    NodeRegressor,
)


# +---------+
# |  MODEL  |
# +---------+


class EdgeSAGEConvEA(MessagePassing):
    """
    Edge-Attribute Aware SAGE convolution layer.
    Implements a SAGE convolution that explicitly incorporates edge attributes
    through linear transformations of both node features and edge attributes.

    Message computation: m_ij = W_n x_j + W_e e_ij
    Aggregation: mean_j(m_ij)
    Output: out_i = mean_j(m_ij) + W_s x_i
    """

    def __init__(self, in_channels: int, edge_dim: int, out_channels: int, bias: bool = True):
        """
        Initialize EdgeSAGEConvEA layer.

        Args:
            in_channels: Input feature dimension
            edge_dim: Edge attribute dimension
            out_channels: Output feature dimension
            bias: Whether to use bias in self linear transformation
        """
        super().__init__(aggr="mean")
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)

        # Xavier initialization for stability
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        if bias and self.lin_self.bias is not None:
            nn.init.zeros_(self.lin_self.bias)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass.

        Args:
            x: Node feature matrix [N, in_channels]
            edge_index: Edge index tensor [2, E]
            edge_attr: Edge attribute matrix [E, edge_dim]

        Returns:
            Updated node features [N, out_channels]
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)  # [N, out]
        out = out + self.lin_self(x)
        return out

    def message(self, x_j, edge_attr):
        """
        Compute messages for each edge.

        Args:
            x_j: Source node features [E, in_channels]
            edge_attr: Edge attributes [E, edge_dim]

        Returns:
            Transformed messages [E, out_channels]
        """
        return self.lin_neigh(x_j) + self.lin_edge(edge_attr)


class E2ASAGEEncoder(nn.Module):
    """
    E2A-SAGE encoder with optional batch normalization and L2 normalization.
    Two-layer Edge-Attribute Aware SAGE encoder with configurable
    normalization and regularization techniques.
    """

    def __init__(
        self,
        in_ch: int,
        edge_dim: int,
        hid: int = 128,
        out: int = 64,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        l2_norm_layers: bool = True,
    ):
        """
        Initialize E2ASAGEEncoder.

        Args:
            in_ch: Input feature dimension
            edge_dim: Edge attribute dimension
            hid: Hidden layer dimension
            out: Output dimension
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
            l2_norm_layers: Whether to apply L2 normalization after layers
        """
        super().__init__()
        self.conv1 = EdgeSAGEConvEA(in_ch, edge_dim, hid)
        self.conv2 = EdgeSAGEConvEA(hid, edge_dim, out)
        self.use_bn = use_batchnorm
        self.bn1 = nn.BatchNorm1d(hid) if use_batchnorm else nn.Identity()
        self.dropout = dropout
        self.l2_norm_layers = l2_norm_layers

    @staticmethod
    def _maybe_l2(x, enable):
        """
        Conditionally apply L2 normalization.

        Args:
            x: Input tensor
            enable: Whether to apply normalization

        Returns:
            Normalized tensor if enabled, else original tensor
        """
        return F.normalize(x, p=2, dim=-1) if enable else x

    def forward(self, x, edge_index, edge_attr, drop_prob: float = 0.2):
        """
        Forward pass with optional edge dropout.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            drop_prob: Edge dropout probability

        Returns:
            Node embeddings
        """
        # Edge dropout
        if drop_prob > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=drop_prob, training=True)
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]

        # Capa 1
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)
        h = self._maybe_l2(h, self.l2_norm_layers)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Capa 2
        z = self.conv2(h, edge_index, edge_attr)
        z = F.relu(z)
        z = self._maybe_l2(z, self.l2_norm_layers)
        z = F.dropout(z, p=self.dropout, training=self.training)

        return z


# +---------+
# |  TRAIN  |
# +---------+


def train_edge_node_multitask_sage(
    data: Data,
    device,
    target_cols: list,
    node_feat_names: list,
    node_target_cols: list,
    hid=128,
    out=64,
    dropout=0.2,
    use_batchnorm: bool = True,
    l2_norm_layers: bool = True,
    lr=1e-3,
    epochs=150,
    weight_decay=1e-4,
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
    dbg_print=True,
):
    """
    Train multitask E2A-SAGE model for edge and node attribute prediction.
    Extended version of the GTMAE training function using Edge-Attribute Aware
    SAGE convolution instead of TransformerConv for the encoder.

    Args:
        data: PyG Data object with graph structure
        device: Training device (CPU/GPU)
        target_cols: Edge target columns to predict
        node_feat_names: Node feature column names
        node_target_cols: Node target columns to predict
        hid: Hidden dimension for encoder
        out: Output dimension for encoder
        dropout: Dropout rate
        use_batchnorm: Whether to use batch normalization in encoder
        l2_norm_layers: Whether to apply L2 normalization in encoder
        lr: Learning rate
        epochs: Number of training epochs
        weight_decay: Weight decay for optimizer
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
        dbg_print: Whether to print debug information

    Returns:
        Tuple of (trained model, node embeddings)
    """

    # Split
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

    all_edge_cols = list(data.edge_continuous_cols) + ["edge_type"]
    edge_target_idx = [all_edge_cols.index(c) for c in target_cols]

    node_target_idx = [node_feat_names.index(c) for c in node_target_cols]
    node_out_dim = len(node_target_idx)

    if dbg_print:
        print(
            f"[SPLIT] train={train_mask.sum().item()}  val={val_mask.sum().item()}  test={test_mask.sum().item()}"
        )
        print(f"[EDGE TARGETS] {target_cols} -> idx {edge_target_idx}")
        print(f"[NODE TARGETS] {node_target_cols} -> idx {node_target_idx}")

    # Model
    class E2ASAGEGMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = E2ASAGEEncoder(
                in_ch=data.num_features,
                edge_dim=data.edge_attr.size(1),
                hid=hid,
                out=out,
                dropout=dropout,
                use_batchnorm=use_batchnorm,
                l2_norm_layers=l2_norm_layers,
            )
            self.edge_dec = EdgeRegressor(
                z_dim=out,
                out_dim=len(edge_target_idx),
                hid=hid,
                use_pair_feats=use_pair_feats,
            )
            self.node_dec = NodeRegressor(z_dim=out, out_dim=node_out_dim, hid=hid)

    model = E2ASAGEGMAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loss
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
        x = split.x.to(device)  # [N, F]
        ei = split.edge_index.to(device)  # [2, Etr]
        ea = split.edge_attr.to(device)  # [Etr, Fe]
        pos_ei = split.pos_edge_label_index.to(device)  # [2, Es]
        y_edge = split.pos_edge_attr.to(device)[:, edge_target_idx]  # [Es, Te]
        return x, ei, ea, pos_ei, y_edge

    x_tr, ei_tr, ea_tr, pos_ei_tr, y_edge_tr = _to_dev(train_data)
    x_va, ei_va, ea_va, pos_ei_va, y_edge_va = _to_dev(val_data)
    x_te, ei_te, ea_te, pos_ei_te, y_edge_te = _to_dev(test_data)

    # Node targets
    def _node_targets(x_tensor):
        return x_tensor[:, node_target_idx]  # [N, Tn]

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

    # Helpers ranking (edge)
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

    # Anti-leak mask for nodal targets
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

    # Train loop
    print("[TRAIN]")
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        # Mask node input
        x_tr_in = _mask_node_inputs(x_tr, node_mask_rate)

        z = model.encoder(x_tr_in, ei_tr, ea_tr, drop_prob=edge_drop_prob)

        # Edge head
        pf_tr = pair_features_from_x(x_tr_in, pos_ei_tr, mode=pair_mode) if use_pair_feats else None
        y_edge_hat = model.edge_dec(z, pos_ei_tr, pf_tr)  # [Es, Te]
        edge_reg_loss = edge_loss_fn(y_edge_hat, y_edge_tr)

        # Node head
        y_node_hat = model.node_dec(z)  # [N, Tn]
        node_reg_loss = (
            node_loss_fn(y_node_hat, y_node_tr)
            if node_out_dim > 0
            else torch.tensor(0.0, device=z.device)
        )

        # Ranking (edge)
        if add_ranking:
            rank_loss = _ranking_loss(z, pos_ei_tr, y_edge_tr, margin=margin)
        else:
            rank_loss = torch.tensor(0.0, device=z.device)

        loss = (
            edge_reg_loss
            + lambda_node * node_reg_loss
            + (lambda_rank * rank_loss if add_ranking else 0.0)
        )
        loss.backward()
        opt.step()

        # VAL
        model.eval()
        with torch.no_grad():
            z_va = model.encoder(x_va, ei_va, ea_va, drop_prob=0.0)

            # Edge val
            pf_va = (
                pair_features_from_x(x_va, pos_ei_va, mode=pair_mode) if use_pair_feats else None
            )
            y_edge_hat_va = model.edge_dec(z_va, pos_ei_va, pf_va)

            # Node val
            y_node_hat_va = model.node_dec(z_va) if node_out_dim > 0 else None

            # EDGE Metrics
            y_edge_va_np = y_edge_va.detach().cpu().numpy()
            y_edge_hat_va_np = y_edge_hat_va.detach().cpu().numpy()
            edge_val_rmse = _rmse(y_edge_va_np, y_edge_hat_va_np)
            edge_val_mae = _mae(y_edge_va_np, y_edge_hat_va_np)
            edge_val_spr = _spearman(y_edge_va_np.ravel(), y_edge_hat_va_np.ravel())
            edge_val_r2 = _r2(y_edge_va_np, y_edge_hat_va_np)

            # NODE Metrics
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
                f"| node={node_reg_loss.item():.4f} | rank={rank_loss.item():.4f} || "
                f"VAL edge(RMSE={edge_val_rmse:.4f}, MAE={edge_val_mae:.4f}, Sp={edge_val_spr:.4f}, R2={edge_val_r2:.4f})"
                f"| node(RMSE={node_val_rmse:.4f}, MAE={node_val_mae:.4f}, Sp={node_val_spr:.4f}, R2={node_val_r2:.4f})"
            )
            print(log)

        # Early stopping by monitor
        if monitor == "val_edge_rmse":
            score, es.mode = edge_val_rmse, "min"
        elif monitor == "val_edge_mae":
            score, es.mode = edge_val_mae, "min"
        elif monitor == "val_edge_spr":
            score, es.mode = (edge_val_spr if not (edge_val_spr != edge_val_spr) else -1e9), "max"
        elif monitor == "val_node_rmse":
            score, es.mode = node_val_rmse, "min"
        elif monitor == "val_node_mae":
            score, es.mode = node_val_mae, "min"
        else:
            score, es.mode = (edge_val_spr if not (edge_val_spr != edge_val_spr) else -1e9), "max"

        if es.step(score, model):
            if print_every:
                print(f"Early stopping en epoch {ep} (mejor {monitor}={es.best_score:.4f}).")
            break

    es.maybe_restore(model)

    # Final embeddings
    model.eval()
    with torch.no_grad():
        Z = model.encoder(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        ).cpu()

    # Embedding DataFrame
    node_ids = getattr(data, "node_ids", list(range(data.num_nodes)))
    emb_cols = [f"z_{i}" for i in range(Z.shape[1])]
    embeddings_df = pd.DataFrame(Z.numpy(), index=node_ids, columns=emb_cols)
    embeddings_df.index.name = getattr(data, "node_id_col", "node_id")

    # TEST
    with torch.no_grad():
        # EDGE
        z_te = model.encoder(x_te, ei_te, ea_te, drop_prob=0.0)
        pf_te = pair_features_from_x(x_te, pos_ei_te, mode=pair_mode) if use_pair_feats else None
        y_edge_hat_te = model.edge_dec(z_te, pos_ei_te, pf_te)
        edge_test_rmse = _rmse(
            y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy()
        )
        edge_test_mae = _mae(y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy())
        edge_test_spr = _spearman(
            y_edge_te.detach().cpu().numpy().ravel(),
            y_edge_hat_te.detach().cpu().numpy().ravel(),
        )
        edge_test_r2 = _r2(y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy())

        # NODE
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
            node_test_rmse = node_test_mae = node_test_spr = node_test_sr2 = float("nan")

    if print_every:
        print("[TEST]")
        print(
            f" [EDGE] RMSE={edge_test_rmse:.6f} | MAE={edge_test_mae:.6f} | Sp={edge_test_spr:.6f} | R2={edge_test_r2:.6f}"
        )
        print(
            f" [NODE] RMSE={node_test_rmse:.6f} | MAE={node_test_mae:.6f} | Sp={node_test_spr:.6f} | R2={node_test_sr2:.6f}"
        )

    return model, Z, embeddings_df
