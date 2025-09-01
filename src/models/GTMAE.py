import pandas as pd
import numpy as np
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dropout_edge

from .utils import global_prepro


# +-----------------+
# |  PREPROCESSING  |
# +-----------------+

class PreproGTMAE:
    """
    Preprocesa tabular/temporal y construye un grafo dirigido con:
      - mdir: aristas dirigidas tal cual
      - mndi: se duplican ambas direcciones
    edge_attr incluye columnas continuas estandarizadas + edge_type (0=dir,1=no-dir)
    """
    def __init__(
            self,
            NODE_ID_COL: str = "cc",
            MDIR_SRC: str = "cc_origen",
            MDIR_DST: str = "cc_destino",
            MDIR_WEIGHT_COLS: list = ["distancia_carretera"],
            MNDI_U : str = "cc_origen",
            MNDI_V : str = "cc_destino",
            MNDI_WEIGHT_COLS : list = ['distancia_vuelo_pajaro', 'bool_colindan_cc', 'n_grado_union_metro', 'n_grado_union_cercanias', 'n_grado_union_buses_EMT', 'n_grado_union_buses_urb', 'n_grado_union_buses_int'],
            YEAR_COL : str = "year",
            TARGET_YEAR : int = 2022,
            EPS : float = 1e-9,
            add_idea_emb: bool = True,
            no_mad: bool = False,
    ):
        self.NODE_ID_COL = NODE_ID_COL
        self.MDIR_SRC = MDIR_SRC
        self.MDIR_DST = MDIR_DST
        self.MDIR_WEIGHT_COLS = MDIR_WEIGHT_COLS
        self.MNDI_U = MNDI_U
        self.MNDI_V = MNDI_V
        self.MNDI_WEIGHT_COLS = MNDI_WEIGHT_COLS
        self.YEAR_COL = YEAR_COL
        self.TARGET_YEAR = TARGET_YEAR
        self.EPS = EPS
        self.add_idea_emb = add_idea_emb
        self.no_mad = no_mad

    def _prepro_tabular_data(self, df):
        mean_not_idea = [col for col in df.columns if col.endswith('_mean') and not col.startswith('idea_')]
        std_not_idea = [col for col in df.columns if col.endswith('_std') and not col.startswith('idea_')]
        mean_xh_not_idea = [col for col in df.columns if col.endswith('_mean_por_hab') and not col.startswith('idea_')]
        std_xh_not_idea = [col for col in df.columns if col.endswith('_std_por_hab') and not col.startswith('idea_')]
        drop_cols = ['geometry', 'localizacion'] + mean_not_idea + std_not_idea + mean_xh_not_idea + std_xh_not_idea
        drop_cols = [c for c in drop_cols if c in df.columns]
        new_df = df.drop(drop_cols, axis=1)
        return new_df

    def _nodes_union(self, tabu, temp, mdir, mndi):
        nodes = set()
        if self.NODE_ID_COL in tabu.columns:
            nodes |= set(tabu[self.NODE_ID_COL].unique())
        if {self.NODE_ID_COL, self.YEAR_COL}.issubset(temp.columns):
            nodes |= set(temp[self.NODE_ID_COL].unique())
        if not mdir.empty:
            nodes |= set(mdir[self.MDIR_SRC].unique())
            nodes |= set(mdir[self.MDIR_DST].unique())
        if not mndi.empty:
            nodes |= set(mndi[self.MNDI_U].unique())
            nodes |= set(mndi[self.MNDI_V].unique())
        nodes = sorted(nodes)
        node2idx = {n: i for i, n in enumerate(nodes)}
        return nodes, node2idx

    def _standardize(self, df):
        mu = df.mean(numeric_only=True)
        sd = df.std(numeric_only=True)
        return (df - mu) / (sd + self.EPS)

    def _build_node_features(self, tabu, temp, nodes):
        stat_cols = [c for c in tabu.columns if c != self.NODE_ID_COL]
        Xs = pd.DataFrame({self.NODE_ID_COL: nodes}).merge(
            tabu[[self.NODE_ID_COL] + stat_cols], on=self.NODE_ID_COL, how="left"
        ).set_index(self.NODE_ID_COL)
        if stat_cols:
            Xs[stat_cols] = Xs[stat_cols].fillna(Xs[stat_cols].mean())
            Xs[stat_cols] = self._standardize(Xs[stat_cols])

        Xt = pd.DataFrame(index=Xs.index)
        if not temp.empty:
            tdf = temp[temp[self.YEAR_COL] <= self.TARGET_YEAR].copy().sort_values([self.NODE_ID_COL, self.YEAR_COL])
            tcols = [c for c in tdf.columns if c not in (self.NODE_ID_COL, self.YEAR_COL)]
            if tcols:
                last_t = (
                    tdf.groupby(self.NODE_ID_COL, as_index=True)
                    .tail(1)[[self.NODE_ID_COL] + tcols]
                    .set_index(self.NODE_ID_COL)
                )
                Xt = pd.DataFrame(index=Xs.index).join(last_t, how="left")
                Xt[tcols] = Xt[tcols].fillna(Xt[tcols].mean())
                Xt[tcols] = self._standardize(Xt[tcols])

        X = pd.concat([Xs, Xt], axis=1).fillna(0.0).astype(np.float32)
        return torch.tensor(X.values, dtype=torch.float), list(X.columns)

    def _align_and_standardize_edge_attrs(self, df, cont_cols, binary_cols=None, extra_cols_keep=None):
        if binary_cols is None: binary_cols = []
        if extra_cols_keep is None: extra_cols_keep = []

        for c in cont_cols + binary_cols + extra_cols_keep:
            if c not in df.columns:
                df[c] = 0.0

        if cont_cols:
            Wc = df[cont_cols]
            Wc = (Wc - Wc.mean()) / (Wc.std() + self.EPS)
            df[cont_cols] = Wc.fillna(0.0)

        for c in binary_cols + extra_cols_keep:
            df[c] = df[c].fillna(0.0)

        keep_cols = ["u", "v"] + cont_cols + binary_cols + extra_cols_keep
        return df[keep_cols]

    def _build_pyg_data_mixed_directed(self, tabu, temp, mdir, mndi):
        nodes, node2idx = self._nodes_union(tabu, temp, mdir, mndi)
        x, node_feat_names = self._build_node_features(tabu, temp, nodes)

        frames = []

        # dirigidas
        if not mdir.empty:
            e_dir = mdir[[self.MDIR_SRC, self.MDIR_DST] + self.MDIR_WEIGHT_COLS].copy()
            e_dir["u"] = e_dir[self.MDIR_SRC].map(node2idx)
            e_dir["v"] = e_dir[self.MDIR_DST].map(node2idx)
            e_dir["edge_type"] = 0.0
            frames.append(e_dir[["u", "v"] + self.MDIR_WEIGHT_COLS + ["edge_type"]])

        # no dirigidas -> duplicar
        if not mndi.empty:
            base = mndi[[self.MNDI_U, self.MNDI_V] + self.MNDI_WEIGHT_COLS].copy()
            rev = base.rename(columns={self.MNDI_U: self.MNDI_V, self.MNDI_V: self.MNDI_U})
            e_und = pd.concat([base, rev], ignore_index=True)
            e_und["u"] = e_und[self.MNDI_U].map(node2idx)
            e_und["v"] = e_und[self.MNDI_V].map(node2idx)
            e_und["edge_type"] = 1.0
            frames.append(e_und[["u", "v"] + self.MNDI_WEIGHT_COLS + ["edge_type"]])

        if not frames:
            raise ValueError("No se han proporcionado aristas en mdir/mndi.")

        E = pd.concat(frames, ignore_index=True)
        E = E.dropna(subset=["u", "v"])
        E[["u", "v"]] = E[["u", "v"]].astype(int)
        E = E[E["u"] != E["v"]]
        E = E.drop_duplicates()

        cont_cols   = sorted(set(self.MDIR_WEIGHT_COLS + self.MNDI_WEIGHT_COLS))
        binary_cols = []             # mueve aquí los 0/1 si tienes
        extra_cols  = ["edge_type"]  # no estandarizar

        E = self._align_and_standardize_edge_attrs(E, cont_cols, binary_cols=binary_cols, extra_cols_keep=extra_cols)

        edge_index = torch.tensor(E[["u", "v"]].values.T, dtype=torch.long)
        edge_attr  = torch.tensor(E[cont_cols + binary_cols + extra_cols].values, dtype=torch.float)

        edge_is_undirected = torch.tensor(E["edge_type"].values == 1.0, dtype=torch.bool)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = x.size(0)
        data.edge_is_undirected = edge_is_undirected
        data.edge_type = torch.tensor(E["edge_type"].values, dtype=torch.float)
        data.edge_continuous_cols = cont_cols
        return data, nodes, node2idx, node_feat_names, (cont_cols + binary_cols + extra_cols)

    def run(self, tabu: pd.DataFrame, temp: pd.DataFrame, mdir: pd.DataFrame, mndi: pd.DataFrame):
        tabu, temp, mdir, mndi = global_prepro(tabu, temp, mdir, mndi, no_mad=self.no_mad, add_idea_emb=self.add_idea_emb)
        tabu_gae = self._prepro_tabular_data(tabu)
        data, nodes, node2idx, node_feat_names, edge_attr_names = self._build_pyg_data_mixed_directed(tabu_gae, temp, mdir, mndi)
        return data, nodes, node2idx, node_feat_names, edge_attr_names


# +------------------+
# |  TRAINING UTILS  |
# +------------------+

def grouped_undirected_split(edge_index, edge_is_undirected, num_nodes,
                             val_ratio=0.05, test_ratio=0.10, seed=42):
    """
    Split por grupos:
    - No dirigidas: (min(u,v), max(u,v)) comparten grupo (evita fuga entre direcciones).
    - Dirigidas: cada (u,v) es su propio grupo.
    Devuelve máscaras booleanas sobre aristas.
    """
    rng = np.random.default_rng(seed)
    u = edge_index[0].cpu().numpy()
    v = edge_index[1].cpu().numpy()
    und = edge_is_undirected.cpu().numpy().astype(bool)

    umin = np.minimum(u, v)
    umax = np.maximum(u, v)
    canon_id = umin * (num_nodes + 1) + umax

    group_id = np.arange(len(u))
    group_id[und] = canon_id[und]

    uniq_groups = np.unique(group_id)
    rng.shuffle(uniq_groups)

    n = len(uniq_groups)
    n_val = int(round(val_ratio * n))
    n_test = int(round(test_ratio * n))

    val_g = set(uniq_groups[:n_val])
    test_g = set(uniq_groups[n_val:n_val + n_test])
    train_g = set(uniq_groups[n_val + n_test:])

    train_mask = np.isin(group_id, list(train_g))
    val_mask   = np.isin(group_id, list(val_g))
    test_mask  = np.isin(group_id, list(test_g))

    return (
        torch.from_numpy(train_mask.astype(np.bool_)),
        torch.from_numpy(val_mask.astype(np.bool_)),
        torch.from_numpy(test_mask.astype(np.bool_)),
    )


def make_supervised_edge_splits(data: Data, train_mask, val_mask, test_mask):
    """
    Crea (train_data, val_data, test_data) con:
    - edge_index = SOLO aristas de entrenamiento (para el mensaje).
    - pos_edge_label_index = aristas del split correspondiente (para el objetivo).
    - edge_attr se submuestrea correspondiente.
    """
    def _subset_edge_index(mask):
        ei = data.edge_index[:, mask]
        ea = data.edge_attr[mask]
        return ei, ea

    ei_tr, ea_tr = _subset_edge_index(train_mask)
    pos_tr, ea_pos_tr = _subset_edge_index(train_mask)
    pos_val, ea_pos_val = _subset_edge_index(val_mask)
    pos_te, ea_pos_te = _subset_edge_index(test_mask)

    def _mk(split_pos, split_ea):
        d = Data(x=data.x.clone(), edge_index=ei_tr.clone(), edge_attr=ea_tr.clone())
        d.num_nodes = data.num_nodes
        d.pos_edge_label_index = split_pos.clone()
        d.pos_edge_attr = split_ea.clone()
        return d

    train_data = _mk(pos_tr, ea_pos_tr)
    val_data   = _mk(pos_val, ea_pos_val)
    test_data  = _mk(pos_te, ea_pos_te)
    return train_data, val_data, test_data


def pair_features_from_x(x: torch.Tensor, edge_index: torch.Tensor, mode: str = "cosine_l2_absdiff"):
    ''' Atributos de par densos para cualquier (u,v). '''
    u, v = edge_index
    xu, xv = x[u], x[v]

    xu_n = F.normalize(xu, p=2, dim=-1)
    xv_n = F.normalize(xv, p=2, dim=-1)
    cos_sim = (xu_n * xv_n).sum(dim=-1, keepdim=True)

    l2 = torch.norm(xu - xv, p=2, dim=-1, keepdim=True)
    mad = torch.mean(torch.abs(xu - xv), dim=-1, keepdim=True)

    if mode == "cosine_l2_absdiff":
        return torch.cat([cos_sim, l2, mad], dim=-1)
    else:
        raise ValueError(f"pair_features mode no soportado: {mode}")


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.0, mode="min", restore_best=True):
        assert mode in ("min","max")
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.restore_best = restore_best
        self.best_score, self.best_state, self.counter = None, None, 0
    def step(self, score, model):
        imp = False
        if self.best_score is None:
            imp = True
        else:
            if self.mode == "min":
                imp = (self.best_score - score) > self.min_delta
            else:
                imp = (score - self.best_score) > self.min_delta
        if imp:
            self.best_score = score
            self.counter = 0
            if self.restore_best: self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience
    def maybe_restore(self, model):
        if self.restore_best and (self.best_state is not None):
            model.load_state_dict(self.best_state)


# +---------+
# |  MODEL  |
# +---------+

class EdgeAwareGCNEncoder(nn.Module):
    """
    Encoder GNN (TransformerConv) que consume edge_attr multivariado.
    """
    def __init__(self, in_ch, edge_dim, hid=128, out=64, heads=2, dropout=0.2):
        super().__init__()
        self.conv1 = TransformerConv(in_ch, hid, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = TransformerConv(hid * heads, out, heads=1, edge_dim=edge_dim, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, drop_prob=0.2):
        if drop_prob > 0 and self.training:
            edge_index, edge_mask = dropout_edge(edge_index, p=drop_prob, training=self.training)
            if edge_attr is not None:
                edge_attr = edge_attr[edge_mask]
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x  # z


class EdgeRegressor(nn.Module):
    """
    Decoder para predecir un escalar o vector de atributos de arista (regresión).
    Puede concatenar "pair features" (cos, l2, |diff|) a [z_u, z_v].
    """
    def __init__(self, z_dim, out_dim=1, hid=128, use_pair_feats=True):
        super().__init__()
        self.use_pair_feats = use_pair_feats
        in_dim = 2*z_dim + (3 if use_pair_feats else 0)  # 3 = pair_features_from_x default
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

    def forward(self, z, edge_index, pair_feats=None):
        u, v = edge_index
        parts = [z[u], z[v]]
        if self.use_pair_feats and pair_feats is not None:
            parts.append(pair_feats)
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)  # [E, out_dim]


class NodeRegressor(nn.Module):
    """
    Decoder para predecir atributos nodales a partir de z_u.
    out_dim = nº de columnas en node_target_cols.
    """
    def __init__(self, z_dim, out_dim, hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, z):
        return self.mlp(z)  # [N, out_dim]



# +-----------+
# |  METRICS  |
# +-----------+

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rho sin SciPy: correlación de rangos (pandas rank).
    Si todo es constante, devuelve nan.
    """
    s_true = pd.Series(y_true).rank(method="average")
    s_pred = pd.Series(y_pred).rank(method="average")
    if s_true.nunique() < 2 or s_pred.nunique() < 2:
        return float('nan')
    return float(np.corrcoef(s_true, s_pred)[0,1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score(y_true, y_pred, multioutput="variance_weighted")


# +---------+
# |  TRAIN  |
# +---------+

def train_edge_node_multitask(
    data: Data, device,
    target_cols: list,                 # columnas de arista a predecir (en data.edge_continuous_cols)
    node_feat_names: list,             # lo devuelve PreproGTMAE.run(...)
    node_target_cols: list,            # columnas de X a predecir como tarea nodal
    hid=128, out=64, lr=1e-3, epochs=150,
    weight_decay=1e-4, dropout=0.2, heads=2, print_every=20,
    # Early stopping (elige métrica principal):
    patience=30, min_delta=0.0, monitor="val_edge_rmse", restore_best=True,
    # Split:
    val_ratio=0.2, test_ratio=0.2, seed=33,
    # Decoder y pair features:
    use_pair_feats=True, pair_mode="cosine_l2_absdiff",
    # Pérdidas:
    edge_loss_type="huber", edge_huber_delta=1.0,
    node_loss_type="huber", node_huber_delta=1.0,
    # Ranking opcional (edge):
    add_ranking=False, lambda_rank=0.5, margin=0.1,
    # Peso de la tarea nodal:
    lambda_node=1.0,
    # Dropout de aristas y *enmascarado de targets nodales* (para evitar copiar):
    edge_drop_prob=0.2, node_mask_rate=0.15,
    # Debug:
    dbg_print=True,
):
    # ---------- split de aristas ----------
    train_mask, val_mask, test_mask = grouped_undirected_split(
        data.edge_index, data.edge_is_undirected, num_nodes=data.num_nodes,
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    train_data, val_data, test_data = make_supervised_edge_splits(data, train_mask, val_mask, test_mask)

    # Mapear objetivos de ARISTA
    all_edge_cols = list(data.edge_continuous_cols) + ["edge_type"]
    edge_target_idx = [all_edge_cols.index(c) for c in target_cols]

    # Mapear objetivos de NODO sobre X
    node_target_idx = [node_feat_names.index(c) for c in node_target_cols]
    node_out_dim = len(node_target_idx)

    if dbg_print:
        print(f"[SPLIT] train={train_mask.sum().item()}  val={val_mask.sum().item()}  test={test_mask.sum().item()}")
        print(f"[EDGE TARGETS] {target_cols} -> idx {edge_target_idx}")
        print(f"[NODE TARGETS] {node_target_cols} -> idx {node_target_idx}")

    # ---------- modelo ----------
    class GTMultiModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = EdgeAwareGCNEncoder(
                data.num_features, edge_dim=data.edge_attr.size(1),
                hid=hid, out=out, dropout=dropout, heads=heads
            )
            self.edge_dec = EdgeRegressor(z_dim=out, out_dim=len(edge_target_idx), hid=hid, use_pair_feats=use_pair_feats)
            self.node_dec = NodeRegressor(z_dim=out, out_dim=node_out_dim, hid=hid)

    model = GTMultiModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Pérdidas
    def _make_reg_loss(kind, delta):
        if kind == "huber": return nn.SmoothL1Loss(beta=delta)
        if kind == "mse":   return nn.MSELoss()
        raise ValueError("loss_type debe ser 'huber' o 'mse'.")
    edge_loss_fn = _make_reg_loss(edge_loss_type, edge_huber_delta)
    node_loss_fn = _make_reg_loss(node_loss_type, node_huber_delta)

    # Tensors base (edge)
    def _to_dev(split):
        x  = split.x.to(device)                  # [N, F]
        ei = split.edge_index.to(device)         # [2, Etr]
        ea = split.edge_attr.to(device)          # [Etr, Fe]
        pos_ei = split.pos_edge_label_index.to(device)         # [2, Es]
        y_edge = split.pos_edge_attr.to(device)[:, edge_target_idx]  # [Es, Te]
        return x, ei, ea, pos_ei, y_edge

    x_tr, ei_tr, ea_tr, pos_ei_tr, y_edge_tr = _to_dev(train_data)
    x_va, ei_va, ea_va, pos_ei_va, y_edge_va = _to_dev(val_data)
    x_te, ei_te, ea_te, pos_ei_te, y_edge_te = _to_dev(test_data)

    # Targets nodales (a partir de X original ya estandarizada)
    def _node_targets(x_tensor):
        # x_tensor: [N, F]; extrae columnas node_target_idx
        return x_tensor[:, node_target_idx]  # [N, Tn]
    y_node_tr = _node_targets(x_tr).detach()
    y_node_va = _node_targets(x_va).detach()
    y_node_te = _node_targets(x_te).detach()

    # Early stopping
    es = EarlyStopper(
        patience=patience, min_delta=min_delta,
        mode=("min" if "rmse" in monitor or "mae" in monitor else "max"),
        restore_best=restore_best
    )

    # ---- helpers ranking (edge) ----
    def _ranking_loss(z, pos_ei, y_true, margin=0.1):
        with torch.no_grad():
            u = pos_ei[0].cpu().numpy()
            v = pos_ei[1].cpu().numpy()
            y = y_true.detach().cpu().numpy()
            if y.ndim == 2: y_scalar = y.mean(axis=1)
            else:           y_scalar = y
            df = pd.DataFrame({"u": u, "v": v, "y": y_scalar})
            pos_pairs = []
            for uu, grp in df.groupby("u"):
                if len(grp) < 2: continue
                v_pos = grp.sort_values("y", ascending=False).iloc[0]["v"]
                v_neg = grp.sort_values("y", ascending=True ).iloc[0]["v"]
                if v_pos == v_neg: continue
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

    # ---- máscara anti-fuga para targets nodales (opcional) ----
    def _mask_node_inputs(x_tensor, rate):
        if rate <= 0: return x_tensor
        x_masked = x_tensor.clone()
        if len(node_target_idx) > 0:
            m = torch.bernoulli(torch.full((x_tensor.size(0), len(node_target_idx)), 1.0 - rate, device=x_tensor.device))
            x_masked[:, node_target_idx] = x_masked[:, node_target_idx] * m
        return x_masked

    # --------------- TRAIN LOOP ---------------
    print("[TRAIN]")
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()

        # Enmascara SOLO columnas target nodales en la entrada de TRAIN
        x_tr_in = _mask_node_inputs(x_tr, node_mask_rate)

        # Encoder con aristas de TRAIN
        z = model.encoder(x_tr_in, ei_tr, ea_tr, drop_prob=edge_drop_prob)

        # ----- Edge head -----
        pf_tr = pair_features_from_x(x_tr_in, pos_ei_tr, mode=pair_mode) if use_pair_feats else None
        y_edge_hat = model.edge_dec(z, pos_ei_tr, pf_tr)      # [Es, Te]
        edge_reg_loss = edge_loss_fn(y_edge_hat, y_edge_tr)

        # ----- Node head -----
        y_node_hat = model.node_dec(z)                        # [N, Tn]
        node_reg_loss = node_loss_fn(y_node_hat, y_node_tr) if node_out_dim > 0 else torch.tensor(0.0, device=z.device)

        # ----- Ranking opcional (edge) -----
        if add_ranking:
            rank_loss = _ranking_loss(z, pos_ei_tr, y_edge_tr, margin=margin)
        else:
            rank_loss = torch.tensor(0.0, device=z.device)

        loss = edge_reg_loss + lambda_node * node_reg_loss + (lambda_rank * rank_loss if add_ranking else 0.0)
        loss.backward()
        opt.step()

        # ---- VALIDACIÓN ----
        model.eval()
        with torch.no_grad():
            z_va = model.encoder(x_va, ei_va, ea_va, drop_prob=0.0)

            # Edge val
            pf_va = pair_features_from_x(x_va, pos_ei_va, mode=pair_mode) if use_pair_feats else None
            y_edge_hat_va = model.edge_dec(z_va, pos_ei_va, pf_va)
            # Node val (sin máscara)
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
                node_val_rmse = node_val_mae = node_val_spr, node_val_r2 = float('nan')

        if (ep % print_every == 0) or (ep == 1):
            log = (f" [{ep:03d}] total={loss.item():.4f} | edge={edge_reg_loss.item():.4f} "
                   f"| node={node_reg_loss.item():.4f} | rank={rank_loss.item():.4f} || "
                   f"VAL edge(RMSE={edge_val_rmse:.4f}, MAE={edge_val_mae:.4f}, Sp={edge_val_spr:.4f}, R2={edge_val_r2:.4f})"
                   f"| node(RMSE={node_val_rmse:.4f}, MAE={node_val_mae:.4f}, Sp={node_val_spr:.4f}, R2={node_val_r2:.4f})")
            print(log)

        # ---- early stopping según 'monitor' ----
        if monitor == "val_edge_rmse": score, es.mode = edge_val_rmse, "min"
        elif monitor == "val_edge_mae": score, es.mode = edge_val_mae, "min"
        elif monitor == "val_edge_spr": score, es.mode = (edge_val_spr if not (edge_val_spr != edge_val_spr) else -1e9), "max"
        elif monitor == "val_node_rmse": score, es.mode = node_val_rmse, "min"
        elif monitor == "val_node_mae": score, es.mode = node_val_mae, "min"
        else: score, es.mode = (node_val_spr if not (node_val_spr != node_val_spr) else -1e9), "max"

        if es.step(score, model):
            if print_every:
                print(f"Early stopping en epoch {ep} (mejor {monitor}={es.best_score:.4f}).")
            break

    es.maybe_restore(model)

    # ---- Embeddings finales en el grafo COMPLETO ----
    model.eval()
    with torch.no_grad():
        Z = model.encoder(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)).cpu()

    # ---- TEST ----
    with torch.no_grad():
        # EDGE
        z_te = model.encoder(x_te, ei_te, ea_te, drop_prob=0.0)
        pf_te = pair_features_from_x(x_te, pos_ei_te, mode=pair_mode) if use_pair_feats else None
        y_edge_hat_te = model.edge_dec(z_te, pos_ei_te, pf_te)
        edge_test_rmse = _rmse(y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy())
        edge_test_mae = _mae (y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy())
        edge_test_spr = _spearman(y_edge_te.detach().cpu().numpy().ravel(), y_edge_hat_te.detach().cpu().numpy().ravel())
        edge_test_r2 = _r2(y_edge_te.detach().cpu().numpy(), y_edge_hat_te.detach().cpu().numpy())
        # NODE
        y_node_hat_te = model.node_dec(z_te) if node_out_dim > 0 else None
        if node_out_dim > 0:
            node_test_rmse = _rmse(y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy())
            node_test_mae = _mae (y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy())
            node_test_spr = _spearman(y_node_te.detach().cpu().numpy().ravel(), y_node_hat_te.detach().cpu().numpy().ravel())
            node_test_sr2 = _r2(y_node_te.detach().cpu().numpy(), y_node_hat_te.detach().cpu().numpy())
        else:
            node_test_rmse = node_test_mae = node_test_spr, node_test_sr2 = float('nan')

    if print_every:
        print("[TEST]")
        print(f" [EDGE] RMSE={edge_test_rmse:.6f} | MAE={edge_test_mae:.6f} | Sp={edge_test_spr:.6f} | Sp={edge_test_r2:.6f}")
        print(f" [NODE] RMSE={node_test_rmse:.6f} | MAE={node_test_mae:.6f} | Sp={node_test_spr:.6f} | Sp={node_test_sr2:.6f}")

    return model, Z
