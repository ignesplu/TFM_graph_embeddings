from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch_geometric.nn import TransformerConv

from .utils import global_prepro


# +-----------------+
# |  PREPROCESSING  |
# +-----------------+


@dataclass
class Preprocessed:
    """
    Container for preprocessed graph and temporal data.
    
    Attributes:
        node_index: Mapping from node identifiers to indices
        X_static: Static node features tensor
        X_temp: Temporal node features tensor
        M_temp: Missing value mask for temporal features
        years: List of years in temporal data
        edge_index_dir: Directed edge indices
        edge_attr_dir: Directed edge attributes
        edge_index_und: Undirected edge indices
        edge_attr_und: Undirected edge attributes
    """
    node_index: Dict[str, int]
    X_static: torch.Tensor
    X_temp: torch.Tensor
    M_temp: Optional[torch.Tensor]
    years: List[int]
    edge_index_dir: torch.Tensor
    edge_attr_dir: Optional[torch.Tensor]
    edge_index_und: torch.Tensor
    edge_attr_und: Optional[torch.Tensor]


class PreproTGT:
    """
    Preprocessor for Temporal Graph Transformer (TGT) model.
    
    Handles data preprocessing, feature engineering, and graph construction
    from tabular, temporal, and edge data for TGT model training.
    """

    def __init__(
        self,
        add_idea_emb: bool = True,
        no_mad: bool = False,
    ):
        """
        Initialize TGT preprocessor.
        
        Args:
            add_idea_emb: Whether to add idea embeddings
            no_mad: Whether to exclude Madrid from data
        """
        self.add_idea_emb = add_idea_emb
        self.no_mad = no_mad

        self.EPS = 1e-8

    def _prepro_tabular_data(self, tabu: pd.DataFrame, temp: pd.DataFrame):
        """
        Preprocess tabular and temporal data for TGT model.
        
        Args:
            tabu: Tabular data DataFrame
            temp: Temporal data DataFrame
            
        Returns:
            Tuple of processed (tabular, temporal) DataFrames
        """
        temp_cols = set(
            ["cc", "year", "geo_dens_poblacion", "y_edad_media", "p_feminidad"]
            + [
                col
                for col in temp.columns
                if (col.endswith("por_hab") | col.endswith("_xhab"))
            ]
        ) - set(["p_feminidad_por_hab", "y_edad_media_por_hab"])

        temp_tgt = temp[list(temp_cols)]

        tabu_idea_cols = [col for col in tabu.columns if col.startswith("idea_")]
        tabu_colinda_cols = [
            col for col in tabu.columns if col.startswith("colinda_con")
        ]
        tabu_other_cols = [
            "cc",
            "superficie",
            "altitud",
            "geo_distancia_capital",
            "n_viviendas_totales_por_hab",
        ]

        tabu_tgt = tabu[tabu_idea_cols + tabu_colinda_cols + tabu_other_cols]

        return tabu_tgt, temp_tgt

    def _find_cols(
        self, df: pd.DataFrame, preferred: Tuple[str, str]
    ) -> Tuple[str, str]:
        """
        Find source and target columns in edge DataFrame.
        
        Args:
            df: Edge DataFrame
            preferred: Preferred column names to try first
            
        Returns:
            Tuple of (source_column, target_column)
        """
        candidates = [
            preferred,
            ("src", "dst"),
            ("source", "target"),
            ("cc_src", "cc_dst"),
            ("cc_origen", "cc_destino"),
            ("i", "j"),
            ("u", "v"),
        ]
        cols = set(df.columns)
        for a, b in candidates:
            if a in cols and b in cols:
                return a, b
        raise ValueError("Could not infer source/target columns in edges DataFrame.")

    def _build_node_index(
        self, tabu: pd.DataFrame, cc_col: str = "cc"
    ) -> Dict[str, int]:
        """
        Build mapping from node identifiers to indices.
        
        Args:
            tabu: Tabular data DataFrame
            cc_col: Node identifier column name
            
        Returns:
            Dictionary mapping node IDs to indices
        """
        if cc_col not in tabu.columns and tabu.index.name == cc_col:
            cc_values = tabu.index.astype(str).tolist()
        else:
            cc_values = tabu[cc_col].astype(str).tolist()
        cc_values = sorted(set(cc_values))
        return {cc: i for i, cc in enumerate(cc_values)}

    def _extract_static_features(
        self, tabu: pd.DataFrame, node_index: Dict[str, int], cc_col: str = "cc"
    ) -> torch.Tensor:
        """
        Extract static node features from tabular data.
        
        Args:
            tabu: Tabular data DataFrame
            node_index: Node ID to index mapping
            cc_col: Node identifier column name
            
        Returns:
            Tensor of static node features
        """
        if cc_col in tabu.columns:
            df = tabu.copy()
        else:
            df = tabu.reset_index().rename(columns={tabu.index.name: cc_col})
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if cc_col in num_cols:
            num_cols.remove(cc_col)
        df = df[[cc_col] + num_cols]
        df[cc_col] = df[cc_col].astype(str)
        N = len(node_index)
        D = len(num_cols)
        X = np.zeros((N, D), dtype=np.float32)
        for _, row in df.iterrows():
            cc = row[cc_col]
            if cc in node_index:
                X[node_index[cc], :] = row[num_cols].to_numpy(
                    dtype=np.float32, copy=True
                )
        return torch.from_numpy(X)

    def _extract_temporal_tensor(
        self,
        temp: pd.DataFrame,
        node_index: Dict[str, int],
        cc_col: str = "cc",
        year_col: str = "year",
        add_missing_mask: bool = True,
        impute: Optional[str] = "ffill",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int], List[str]]:
        """
        Extract temporal node features as 3D tensor.
        
        Args:
            temp: Temporal data DataFrame
            node_index: Node ID to index mapping
            cc_col: Node identifier column name
            year_col: Year column name
            add_missing_mask: Whether to create missing value mask
            impute: Imputation strategy ('ffill', 'bfill', or None)
            
        Returns:
            Tuple of (temporal features, missing mask, years, feature names)
        """
        df = temp.copy()
        df[cc_col] = df[cc_col].astype(str)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if year_col not in num_cols:
            num_cols = [c for c in num_cols if c != year_col]
        feature_cols = [c for c in num_cols if c != year_col]

        years = sorted(df[year_col].dropna().unique().tolist())
        T = len(years)
        N = len(node_index)
        D = len(feature_cols)
        X = np.zeros((T, N, D), dtype=np.float32)
        mask = np.zeros((T, N, D), dtype=np.float32) if add_missing_mask else None

        for d, feat in enumerate(feature_cols):
            sub = df[[cc_col, year_col, feat]].copy()
            if impute in ("ffill", "bfill"):
                sub = sub.sort_values([cc_col, year_col])
                sub[feat] = (
                    sub.groupby(cc_col)[feat].ffill()
                    if impute == "ffill"
                    else sub.groupby(cc_col)[feat].bfill()
                )
            for t_idx, y in enumerate(years):
                ydf = sub[sub[year_col] == y]
                for _, r in ydf.iterrows():
                    cc = r[cc_col]
                    if cc not in node_index:
                        continue
                    i = node_index[cc]
                    val = r[feat]
                    if pd.notna(val):
                        X[t_idx, i, d] = float(val)
                        if mask is not None:
                            mask[t_idx, i, d] = 1.0

        X_t = torch.from_numpy(X)
        mask_t = torch.from_numpy(mask) if mask is not None else None
        return X_t, mask_t, years, feature_cols

    def _standardize_edge_features(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        binary_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Standardize edge features while preserving binary columns.
        
        Args:
            df: Edge DataFrame
            feat_cols: Feature columns to standardize
            binary_cols: Binary columns to preserve
            
        Returns:
            DataFrame with standardized features
        """
        if not feat_cols:
            return df
        binary_cols = binary_cols or []
        cont_cols = [c for c in feat_cols if c not in binary_cols]
        if cont_cols:
            mu = df[cont_cols].mean()
            sd = df[cont_cols].std().replace(0.0, 1.0)
            df[cont_cols] = (df[cont_cols] - mu) / (sd + self.EPS)
        for c in feat_cols:
            df[c] = df[c].fillna(0.0)
        return df

    def _build_edges(
        self,
        mdir: pd.DataFrame,
        mndi: pd.DataFrame,
        node_index: Dict[str, int],
        prefer_cols: Tuple[str, str] = ("src", "dst"),
        binary_cols_dir: Optional[List[str]] = None,
        binary_cols_und: Optional[List[str]] = None,
    ):
        """
        Build directed and undirected edge tensors.
        
        Args:
            mdir: Directed edge DataFrame
            mndi: Undirected edge DataFrame
            node_index: Node ID to index mapping
            prefer_cols: Preferred column names
            binary_cols_dir: Binary columns for directed edges
            binary_cols_und: Binary columns for undirected edges
            
        Returns:
            Tuple of (directed edges, directed attributes, undirected edges, undirected attributes)
        """
        # --- Directed ---
        edge_index_dir = torch.empty((2, 0), dtype=torch.long)
        edge_attr_dir = None
        feat_cols_dir: List[str] = []

        if mdir is not None and not mdir.empty:
            s_col, t_col = self._find_cols(mdir, prefer_cols)
            df = mdir.copy()
            df[s_col] = df[s_col].astype(str)
            df[t_col] = df[t_col].astype(str)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feat_cols_dir = [c for c in num_cols if c not in (s_col, t_col)]

            # eliminar lazos
            df = df[df[s_col] != df[t_col]].copy()
            # estandarizar por relación
            df = self._standardize_edge_features(
                df, feat_cols_dir, binary_cols=binary_cols_dir
            )

            # map a índices y filtrar nodos desconocidos
            df = df[df[s_col].isin(node_index) & df[t_col].isin(node_index)]
            src = df[s_col].map(node_index).astype(int).to_numpy()
            dst = df[t_col].map(node_index).astype(int).to_numpy()

            # eliminar duplicados exactos (incluyendo atributos)
            if feat_cols_dir:
                df = df.drop_duplicates(subset=[s_col, t_col] + feat_cols_dir)
            else:
                df = df.drop_duplicates(subset=[s_col, t_col])

            src = torch.as_tensor(df[s_col].map(node_index).values, dtype=torch.long)
            dst = torch.as_tensor(df[t_col].map(node_index).values, dtype=torch.long)
            edge_index_dir = torch.stack([src, dst], dim=0)

            edge_attr_dir = (
                torch.tensor(
                    df[feat_cols_dir].to_numpy(np.float32), dtype=torch.float32
                )
                if feat_cols_dir
                else None
            )

        # --- Undirected (duplicar ambas direcciones si falta la inversa) ---
        edge_index_und = torch.empty((2, 0), dtype=torch.long)
        edge_attr_und = None
        feat_cols_u: List[str] = []

        if mndi is not None and not mndi.empty:
            s_col2, t_col2 = self._find_cols(mndi, prefer_cols)
            dfu = mndi.copy()
            dfu[s_col2] = dfu[s_col2].astype(str)
            dfu[t_col2] = dfu[t_col2].astype(str)
            num_cols_u = dfu.select_dtypes(include=[np.number]).columns.tolist()
            feat_cols_u = [c for c in num_cols_u if c not in (s_col2, t_col2)]

            # eliminar lazos
            dfu = dfu[dfu[s_col2] != dfu[t_col2]].copy()

            # duplicar si falta inversa
            existing_pairs = set(zip(dfu[s_col2], dfu[t_col2]))
            # filas originales
            rows = [dfu]
            # inversas que faltan
            missing_rev_mask = (
                ~dfu[[t_col2, s_col2]].apply(tuple, axis=1).isin(existing_pairs)
            )
            if missing_rev_mask.any():
                rev = dfu.loc[missing_rev_mask].rename(
                    columns={s_col2: t_col2, t_col2: s_col2}
                )
                rows.append(rev)
            dfu2 = pd.concat(rows, ignore_index=True)

            # estandarizar por relación (sobre dfu2 para que ambas direcciones queden igual)
            dfu2 = self._standardize_edge_features(
                dfu2, feat_cols_u, binary_cols=binary_cols_und
            )

            # filtrar nodos desconocidos
            dfu2 = dfu2[dfu2[s_col2].isin(node_index) & dfu2[t_col2].isin(node_index)]

            # eliminar duplicados exactos
            if feat_cols_u:
                dfu2 = dfu2.drop_duplicates(subset=[s_col2, t_col2] + feat_cols_u)
            else:
                dfu2 = dfu2.drop_duplicates(subset=[s_col2, t_col2])

            src_u = torch.as_tensor(
                dfu2[s_col2].map(node_index).values, dtype=torch.long
            )
            dst_u = torch.as_tensor(
                dfu2[s_col2].map(node_index).values, dtype=torch.long
            )
            edge_index_und = torch.stack([src_u, dst_u], dim=0)

            edge_attr_und = (
                torch.tensor(
                    dfu2[feat_cols_u].to_numpy(np.float32), dtype=torch.float32
                )
                if feat_cols_u
                else None
            )

        return edge_index_dir, edge_attr_dir, edge_index_und, edge_attr_und

    def run(
        self,
        tabu: pd.DataFrame,
        temp: pd.DataFrame,
        mdir: pd.DataFrame,
        mndi: pd.DataFrame,
        cc_col="cc",
        year_col="year",
        binary_cols_dir: Optional[List[str]] = None,
        binary_cols_und: Optional[List[str]] = None,
    ) -> Preprocessed:
        """
        Execute the full preprocessing pipeline.
        
        Args:
            tabu: Tabular data
            temp: Temporal data
            mdir: Directed edge data
            mndi: Undirected edge data
            cc_col: Node identifier column
            year_col: Year column
            binary_cols_dir: Binary columns for directed edges
            binary_cols_und: Binary columns for undirected edges
            
        Returns:
            Preprocessed data container
        """

        tabu, temp, mdir, mndi = global_prepro(
            tabu, temp, mdir, mndi, no_mad=self.no_mad, add_idea_emb=self.add_idea_emb
        )
        tabu, temp = self._prepro_tabular_data(tabu, temp)
        node_index = self._build_node_index(tabu, cc_col=cc_col)
        X_static = self._extract_static_features(tabu, node_index, cc_col=cc_col)
        X_temp, M_temp, years, _ = self._extract_temporal_tensor(
            temp,
            node_index,
            cc_col=cc_col,
            year_col=year_col,
            add_missing_mask=True,
            impute="ffill",
        )
        e_dir, ea_dir, e_und, ea_und = self._build_edges(
            mdir,
            mndi,
            node_index,
            binary_cols_dir=binary_cols_dir,
            binary_cols_und=binary_cols_und,
        )

        return Preprocessed(
            node_index, X_static, X_temp, M_temp, years, e_dir, ea_dir, e_und, ea_und
        )


# +---------+
# |  MODEL  |
# +---------+


def sinusoidal_time_encoding(
    years: List[int], ref_year: int = 2022, d_model: int = 32
) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings for temporal data.
    
    Args:
        years: List of years
        ref_year: Reference year for relative encoding
        d_model: Encoding dimension
        
    Returns:
        Sinusoidal positional encodings
    """
    rel = torch.tensor([y - ref_year for y in years], dtype=torch.float32).unsqueeze(
        1
    )  # [T,1]
    i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)  # [1,d]
    div = torch.pow(10000, (2 * (i // 2)) / d_model)
    pe = rel / div
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe  # [T,d_model]


def exponential_weights(
    years: List[int], target_year: int = 2022, decay: float = 0.35
) -> torch.Tensor:
    """
    Compute exponential decay weights for temporal aggregation.
    
    Args:
        years: List of years
        target_year: Target year for weighting
        decay: Decay rate
        
    Returns:
        Weight tensor for temporal aggregation
    """
    gaps = torch.tensor([max(0, target_year - y) for y in years], dtype=torch.float32)
    w = torch.exp(-decay * gaps)
    w = w / (w.sum() + 1e-8)
    return w


@dataclass
class TGTConfig:
    """
    Configuration for Temporal Graph Transformer model.
    
    Attributes:
        hidden: Hidden dimension
        heads: Number of attention heads
        edge_drop: Edge dropout rate
        dropout: General dropout rate
        time_enc_dim: Time encoding dimension
        tf_layers: Number of transformer layers
        tf_ff: Transformer feedforward dimension
        tf_heads: Transformer attention heads
    """
    hidden: int = 128
    heads: int = 4
    edge_drop: float = 0.0
    dropout: float = 0.1
    time_enc_dim: int = 32
    tf_layers: int = 2
    tf_ff: int = 256
    tf_heads: int = 4


class HeteroSpatialEncoder(nn.Module):
    """
    Heterogeneous spatial encoder with relation-specific convolutions.
    
    Uses separate TransformerConv blocks for directed and undirected edges,
    then fuses them through concatenation and linear transformation.
    """

    def __init__(
        self,
        in_dim: int,
        e_dir_dim: int = 0,
        e_und_dim: int = 0,
        hidden: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize heterogeneous spatial encoder.
        
        Args:
            in_dim: Input feature dimension
            e_dir_dim: Directed edge attribute dimension
            e_und_dim: Undirected edge attribute dimension
            hidden: Hidden dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.conv_dir = TransformerConv(
            in_dim,
            hidden,
            heads=heads,
            dropout=dropout,
            edge_dim=(e_dir_dim if e_dir_dim > 0 else None),
            beta=False,
            root_weight=True,
        )
        self.conv_und = TransformerConv(
            in_dim,
            hidden,
            heads=heads,
            dropout=dropout,
            edge_dim=(e_und_dim if e_und_dim > 0 else None),
            beta=False,
            root_weight=True,
        )
        self.lin = nn.Linear(2 * hidden * heads, hidden)  # fuse

    def forward(
        self,
        x: torch.Tensor,
        edge_index_dir: torch.Tensor,
        edge_attr_dir: Optional[torch.Tensor],
        edge_index_und: torch.Tensor,
        edge_attr_und: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with heterogeneous edge processing.
        
        Args:
            x: Node features
            edge_index_dir: Directed edge indices
            edge_attr_dir: Directed edge attributes
            edge_index_und: Undirected edge indices
            edge_attr_und: Undirected edge attributes
            
        Returns:
            Spatially encoded node representations
        """
        h_dir = self.conv_dir(x, edge_index_dir, edge_attr_dir)  # [N, hidden*heads]
        h_und = self.conv_und(x, edge_index_und, edge_attr_und)  # [N, hidden*heads]
        h = torch.cat([h_dir, h_und], dim=-1)
        return torch.relu(self.lin(h))


class TemporalGraphTransformer(nn.Module):
    """
    Temporal Graph Transformer for spatio-temporal graph data.
    
    Combines spatial graph processing with temporal transformer
    for joint spatio-temporal representation learning.
    """

    def __init__(
        self,
        x_static_dim: int,
        x_temp_dim: int,
        e_dir_dim: int,
        e_und_dim: int,
        cfg: TGTConfig = TGTConfig(),
    ):
        """
        Initialize Temporal Graph Transformer.
        
        Args:
            x_static_dim: Static feature dimension
            x_temp_dim: Temporal feature dimension
            e_dir_dim: Directed edge attribute dimension
            e_und_dim: Undirected edge attribute dimension
            cfg: Model configuration
        """
        super().__init__()
        self.cfg = cfg
        in_dim = (
            x_static_dim + x_temp_dim + x_temp_dim
        )  # include mask as extra x_temp_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden)
        self.spatial = HeteroSpatialEncoder(
            cfg.hidden,
            e_dir_dim,
            e_und_dim,
            hidden=cfg.hidden,
            heads=cfg.heads,
            dropout=cfg.dropout,
        )
        self.time_proj = nn.Linear(cfg.time_enc_dim, cfg.hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden,
            nhead=cfg.tf_heads,
            dim_feedforward=cfg.tf_ff,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.temporal_tf = nn.TransformerEncoder(enc_layer, num_layers=cfg.tf_layers)
        self.readout = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(
        self,
        X_static: torch.Tensor,  # [N, D_s]
        X_temp: torch.Tensor,  # [T, N, D_temp]
        M_temp: Optional[torch.Tensor],  # [T, N, D_temp] or None
        years: List[int],
        edge_index_dir: torch.Tensor,
        edge_attr_dir: Optional[torch.Tensor],
        edge_index_und: torch.Tensor,
        edge_attr_und: Optional[torch.Tensor],
        target_year: int = 2022,
        decay: float = 0.35,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with spatio-temporal processing.
        
        Args:
            X_static: Static node features
            X_temp: Temporal node features
            M_temp: Missing value mask
            years: List of years
            edge_index_dir: Directed edge indices
            edge_attr_dir: Directed edge attributes
            edge_index_und: Undirected edge indices
            edge_attr_und: Undirected edge attributes
            target_year: Target year for weighting
            decay: Decay rate for temporal weighting
            
        Returns:
            Tuple of (node embeddings, temporal weights)
        """
        device = X_static.device
        T, N, D_temp = X_temp.shape
        if M_temp is None:
            M_temp = torch.zeros_like(X_temp)

        # Build per-year spatial encodings
        H_list = []
        for t in range(T):
            xt = torch.cat(
                [X_static, X_temp[t], M_temp[t]], dim=-1
            )  # [N, D_s + 2*D_temp]
            xt = self.input_proj(xt)  # [N, hidden]
            ht = self.spatial(
                xt, edge_index_dir, edge_attr_dir, edge_index_und, edge_attr_und
            )  # [N, hidden]
            H_list.append(ht)
        H = torch.stack(H_list, dim=0)  # [T, N, hidden]

        # Temporal encoding
        pe = sinusoidal_time_encoding(
            years, ref_year=target_year, d_model=self.cfg.time_enc_dim
        ).to(
            device
        )  # [T, d_t]
        pe_h = self.time_proj(pe).unsqueeze(1).repeat(1, N, 1)  # [T, N, hidden]
        H = H + pe_h

        # Transformer over time (batch is nodes): rearrange to [N, T, hidden]
        H_bt = H.permute(1, 0, 2)  # [N, T, hidden]
        H_out = self.temporal_tf(H_bt)  # [N, T, hidden]

        # Weighted aggregation to target year
        w = exponential_weights(years, target_year=target_year, decay=decay).to(
            device
        )  # [T]
        z = (H_out * w.view(1, -1, 1)).sum(dim=1)  # [N, hidden]
        z = self.readout(z)
        return z, w  # [N, hidden], [T]


def compute_tgt_embeddings(
    prep: Preprocessed,
    target_year: int = 2022,
    device: Optional[torch.device] = None,
    cfg: TGTConfig = TGTConfig(),
    decay: float = 0.35,
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    """
    Compute Temporal Graph Transformer embeddings for preprocessed data.
    
    Args:
        prep: Preprocessed data container
        target_year: Target year for temporal weighting
        device: Computation device
        cfg: Model configuration
        decay: Decay rate for temporal weighting
        
    Returns:
        Tuple of (embeddings, years, weights)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D_s = prep.X_static.size(1)
    D_temp = prep.X_temp.size(2)
    e_dir_dim = 0 if prep.edge_attr_dir is None else prep.edge_attr_dir.size(1)
    e_und_dim = 0 if prep.edge_attr_und is None else prep.edge_attr_und.size(1)

    model = TemporalGraphTransformer(D_s, D_temp, e_dir_dim, e_und_dim, cfg=cfg).to(
        device
    )
    Xs = prep.X_static.to(device)
    Xt = prep.X_temp.to(device)
    Mt = prep.M_temp.to(device) if prep.M_temp is not None else None
    eidir = prep.edge_index_dir.to(device)
    eeadir = prep.edge_attr_dir.to(device) if prep.edge_attr_dir is not None else None
    eiund = prep.edge_index_und.to(device)
    eeaund = prep.edge_attr_und.to(device) if prep.edge_attr_und is not None else None

    Z, w = model(
        Xs,
        Xt,
        Mt,
        prep.years,
        eidir,
        eeadir,
        eiund,
        eeaund,
        target_year=target_year,
        decay=decay,
    )
    return Z.detach().cpu(), prep.years, w.detach().cpu()
