from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from .utils import global_prepro


# +-----------------+
# |  PREPROCESSING  |
# +-----------------+


@dataclass
class GraphInputs:
    """
    Container for heterogeneous graph snapshots across multiple years.

    Attributes:
        data_per_year: Dictionary mapping years to HeteroData snapshots
        years_sorted: Chronologically sorted list of years
        node_index: Mapping from node identifiers to indices
    """

    data_per_year: Dict[int, HeteroData]
    years_sorted: List[int]
    node_index: Dict[str, int]


class PreproHGTTE:
    """
    Preprocessor for Heterogeneous Graph Transformer Temporal Encoded (HGT-TE).

    Handles data preprocessing for spatio-temporal heterogeneous graph data,
    creating yearly snapshots with node features and edge relationships.
    """

    def __init__(self, add_idea_emb: bool = True, no_mad: bool = False):
        """
        Initialize HGT-TE preprocessor.

        Args:
            add_idea_emb: Whether to add idea embeddings
            no_mad: Whether to exclude Madrid from data
        """
        self.add_idea_emb = add_idea_emb
        self.no_mad = no_mad

    def _prepro_tabular_data(self, tabu, temp):
        """
        Preprocess tabular and temporal data for HGT-TE model.

        Args:
            tabu: Tabular data DataFrame
            temp: Temporal data DataFrame

        Returns:
            Tuple of processed (tabular, temporal) DataFrames
        """
        temp_cols = set(
            ["cc", "year", "geo_dens_poblacion", "y_edad_media", "p_feminidad"]
            + [col for col in temp.columns if (col.endswith("por_hab") | col.endswith("_xhab"))]
        ) - set(["p_feminidad_por_hab", "y_edad_media_por_hab"])

        temp_hgtte = temp[list(temp_cols)]
        temp_hgtte.loc[:, "cc"] = temp_hgtte["cc"].astype("string")

        tabu_idea_cols = [col for col in tabu.columns if col.startswith("idea_")]
        tabu_colinda_cols = [col for col in tabu.columns if col.startswith("colinda_con")]
        tabu_other_cols = [
            "cc",
            "superficie",
            "altitud",
            "geo_distancia_capital",
            "n_viviendas_totales_por_hab",
        ]

        tabu_hgtte = tabu[tabu_idea_cols + tabu_colinda_cols + tabu_other_cols]
        tabu_hgtte.loc[:, "cc"] = tabu_hgtte["cc"].astype("string")

        return tabu_hgtte, temp_hgtte

    def _build_index_from_tabu(self, tabu_df, node_id_col: str = "cc") -> Dict[str, int]:
        """
        Build stable mapping from node identifiers to indices.

        Args:
            tabu_df: Tabular data DataFrame
            node_id_col: Node identifier column name

        Returns:
            Dictionary mapping node IDs to indices
        """
        ids = sorted(map(str, tabu_df[node_id_col].unique().tolist()))
        return {cc: i for i, cc in enumerate(ids)}

    def _zscore(self, x: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Apply Z-score normalization to tensor.

        Args:
            x: Input tensor
            eps: Epsilon for numerical stability

        Returns:
            Z-score normalized tensor
        """
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, unbiased=False, keepdim=True)
        return (x - mu) / (sd + eps)

    def _forward_fill_per_row(self, mat: Tensor, mask: Tensor) -> Tensor:
        """
        Apply forward fill imputation along temporal axis.

        Args:
            mat: 3D tensor [T, N, D] of values
            mask: 3D tensor [T, N, D] of observation indicators

        Returns:
            Forward-filled tensor
        """
        T, N, D = mat.shape
        out = mat.clone()
        last = torch.zeros((N, D), device=mat.device)
        has_last = torch.zeros((N, D), dtype=torch.bool, device=mat.device)
        for t in range(T):
            m = mask[t].bool()
            last = torch.where(m, out[t], last)
            has_last = torch.where(m, torch.ones_like(has_last), has_last)
            out[t] = torch.where(m, out[t], torch.where(has_last, last, out[t]))
        return out

    def _pandas_to_tensor(
        self, df, cols: List[str], index_map: Dict[str, int], id_col: str
    ) -> Tensor:
        """
        Convert pandas DataFrame columns to tensor format.

        Args:
            df: Input DataFrame
            cols: Columns to extract
            index_map: Node ID to index mapping
            id_col: Identifier column name

        Returns:
            Feature tensor
        """
        N, D = len(index_map), len(cols)
        X = torch.zeros((N, D), dtype=torch.float32)

        for _, row in df.iterrows():
            idx = index_map[str(row[id_col])]
            X[idx] = torch.tensor(row[cols].to_numpy(dtype=np.float32))
        return X

    def _build_hetero_snapshots(
        self,
        device,
        tabu_df,
        temp_df,
        mdir_df,
        mndi_df,
        node_id_col: str = "cc",
        src_col: str = "src",
        dst_col: str = "dst",
        year_col: str = "year",
        static_cols: Optional[List[str]] = None,
        temp_cols: Optional[List[str]] = None,
        edge_attr_cols_dir: Optional[List[str]] = None,
        edge_attr_cols_undir: Optional[List[str]] = None,
    ) -> GraphInputs:
        """
        Build heterogeneous graph snapshots for each year.

        Args:
            device: Computation device
            tabu_df: Tabular data
            temp_df: Temporal data
            mdir_df: Directed edge data
            mndi_df: Undirected edge data
            node_id_col: Node identifier column
            src_col: Source column for edges
            dst_col: Destination column for edges
            year_col: Year column
            static_cols: Static feature columns
            temp_cols: Temporal feature columns
            edge_attr_cols_dir: Directed edge attribute columns
            edge_attr_cols_undir: Undirected edge attribute columns

        Returns:
            GraphInputs container with yearly snapshots
        """

        index_map = self._build_index_from_tabu(tabu_df, node_id_col)
        years_sorted = sorted(temp_df[year_col].unique().tolist())
        N = len(index_map)

        # Static atts
        if static_cols is None:
            static_cols = [c for c in tabu_df.columns if c not in (node_id_col,)]
        X_static = self._pandas_to_tensor(
            tabu_df[[node_id_col] + static_cols], static_cols, index_map, node_id_col
        )
        X_static = self._zscore(X_static)

        # Temporal (+mask) atts
        if temp_cols is None:
            temp_cols = [c for c in temp_df.columns if c not in (node_id_col, year_col)]

        # Build tensors [T, N, D]
        T = len(years_sorted)
        Dtemp = len(temp_cols)
        X_temp = torch.zeros((T, N, Dtemp), dtype=torch.float32)
        M_temp = torch.zeros((T, N, Dtemp), dtype=torch.float32)

        # pivot by year
        for ti, y in enumerate(years_sorted):
            sub = temp_df[temp_df[year_col] == y]

            for _, row in sub.iterrows():
                cc = str(row[node_id_col])
                if cc not in index_map:
                    continue
                nidx = index_map[cc]
                vals = []
                mask_vals = []
                for c in temp_cols:
                    val = row[c]
                    if pd.isna(val):
                        vals.append(0.0)
                        mask_vals.append(0.0)
                    else:
                        vals.append(float(val))
                        mask_vals.append(1.0)
                X_temp[ti, nidx] = torch.tensor(vals, dtype=torch.float32)
                M_temp[ti, nidx] = torch.tensor(mask_vals, dtype=torch.float32)

        X_temp = self._forward_fill_per_row(X_temp, M_temp)
        for ti in range(T):
            X_temp[ti] = self._zscore(X_temp[ti])

        # Edges
        def df_to_edges(df, attr_cols: Optional[List[str]]) -> Tuple[Tensor, Optional[Tensor]]:
            E = df.shape[0]
            edge_index = torch.zeros((2, E), dtype=torch.long)
            edge_attr = None
            if attr_cols:
                edge_attr = torch.zeros((E, len(attr_cols)), dtype=torch.float32)
            for i, row in enumerate(df.itertuples(index=False)):
                s = index_map[str(getattr(row, src_col))]
                d = index_map[str(getattr(row, dst_col))]
                edge_index[0, i] = s
                edge_index[1, i] = d
                if attr_cols:
                    edge_attr[i] = torch.tensor(
                        [float(getattr(row, c)) for c in attr_cols], dtype=torch.float32
                    )

            if edge_attr is not None and edge_attr.numel() > 0:
                edge_attr = self._zscore(edge_attr)
            return edge_index, edge_attr

        if edge_attr_cols_dir is None:
            edge_attr_cols_dir = [c for c in mdir_df.columns if c not in (src_col, dst_col)]
        if edge_attr_cols_undir is None:
            edge_attr_cols_undir = [c for c in mndi_df.columns if c not in (src_col, dst_col)]

        # Directed
        eidx_dir, eattr_dir = df_to_edges(mdir_df, edge_attr_cols_dir)

        # Non-directed
        eidx_u, eattr_u = df_to_edges(mndi_df, edge_attr_cols_undir)
        eidx_undir = torch.cat([eidx_u, eidx_u.flip(0)], dim=1)
        eattr_undir = torch.cat([eattr_u, eattr_u], dim=0) if eattr_u is not None else None

        def edge_attr_to_weight(eattr: Optional[Tensor]) -> Optional[Tensor]:
            if eattr is None:
                return None
            w = eattr.mean(dim=1, keepdim=True)  # simple: media como peso
            return w.squeeze(-1)

        w_dir = edge_attr_to_weight(eattr_dir)
        w_undir = edge_attr_to_weight(eattr_undir)

        # snapshots HeteroData by year
        data_per_year: Dict[int, HeteroData] = {}
        for ti, y in enumerate(years_sorted):
            data = HeteroData()

            xt = torch.cat([X_static, X_temp[ti], M_temp[ti]], dim=1)
            data["muni"].x = xt
            # edges
            data["muni", "dir", "muni"].edge_index = eidx_dir
            if w_dir is not None:
                data["muni", "dir", "muni"].edge_weight = w_dir
            data["muni", "undir", "muni"].edge_index = eidx_undir
            if w_undir is not None:
                data["muni", "undir", "muni"].edge_weight = w_undir
            data_per_year[y] = data.to(device)

        return GraphInputs(
            data_per_year=data_per_year, years_sorted=years_sorted, node_index=index_map
        )

    def run(
        self,
        device,
        tabu: pd.DataFrame,
        temp: pd.DataFrame,
        mdir: pd.DataFrame,
        mndi: pd.DataFrame,
        node_id_col: str = "cc",
        src_col: str = "src",
        dst_col: str = "dst",
        year_col: str = "year",
        static_cols: Optional[List[str]] = None,
        temp_cols: Optional[List[str]] = None,
        edge_attr_cols_dir: Optional[List[str]] = None,
        edge_attr_cols_undir: Optional[List[str]] = None,
    ):
        """
        Execute the full HGT-TE preprocessing pipeline.

        Args:
            device: Computation device
            tabu: Tabular data
            temp: Temporal data
            mdir: Directed edge data
            mndi: Undirected edge data
            node_id_col: Node identifier column
            src_col: Source column for edges
            dst_col: Destination column for edges
            year_col: Year column
            static_cols: Static feature columns
            temp_cols: Temporal feature columns
            edge_attr_cols_dir: Directed edge attribute columns
            edge_attr_cols_undir: Undirected edge attribute columns

        Returns:
            Tuple of (graph inputs, processed tabular data, processed temporal data, metadata, input dimension)
        """
        tabu, temp, mdir, mndi = global_prepro(
            tabu, temp, mdir, mndi, no_mad=self.no_mad, add_idea_emb=self.add_idea_emb
        )
        tabu_hgtte, temp_hgtte = self._prepro_tabular_data(tabu, temp)
        ginputs = self._build_hetero_snapshots(
            tabu_df=tabu_hgtte,
            temp_df=temp_hgtte,
            mdir_df=mdir,
            mndi_df=mndi,
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

        # Graph metadata (typos) from snapshot
        any_year = ginputs.years_sorted[0]
        meta = ginputs.data_per_year[any_year].metadata()

        # Input dimension: static + temp + mask
        in_dim = ginputs.data_per_year[any_year]["muni"].x.size(1)

        return ginputs, tabu_hgtte, temp_hgtte, meta, in_dim


# +---------+
# |  MODEL  |
# +---------+


class TemporalEncoding(nn.Module):
    """
    Sinusoidal absolute temporal encoding for years.

    Generates fixed positional encodings based on year differences from a base year.
    """

    def __init__(self, d_model: int, base_year: int = 2022, max_delta: int = 128):
        """
        Initialize temporal encoder.

        Args:
            d_model: Encoding dimension
            base_year: Reference year for encoding
            max_delta: Maximum year difference to encode
        """
        super().__init__()
        self.d_model = d_model
        self.base_year = base_year

        pe = torch.zeros((2 * max_delta + 1, d_model))
        position = torch.arange(-max_delta, max_delta + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [2*max_delta+1, d_model]
        self.max_delta = max_delta

    def forward(self, years: List[int]) -> Tensor:
        """
        Generate temporal encodings for list of years.

        Args:
            years: List of years to encode

        Returns:
            Temporal encodings tensor
        """
        idxs = []
        for y in years:
            delta = max(-self.max_delta, min(self.max_delta, y - self.base_year))
            idxs.append(delta + self.max_delta)
        idxs = torch.tensor(idxs, device=self.pe.device, dtype=torch.long)
        return self.pe[idxs]  # [T, d_model]


class HGTSpatialEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer spatial encoder.

    Processes individual graph snapshots using HGTConv layers
    to capture spatial relationships within each time step.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        metadata,
        heads: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize HGT spatial encoder.

        Args:
            in_dim: Input feature dimension
            hidden: Hidden dimension
            out_dim: Output dimension
            metadata: Graph metadata for heterogeneous convolution
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        self.hgt1 = HGTConv(hidden, hidden, metadata=metadata, heads=heads)
        self.hgt2 = HGTConv(hidden, out_dim, metadata=metadata, heads=heads)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> Tensor:
        """
        Forward pass through spatial encoder.

        Args:
            data: Heterogeneous graph data snapshot

        Returns:
            Spatially encoded node representations
        """
        x_dict = {k: self.lin_in(v) for k, v in data.x_dict.items()}
        x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}
        x_dict = self.hgt1(x_dict, data.edge_index_dict)
        x_dict = {k: self.act(self.dropout(v)) for k, v in x_dict.items()}
        x_dict = self.hgt2(x_dict, data.edge_index_dict)
        return x_dict["muni"]  # [N, out_dim]


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for sequence processing.

    Applies transformer layers to capture temporal dependencies
    across multiple time steps.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal transformer.

        Args:
            d_model: Input/output dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ff_dim: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through temporal transformer.

        Args:
            x: Input sequence tensor [N, T, D]

        Returns:
            Temporally transformed sequence
        """
        return self.encoder(x)


class HGTTemporalEncoder(nn.Module):
    """
    Complete HGT Temporal Encoder combining spatial and temporal processing.

    Integrates HGT spatial encoding with temporal transformer
    for spatio-temporal representation learning.
    """

    def __init__(
        self,
        *,
        input_dim_per_year: int,
        spatial_hidden: int = 128,
        spatial_out: int = 128,
        heads: int = 2,
        dropout: float = 0.1,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_ff: int = 512,
        target_year: int = 2022,
        years_sorted: List[int] = None,
        metadata=None,
        temporal_pe_dim: Optional[int] = None,
        lambda_focus: float = 0.25,
    ):
        """
        Initialize HGT Temporal Encoder.

        Args:
            input_dim_per_year: Input dimension per time step [static|temp|mask]
            spatial_hidden: Spatial encoder hidden dimension
            spatial_out: Spatial encoder output dimension
            heads: Number of attention heads
            dropout: Dropout rate
            temporal_layers: Number of temporal transformer layers
            temporal_heads: Number of temporal attention heads
            temporal_ff: Temporal feedforward dimension
            target_year: Target year for temporal weighting
            years_sorted: Chronologically sorted years
            metadata: Graph metadata for HGT
            temporal_pe_dim: Temporal positional encoding dimension
            lambda_focus: Exponential decay factor for temporal weighting
        """
        super().__init__()
        self.target_year = target_year
        self.lambda_focus = lambda_focus
        self.years_sorted = years_sorted
        self.temporal_pe_dim = temporal_pe_dim or spatial_out

        # 1) Temporal PE
        self.tempe = TemporalEncoding(self.temporal_pe_dim, base_year=target_year)
        self.proj_pe = nn.Linear(self.temporal_pe_dim, spatial_out)

        # 2) Spatial Encoder HGT (same structure every snapshot)
        self.spatial = HGTSpatialEncoder(
            in_dim=input_dim_per_year,
            hidden=spatial_hidden,
            out_dim=spatial_out,
            metadata=metadata,
            heads=heads,
            dropout=dropout,
        )

        # 3) Temporal Transformer over municipalities secuence
        self.temporal = TemporalTransformer(
            d_model=spatial_out,
            n_layers=temporal_layers,
            n_heads=temporal_heads,
            ff_dim=temporal_ff,
            dropout=dropout,
        )

    def forward(self, data_per_year: Dict[int, HeteroData]) -> Tensor:
        """
        Forward pass through complete HGT-TE model.

        Args:
            data_per_year: Dictionary of yearly graph snapshots

        Returns:
            Final node embeddings focused on target year
        """
        years = self.years_sorted
        N = data_per_year[years[0]]["muni"].num_nodes

        # 1)
        Hs = []  # [N, D] list
        for y in years:
            h = self.spatial(data_per_year[y])  # [N, D]
            Hs.append(h)
        H = torch.stack(Hs, dim=1)  # [N, T, D]

        # 2)
        pe = self.tempe(years)  # [T, D_pe]
        pe = self.proj_pe(pe)  # [T, D]
        H = H + pe.unsqueeze(0)  # broadcast to [N, T, D]

        # 3)
        Htilde = self.temporal(H)  # [N, T, D]

        # 4) Weighted aggregation towards target_year (exponential approach)
        device = Htilde.device
        years_t = torch.tensor(years, dtype=torch.float32, device=device)
        deltas = torch.clamp(self.target_year - years_t, min=0.0)  # past years: >=0; future -> 0
        weights = torch.exp(-self.lambda_focus * deltas)  # [T]
        weights = weights / (weights.sum() + 1e-8)
        z = torch.einsum("t,ntd->nd", weights, Htilde)  # [N, D]
        return z  # embedding by municipality focused on target_year


# +---------+
# |  TRAIN  |
# +---------+


def train_hgtte(
    device,
    tabu: pd.DataFrame,
    temp: pd.DataFrame,
    mdir: pd.DataFrame,
    mndi: pd.DataFrame,
    add_idea_emb: bool = True,
    no_mad: bool = False,
    node_id_col: str = "cc",
    src_col: str = "src",
    dst_col: str = "dst",
    year_col: str = "year",
    static_cols: Optional[List[str]] = None,
    temp_cols: Optional[List[str]] = None,
    edge_attr_cols_dir: Optional[List[str]] = None,
    edge_attr_cols_undir: Optional[List[str]] = None,
    spatial_hidden: int = 128,
    spatial_out: int = 128,
    heads: int = 2,
    dropout: float = 0.1,
    temporal_layers: int = 2,
    temporal_heads: int = 4,
    temporal_ff: int = 512,
    temporal_pe_dim: Optional[int] = None,
    target_year: int = 2022,
    lambda_focus: float = 0.25,
):
    """
    Train Heterogeneous Graph Temporal Transformer Encoder.

    Complete training pipeline for HGT-TE model including preprocessing,
    model initialization, and inference.

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
        spatial_hidden: Spatial encoder hidden dimension
        spatial_out: Spatial encoder output dimension
        heads: Number of attention heads
        dropout: Dropout rate
        temporal_layers: Number of temporal layers
        temporal_heads: Number of temporal attention heads
        temporal_ff: Temporal feedforward dimension
        target_year: Target year for prediction
        lambda_focus: Exponential decay for temporal weighting

    Returns:
        Final node embeddings
    """

    p = PreproHGTTE(add_idea_emb=add_idea_emb, no_mad=no_mad)
    ginputs, _, _, meta, in_dim = p.run(
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

    model = HGTTemporalEncoder(
        input_dim_per_year=in_dim,
        spatial_hidden=spatial_hidden,
        spatial_out=spatial_out,
        heads=heads,
        dropout=dropout,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        temporal_ff=temporal_ff,
        target_year=target_year,
        temporal_pe_dim=temporal_pe_dim,
        years_sorted=ginputs.years_sorted,
        metadata=meta,
        lambda_focus=lambda_focus,
    ).to(device)

    model.eval()
    with torch.no_grad():
        Z = model(ginputs.data_per_year)  # [N, D]

    # Embedding DataFrame
    Z_cpu = Z.detach().cpu()
    N, D = Z_cpu.shape
    inv = [""] * N

    index_map = p._build_index_from_tabu(
        tabu if "cc" in tabu.columns else tabu.reset_index().rename(columns={tabu.index.name: "cc"})
    )
    for cc, idx in index_map.items():
        if 0 <= idx < N:
            inv[idx] = str(cc)

    col_names = [f"emb_{j}" for j in range(D)]
    df_emb = pd.DataFrame(Z_cpu.numpy(), columns=col_names)
    df_emb.insert(0, "cc", inv)

    return Z, df_emb
