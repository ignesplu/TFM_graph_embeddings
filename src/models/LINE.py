import numpy as np
import pandas as pd
from itertools import product, combinations
import math

from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch.optim import Adam

from .utils import global_prepro


# +-----------------+
# |  PREPROCESSING  |
# +-----------------+


class PreproLINE:
    """
    Preprocessor for LINE (Large-scale Information Network Embedding) model.

    Handles data preprocessing, feature engineering, and graph construction
    from tabular and temporal data for subsequent LINE model training.
    """

    def __init__(self, add_idea_emb: bool = True, no_mad: bool = False, year: int = 2022):
        """
        Initialize the LINE preprocessor.

        Args:
            add_idea_emb: Whether to add idea embeddings to the data
            no_mad: Whether to exclude Madrid from the data
            year: The year to use for temporal data filtering
        """
        self.add_idea_emb = add_idea_emb
        self.no_mad = no_mad
        self.year = year

    def _prepro_tabular_data(
        self,
        tabu: pd.DataFrame,
        temp: pd.DataFrame,
    ):
        """
        Preprocess tabular and temporal data for LINE model.

        Args:
            tabu: DataFrame containing tabular data
            temp: DataFrame containing temporal data

        Returns:
            Tuple of (processed DataFrame, auxiliary columns dictionary)
        """

        # Tabular Static
        tabu_idea_cols = [col for col in tabu.columns if col.startswith("idea_")]
        tabu_colinda_cols = [col for col in tabu.columns if col.startswith("colinda_con")]
        tabu_other_cols = [
            "cc",
            "superficie",
            "altitud",
            "geo_distancia_capital",
            "n_viviendas_totales_por_hab",
        ]

        tabu_line = tabu[tabu_idea_cols + tabu_colinda_cols + tabu_other_cols]

        # Tabular Temporal
        temp_cols = ["cc", "geo_dens_poblacion", "y_edad_media", "p_feminidad"] + [
            col for col in temp.columns if (col.endswith("por_hab") | col.endswith("_xhab"))
        ]
        temp_line = temp[temp.year == self.year][temp_cols]

        # Unify
        line_cols = temp_cols + tabu_idea_cols + tabu_colinda_cols + tabu_other_cols
        not_in_line = list(set(tabu.columns.tolist() + temp.columns.tolist()) - set(line_cols))

        full_tabu_df = temp_line.set_index("cc").join(tabu_line.set_index("cc")).reset_index()
        aux_cols = {
            "num_cols": temp_cols + tabu_idea_cols + tabu_other_cols,
            "not_in_line": not_in_line,
        }

        return full_tabu_df, aux_cols

    def _ratios_cc(
        self,
        df: pd.DataFrame,
        num_tabu_cols: list,
        cc_col: str = "cc",
        both_directions: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate ratios between all pairwise combinations of different 'cc' values.

        Args:
            df: Original DataFrame with identifier column and value columns
            num_tabu_cols: List of numerical tabular columns
            cc_col: Name of the identifier column (default 'cc')
            both_directions: Whether to return inverse pairs with inverse ratios

        Returns:
            DataFrame with columns: cc_origen, cc_destino, ratio_<col> for each column
        """
        base = df.copy()
        value_cols = list(set(base.columns) - set(["cc"]))

        if both_directions:
            pairs_idx = product(base[cc_col], repeat=2)
            pairs_idx = [(i, j) for i, j in pairs_idx if i != j]
        else:
            pairs_idx = combinations(base[cc_col], 2)

        pairs = pd.DataFrame(pairs_idx, columns=["cc_origen", "cc_destino"])

        pairs = pairs.merge(base.rename(columns={cc_col: "cc_origen"}), on="cc_origen")
        pairs = pairs.merge(
            base.rename(columns={cc_col: "cc_destino"}),
            on="cc_destino",
            suffixes=("_origen", "_destino"),
        )

        def jaccard_binario(a, b):
            """
            Índice de Jaccard binario entre dos valores dicotómicos (0/1).

            Retorna:
            - 1 si ambos son 1
            - 0 si ambos son 0 o si solo uno es 1
            """
            union = (a == 1) or (b == 1)
            inter = (a == 1) and (b == 1)
            if not union:
                return 0  # caso (0,0)
            return int(inter) / int(union)

        # Calcular ratios por columna
        for col in value_cols:
            num = pairs[f"{col}_origen"]
            den = pairs[f"{col}_destino"]
            if col in num_tabu_cols:
                # Ratio proporcion direccional para los que no son dicotomicos
                pairs[f"ratio_{col}"] = num / (num + den)
            else:
                # Jaccard Binario (simétrico)
                pairs[f"ratio_{col}"] = pairs.apply(
                    lambda row: jaccard_binario(row[f"{col}_origen"], row[f"{col}_destino"]),
                    axis=1,
                )

        # Limpiar infinitos por divisiones entre 0
        ratio_cols = [f"ratio_{c}" for c in value_cols]
        pairs[ratio_cols] = pairs[ratio_cols].replace([np.inf, -np.inf, np.nan], 0)

        # Seleccionar columnas de salida
        out = pairs[["cc_origen", "cc_destino"] + ratio_cols].reset_index(drop=True)
        return out

    def run(
        self,
        tabu: pd.DataFrame,
        temp: pd.DataFrame,
        mdir: pd.DataFrame,
        mndi: pd.DataFrame,
    ):
        """
        Execute the full preprocessing pipeline for LINE model.

        Args:
            tabu: Tabular data DataFrame
            temp: Temporal data DataFrame
            mdir: Direct migration data DataFrame
            mndi: Indirect migration data DataFrame

        Returns:
            Tuple of four DataFrames with different aggregation methods
        """
        tabu, temp, mdir, mndi = global_prepro(
            tabu, temp, mdir, mndi, no_mad=self.no_mad, add_idea_emb=self.add_idea_emb
        )

        tabu_df, aux_cols = self._prepro_tabular_data(tabu, temp)

        tabu_line = self._ratios_cc(
            tabu_df,
            num_tabu_cols=aux_cols["num_cols"],
            cc_col="cc",
            both_directions=True,
        )

        df_line = (
            tabu_line.set_index(["cc_origen", "cc_destino"])
            .join(mndi.set_index(["cc_origen", "cc_destino"]))
            .join(mdir.set_index(["cc_origen", "cc_destino"]))
            .reset_index()
        )

        for col in df_line.columns:
            if col not in ["cc_origen", "cc_destino"]:
                df_col = df_line[col].copy()
                # MMS Scaling
                scaler = MinMaxScaler()
                df_line[f"mms_{col}"] = scaler.fit_transform(df_col.values.reshape(-1, 1))
                # R Scaling
                df_line[f"rank_{col}"] = df_col.rank(method="average", ascending=False)

        df_line_mms = df_line[
            ["cc_origen", "cc_destino"] + [col for col in df_line.columns if col.startswith("mms")]
        ]
        df_line_r = df_line[
            ["cc_origen", "cc_destino"] + [col for col in df_line.columns if col.startswith("r")]
        ]

        df_line_mms_mean = (
            df_line_mms.set_index(["cc_origen", "cc_destino"]).mean(axis=1).reset_index()
        )
        df_line_mms_mean.columns = ["cc_origen", "cc_destino", "value"]
        df_line_mms_sum = (
            df_line_mms.set_index(["cc_origen", "cc_destino"]).sum(axis=1).reset_index()
        )
        df_line_mms_sum.columns = ["cc_origen", "cc_destino", "value"]
        df_line_r_mean = df_line_r.set_index(["cc_origen", "cc_destino"]).mean(axis=1).reset_index()
        df_line_r_mean.columns = ["cc_origen", "cc_destino", "value"]
        df_line_r_sum = df_line_r.set_index(["cc_origen", "cc_destino"]).sum(axis=1).reset_index()
        df_line_r_sum.columns = ["cc_origen", "cc_destino", "value"]

        return df_line_mms_mean, df_line_mms_sum, df_line_r_mean, df_line_r_sum


# +---------+
# |  MODEL  |
# +---------+


class LINEModel(nn.Module):
    """
    Minimal implementation of the LINE (Large-scale Information Network Embedding) model.
    Supports both first-order (direct edges) and second-order (neighborhood) proximity.
    """

    def __init__(self, num_nodes: int, dim: int, mode: str = "first"):
        """
        Initialize the LINE model.

        Args:
            num_nodes: Number of nodes in the graph
            dim: Dimension of the output embeddings
            mode: Proximity mode - 'first' for direct edges, 'second' for neighborhoods
        """
        super().__init__()
        assert mode in ("first", "second")
        self.mode = mode

        # Principal Embeddings (target nodes)
        self.target = nn.Embedding(num_nodes, dim)
        # Initialization
        nn.init.xavier_uniform_(self.target.weight)

        if mode == "second":
            # For the 2nd order, a table of ‘context’ embeddings is required
            self.context = nn.Embedding(num_nodes, dim)
            nn.init.xavier_uniform_(self.context.weight)
        else:
            self.context = None

    def forward(self, src, dst, negs):
        """
        Forward pass for the LINE model.

        Args:
            src: Source nodes [B]
            dst: Destination (positive) nodes [B]
            negs: Negative destination nodes [B,K]

        Returns:
            Batch loss value
        """
        if self.mode == "first":
            v_src = self.target(src)  # source embedding
            v_dst = self.target(dst)  # destination embedding
            v_neg = self.target(negs)  # negatives embeddings
        else:
            v_src = self.target(src)
            v_dst = self.context(dst)
            v_neg = self.context(negs)

        # Score positive = <src,dst>
        pos_score = torch.sum(v_src * v_dst, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

        # Score negative = <src,neg>
        neg_score = torch.einsum("bd,bkd->bk", v_src, v_neg)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-15).mean()

        return pos_loss + neg_loss

    def get_embeddings(self):
        """
        Get the learned node embeddings.

        Returns:
            N x d matrix of node embeddings
        """
        return self.target.weight.detach().cpu().numpy()


class EdgeSampler:
    """
    Samples edges with probability proportional to their weight.
    Implements the alias method for efficient sampling.
    """

    def __init__(self, src, dst, weights):
        """
        Initialize the edge sampler.

        Args:
            src: Source nodes array
            dst: Destination nodes array
            weights: Edge weights array
        """
        self.src = src.astype(np.int64)
        self.dst = dst.astype(np.int64)
        w = np.maximum(weights.astype(np.float64), 1e-12)  # to avoiud zero values
        self.prob, self.alias = self._alias_setup(w / w.sum())

    @staticmethod
    def _alias_setup(prob):
        """
        Construct an alias table for O(1) sampling using the alias method.

        This method sets up the data structures needed for efficient sampling
        from a discrete probability distribution. It creates an alias table
        that allows sampling in constant time after the initial setup.

        Based on the algorithm described in:
        "A Linear Time Algorithm for Generating Random Numbers with a Given Distribution"
        by Michael Vose.

        Args:
            prob: A 1D numpy array of probabilities that sum to 1

        Returns:
            tuple: A tuple containing two arrays:
                - prob_scaled: Modified probability array
                - alias: Alias table where each entry points to another outcome
        """
        n = len(prob)
        alias = np.zeros(n, dtype=np.int64)
        prob_scaled = prob * n
        small, large = [], []
        for i, p in enumerate(prob_scaled):
            (small if p < 1.0 else large).append(i)
        while small and large:
            s, l = small.pop(), large.pop()
            alias[s] = l
            prob_scaled[l] -= 1.0 - prob_scaled[s]
            if prob_scaled[l] < 1.0:
                small.append(l)
            else:
                large.append(l)
        return np.minimum(prob_scaled, 1.0), alias

    def sample(self, batch_size):
        """
        Sample a batch of edges.

        Args:
            batch_size: Number of edges to sample

        Returns:
            Tuple of (source nodes, destination nodes) arrays
        """
        n = len(self.prob)
        kk = np.random.randint(0, n, size=batch_size)
        accept = np.random.rand(batch_size) < self.prob[kk]
        idx = np.where(accept, kk, self.alias[kk])
        return self.src[idx], self.dst[idx]


def build_graph(df: pd.DataFrame, undirected: bool = False):
    """
    Convert edge DataFrame into arrays ready for training.

    Args:
        df: DataFrame with columns cc_origen, cc_destino, value
        undirected: Whether to duplicate edges in both directions

    Returns:
        Tuple of (nodes, id2idx mapping, number of nodes, source array,
                 destination array, weight array)
    """
    nodes = pd.Index(pd.unique(df[["cc_origen", "cc_destino"]].values.ravel()))
    id2idx = {n: i for i, n in enumerate(nodes)}

    # Map minipalities ids to index
    df = df.copy()
    df["src"] = df["cc_origen"].map(id2idx)
    df["dst"] = df["cc_destino"].map(id2idx)
    df["w"] = df["value"].astype(float)

    if undirected:
        df_rev = df.rename(columns={"src": "dst", "dst": "src"})
        df = pd.concat([df, df_rev], ignore_index=True)

    # Colapse duplicates adding weights
    df = df.groupby(["src", "dst"], as_index=False)["w"].sum()

    return (
        nodes,
        id2idx,
        len(nodes),
        df["src"].to_numpy(),
        df["dst"].to_numpy(),
        df["w"].to_numpy(),
    )


def train_line(
    num_nodes: int,
    src,
    dst,
    w,
    dim: int,
    epochs: int,
    batch_size: int,
    neg: int,
    lr: float,
    device,
    mode: str,
    print_bool: bool,
):
    """
    Train LINE model in either 'first' or 'second' order mode.

    Args:
        num_nodes: Number of nodes in the graph
        src: Source nodes array
        dst: Destination nodes array
        w: Edge weights array
        dim: Embedding dimension
        epochs: Number of training epochs
        batch_size: Training batch size
        neg: Number of negative samples
        lr: Learning rate
        device: Device to train on (CPU/GPU)
        mode: Training mode - 'first' or 'second'
        print_bool: Whether to print training progress

    Returns:
        Trained LINE model
    """
    model = LINEModel(num_nodes, dim, mode).to(device)
    opt = Adam(model.parameters(), lr=lr)
    sampler = EdgeSampler(src, dst, w)

    steps_per_epoch = max(1, math.ceil(len(src) / batch_size))
    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
            # Positive edges sampler
            s_np, d_np = sampler.sample(batch_size)
            s = torch.from_numpy(s_np).long().to(device)
            d = torch.from_numpy(d_np).long().to(device)

            # Negative sampler: random nodes
            negs = torch.randint(0, num_nodes, (batch_size, neg), device=device)

            # Calculate loss + update
            loss = model(s, d, negs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if print_bool:
            print(f"[{mode}] Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")

    return model


# +---------+
# |  TRAIN  |
# +---------+


def full_train_line(
    device,
    df: pd.DataFrame,
    undirected: bool = True,
    emb_dim: int = 128,
    n_epochs: int = 5,
    batch_size: int = 10000,
    neg: int = 5,
    lr: float = 0.0025,
    print_bool: bool = False,
) -> pd.DataFrame:
    """
    Complete training pipeline for LINE model with both first and second order proximities.

    Args:
        device: Device to train on (CPU/GPU)
        df: Input DataFrame with edge information
        undirected: Whether the graph is undirected
        emb_dim: Total embedding dimension
        n_epochs: Number of training epochs
        batch_size: Training batch size
        neg: Number of negative samples
        lr: Learning rate
        print_bool: Whether to print training progress

    Returns:
        DataFrame with learned node embeddings
    """
    nodes, _, num_nodes, e_src, e_dst, w = build_graph(df, undirected=undirected)

    dim = emb_dim
    half = dim // 2

    # LINE first-order
    if print_bool:
        print("Entrenando LINE 1er orden...")
    m1 = train_line(
        num_nodes,
        e_src,
        e_dst,
        w,
        half,
        epochs=n_epochs,
        batch_size=batch_size,
        neg=neg,
        lr=lr,
        device=device,
        mode="first",
        print_bool=print_bool,
    )

    # LINE second-order
    if print_bool:
        print("Entrenando LINE 2º orden...")
    m2 = train_line(
        num_nodes,
        e_src,
        e_dst,
        w,
        half,
        epochs=n_epochs,
        batch_size=batch_size,
        neg=neg,
        lr=lr,
        device=device,
        mode="second",
        print_bool=print_bool,
    )

    # Concatenate embeddings
    emb = np.concatenate([m1.get_embeddings(), m2.get_embeddings()], axis=1)

    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "node_id", np.array(nodes))

    return emb_df
