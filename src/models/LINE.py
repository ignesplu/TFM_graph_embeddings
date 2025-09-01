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

    def __init__(
            self,
            add_idea_emb: bool = True,
            no_mad: bool = False,
            year: int = 2022
    ):
        self.add_idea_emb = add_idea_emb
        self.no_mad = no_mad
        self.year = year

    def _prepro_tabular_data(
            self,
            tabu: pd.DataFrame,
            temp: pd.DataFrame,
    ):

        # Tabular Static Columns
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

        # Tabular Temporal Columns
        temp_cols = ["cc", "geo_dens_poblacion", "y_edad_media", "p_feminidad"] + [
            col for col in temp.columns if (col.endswith("por_hab") | col.endswith("_xhab"))
        ]
        temp_line = temp[temp.year == self.year][temp_cols]

        # Unify + Columns
        line_cols = temp_cols + tabu_idea_cols + tabu_colinda_cols + tabu_other_cols
        not_in_line = list(set(tabu.columns.tolist() + temp.columns.tolist()) - set(line_cols))

        full_tabu_df = temp_line.set_index("cc").join(tabu_line.set_index("cc")).reset_index()
        aux_cols = {
            "num_cols": temp_cols + tabu_idea_cols + tabu_other_cols,
            "not_in_line": not_in_line,
        }

        return full_tabu_df, aux_cols

    def ratios_cc(
            self,
            df: pd.DataFrame,
            num_tabu_cols: list,
            cc_col: str = "cc",
            both_directions: bool = False,
    ) -> pd.DataFrame:
        """
        Calcula ratios entre todas las combinaciones 2 a 2 de los diferentes 'cc'.

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame original con una columna identificadora (por defecto 'cc') y columnas de valores.
        num_tabu_cols : list
            Listado de columnas numéricas tabulares.
        cc_col : str, default 'cc'
            Nombre de la columna identificadora.
        both_directions : bool, default False
            Si True, devuelve también la pareja inversa (cc_destino, cc_origen) con el ratio inverso.
            Si False, solo devuelve combinaciones (no permutaciones), i.e., una sola fila por pareja.

        Returns
        -------
        pd.DataFrame
            DataFrame con columnas: cc_origen, cc_destino, ratio_<col> para cada col.
            Ratios inf o -inf se convierten a 0, ya que se interpreta que ningún lado aporta.
        """
        base = df.copy()
        value_cols = list(set(base.columns) - set(["cc"]))

        if both_directions:
            # Quitar pares identidad (mismo cc con cc)
            pairs_idx = product(base[cc_col], repeat=2)
            pairs_idx = [(i, j) for i, j in pairs_idx if i != j]
        else:
            # Mantener solo combinaciones únicas imponiendo un orden
            pairs_idx = combinations(base[cc_col], 2)

        # Expandir a dataframe
        pairs = pd.DataFrame(pairs_idx, columns=["cc_origen", "cc_destino"])

        # Join para obtener valores
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
        tabu, temp, mdir, mndi = global_prepro(tabu, temp, mdir, mndi, no_mad=self.no_mad, add_idea_emb=self.add_idea_emb)

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

        df_line_mms = df_line[["cc_origen", "cc_destino"] + [col for col in df_line.columns if col.startswith("mms")]]
        df_line_r = df_line[["cc_origen", "cc_destino"] + [col for col in df_line.columns if col.startswith("r")]]

        df_line_mms_mean = df_line_mms.set_index(["cc_origen", "cc_destino"]).mean(axis=1).reset_index()
        df_line_mms_mean.columns = ["cc_origen", "cc_destino", "value"]
        df_line_mms_sum = df_line_mms.set_index(["cc_origen", "cc_destino"]).sum(axis=1).reset_index()
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
    Implementación mínima del modelo LINE.
    - mode='first'  -> proximidad de primer orden (aristas directas)
    - mode='second' -> proximidad de segundo orden (vecindarios)
    """

    def __init__(self, num_nodes: int, dim: int, mode: str = "first"):
        super().__init__()
        assert mode in ("first", "second")
        self.mode = mode
        # Embeddings principales (target nodes)
        self.target = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.target.weight)  # inicialización

        if mode == "second":
            # Para 2º orden se necesita una tabla de embeddings de "contexto"
            self.context = nn.Embedding(num_nodes, dim)
            nn.init.xavier_uniform_(self.context.weight)
        else:
            self.context = None

    def forward(self, src, dst, negs):
        """
        src: nodos origen [B]
        dst: nodos destino (positivos) [B]
        negs: nodos destino negativos [B,K]
        Devuelve la loss de un batch.
        """
        if self.mode == "first":
            v_src = self.target(src)  # embedding del origen
            v_dst = self.target(dst)  # embedding del destino
            v_neg = self.target(negs)  # embeddings de negativos
        else:
            v_src = self.target(src)
            v_dst = self.context(dst)
            v_neg = self.context(negs)

        # Score positivo = <src,dst>
        pos_score = torch.sum(v_src * v_dst, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

        # Score negativo = <src,neg>
        neg_score = torch.einsum("bd,bkd->bk", v_src, v_neg)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-15).mean()

        return pos_loss + neg_loss

    def get_embeddings(self):
        """Devuelve los embeddings aprendidos de los nodos (matriz N x d)."""
        return self.target.weight.detach().cpu().numpy()


class EdgeSampler:
    """
    Permite muestrear aristas con probabilidad proporcional a su peso.
    Implementa el alias method para hacerlo eficiente.
    """

    def __init__(self, src, dst, weights):
        self.src = src.astype(np.int64)
        self.dst = dst.astype(np.int64)
        w = np.maximum(weights.astype(np.float64), 1e-12)  # evitar ceros
        self.prob, self.alias = self._alias_setup(w / w.sum())

    @staticmethod
    def _alias_setup(prob):
        """
        Construcción del alias table para muestreo O(1).
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
        Devuelve un lote de (src,dst) de tamaño batch_size
        usando alias sampling.
        """
        n = len(self.prob)
        kk = np.random.randint(0, n, size=batch_size)
        accept = np.random.rand(batch_size) < self.prob[kk]
        idx = np.where(accept, kk, self.alias[kk])
        return self.src[idx], self.dst[idx]


def build_graph(df: pd.DataFrame, undirected: bool = False):
    """
    Convierte el DataFrame de aristas en arrays listos para entrenar.
    - df: con columnas cc_origen, cc_destino, value
    - undirected=True: duplica aristas (u,v) y (v,u)
    """
    # Crear índice de nodos (0..N-1)
    nodes = pd.Index(pd.unique(df[["cc_origen", "cc_destino"]].values.ravel()))
    id2idx = {n: i for i, n in enumerate(nodes)}

    # Mapear ids de municipios a índices
    df = df.copy()
    df["src"] = df["cc_origen"].map(id2idx)
    df["dst"] = df["cc_destino"].map(id2idx)
    df["w"] = df["value"].astype(float)

    if undirected:
        # duplicar aristas en ambos sentidos
        df_rev = df.rename(columns={"src": "dst", "dst": "src"})
        df = pd.concat([df, df_rev], ignore_index=True)

    # Colapsar duplicados sumando pesos
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
    Entrena LINE en modo 'first' o 'second'.
    """
    model = LINEModel(num_nodes, dim, mode).to(device)
    opt = Adam(model.parameters(), lr=lr)
    sampler = EdgeSampler(src, dst, w)

    steps_per_epoch = max(1, math.ceil(len(src) / batch_size))
    for epoch in range(epochs):
        losses = []
        for _ in range(steps_per_epoch):
            # Muestreo de aristas positivas
            s_np, d_np = sampler.sample(batch_size)
            s = torch.from_numpy(s_np).long().to(device)
            d = torch.from_numpy(d_np).long().to(device)
            # Muestreo negativo: nodos aleatorios
            negs = torch.randint(0, num_nodes, (batch_size, neg), device=device)

            # Calcular loss y actualizar
            loss = model(s, d, negs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if print_bool:
            print(f"[{mode}] Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")
    return model


# +----------------------------+
# |  ENTRENAMIENTO PRINCIPAL   |
# +----------------------------+



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

    # Construir grafo y arrays
    nodes, _, num_nodes, e_src, e_dst, w = build_graph(df, undirected=undirected)

    # Definir dimensión final (ej: 128 -> 64 para 1er orden y 64 para 2º orden)
    dim = emb_dim
    half = dim // 2

    # Entrenar LINE 1º orden
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

    # Entrenar LINE 2º orden
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

    # Concatenar embeddings
    emb = np.concatenate([m1.get_embeddings(), m2.get_embeddings()], axis=1)

    # Crear DataFrame final de embeddings
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "node_id", np.array(nodes))

    return emb_df
