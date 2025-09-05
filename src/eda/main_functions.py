"""
Dentro del presente fichero se comparten únicamente las funcionalidades consideradas como relevantes
utilizadas en el análisis univariante y multivariante de los datos usados en el trabajo (data/sources).
Código menos relevante como visualizaciones básicas de datos (histogramas, diagramas de barras, etc.)
no se muestran en el repositorio.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from itertools import combinations
from tqdm import tqdm

from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler


def plot_map(
    geo_df,
    plot_type: str = "ind",
    col_prefix: str = "geo_dens_poblacion_",
    exclude_capital: bool = True,
    var: str = "geo_dens_poblacion_2022",
) -> None:
    """
    Plot geographical maps for single or multiple variables.

    Creates choropleth maps for geographical data visualization. Supports
    individual variable plots or grid layouts for multiple time series variables.

    Args:
        geo_df: GeoDataFrame with geographical data
        plot_type: Type of plot - 'ind' for individual, 'many' for multiple
        col_prefix: Column prefix for multiple variable plots (plot_type='many')
        exclude_capital: Whether to exclude Madrid from visualization (plot_type='many')
        var: Specific variable for individual plot (plot_type='ind')
    """
    df_copy = geo_df.copy()

    if plot_type == "many":
        columnas = [col for col in df_copy.columns if col.startswith(col_prefix)]
        n = len(columnas)

        # Grid size
        n_cols = 6
        n_rows = math.ceil(n / n_cols)

        # Subplots
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False
        )
        axes = axes.flatten()

        for i, col in enumerate(columnas):
            if exclude_capital:
                df_copy.loc[df_copy.NAMEUNIT == "Madrid", col] = np.nan
                df_copy[df_copy[col].notna()].plot(
                    column=col,
                    cmap="Oranges",
                    linewidth=0.8,
                    edgecolor="0.8",
                    legend=True,
                    ax=axes[i],
                )
            else:
                df_copy.plot(
                    column=col,
                    cmap="Oranges",
                    linewidth=0.8,
                    edgecolor="0.8",
                    legend=True,
                    ax=axes[i],
                )
            axes[i].axis("off")
            axes[i].set_title(col[-4:], fontsize=12)

        # Drop empty subplots
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

    elif plot_type == "ind":

        df_copy.loc[df_copy.NAMEUNIT == "Madrid", var] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        df_copy.plot(
            column=var,
            cmap="Oranges",
            linewidth=0.8,
            edgecolor="0.8",
            legend=True,
            ax=ax,
        )

        ax.set_title(var, fontsize=15)
        ax.axis("off")

    plt.show()


def heatmap(
    df: pd.DataFrame,
    cols: list,
    scale: bool = False,
    pltfigsize=(12, 12),
    pltannot: bool = True,
    th_corr: float = 0.0,
) -> pd.DataFrame:
    """
    Generate correlation heatmap with optional scaling and threshold filtering.

    Creates a heatmap visualization of correlation matrix and returns
    high-correlation variable pairs above specified threshold.

    Args:
        df: Input DataFrame
        cols: Columns to include in correlation analysis
        scale: Whether to apply MinMax scaling before correlation
        pltfigsize: Figure size for heatmap
        pltannot: Whether to display correlation values on heatmap
        th_corr: Correlation threshold for filtering results

    Returns:
        DataFrame of high-correlation variable pairs
    """

    if scale:
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[cols] = scaler.fit_transform(df[cols])
        corr_matrix = df_scaled[cols].corr()
    else:
        corr_matrix = df[cols].corr()

    plt.figure(figsize=pltfigsize)
    sns.heatmap(corr_matrix, annot=pltannot, cmap="coolwarm", vmin=0, vmax=1)
    plt.show()

    mask = abs(corr_matrix) > th_corr

    # High correlated pairs
    correlaciones_altas = corr_matrix[mask].stack().reset_index()
    correlaciones_altas.columns = ["Variable 1", "Variable 2", "Correlación"]

    return correlaciones_altas[
        correlaciones_altas["Variable 1"] < correlaciones_altas["Variable 2"]
    ]


def _get_series_by_prefix(df: pd.DataFrame, prefix: str, col_id="cc"):
    """
    Extract and scale time series data for a given variable prefix.

    Internal function to retrieve and normalize time series data
    for per-capita variables across municipalities.

    Args:
        df: Input DataFrame
        prefix: Variable prefix to extract
        col_id: Municipality identifier column

    Returns:
        Dictionary of scaled time series by municipality
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{4}})_por_hab$")

    series = {}
    for cc in df[col_id].unique():
        df_cc = df[df[col_id] == cc]
        year_cols = [
            (int(match.group(1)), col)
            for col in df_cc.columns
            if (match := pattern.match(col))
        ]
        if len(year_cols) < 2:
            continue

        year_cols.sort()
        years, cols = zip(*year_cols)
        valores = df_cc.iloc[0][list(cols)].values.astype(float)

        valores_scaled = MinMaxScaler().fit_transform(valores.reshape(-1, 1)).flatten()
        serie = pd.Series(valores_scaled, index=years)
        series[cc] = serie.dropna()

    return series


def _extract_prefix(df):
    """
    Extract unique variable prefixes from per-capita column names.

    Identifies all unique variable prefixes in columns following
    the pattern 'prefix_YYYY_por_hab'.

    Args:
        df: Input DataFrame

    Returns:
        List of unique variable prefixes
    """
    pattern = re.compile(r"^(.*)_(\d{4})_por_hab$")
    prefix = set()
    for col in df.columns:
        if match := pattern.match(col):
            prefix.add(match.group(1))
    return list(prefix)


def dtw_distance(df: pd.DataFrame, col_id: str = "cc"):
    """
    Calculate Dynamic Time Warping distance matrix between per-capita variables.

    Computes average DTW distances between time series of different variables
    across municipalities, handling missing years and variable lengths.

    Args:
        df: Input DataFrame with time series data
        col_id: Municipality identifier column

    Returns:
        DataFrame of average DTW distances between variables
    """
    prefixes = _extract_prefix(df)
    series = {pref: _get_series_by_prefix(df, pref, col_id) for pref in prefixes}

    distances = pd.DataFrame(index=prefixes, columns=prefixes, dtype=float)

    for var1, var2 in tqdm(combinations(prefixes, 2), desc="Calculando DTW"):
        distance_ij = []
        common = set(series[var1].keys()) & set(series[var2].keys())

        for cc in common:
            s1 = series[var1][cc]
            s2 = series[var2][cc]
            common_years = s1.index.intersection(s2.index)

            if len(common_years) < 2:
                continue

            v1 = s1.loc[common_years].values
            v2 = s2.loc[common_years].values

            if np.isnan(v1).any() or np.isnan(v2).any():
                continue

            try:
                d = dtw.distance(v1, v2)
                distance_ij.append(d)
            except Exception:
                continue

        if distance_ij:
            mean_val = np.mean(distance_ij)
            distances.loc[var1, var2] = mean_val
            distances.loc[var2, var1] = mean_val

    np.fill_diagonal(distances.values, 0)

    plt.figure(figsize=(12, 10))
    # Round to 1 decimal
    sns.heatmap(distances.round(1), cmap="Reds", annot=True)
    plt.title("Distancia DTW media entre variables (_por_hab)")
    plt.tight_layout()
    plt.show()

    return distances
