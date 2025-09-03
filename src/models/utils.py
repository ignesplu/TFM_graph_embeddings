import pandas as pd
import re


def _infer_matrix_null_values(mndi: pd.DataFrame) -> pd.DataFrame:
    """
    Infer and fill null values in the MNDI matrix with appropriate defaults.

    Replaces null values in connectivity metrics with twice the maximum observed value
    to represent maximum distance/separation for missing connections.

    Args:
        mndi: Municipal non-directed interactions DataFrame

    Returns:
        Processed DataFrame with filled null values
    """
    mndi["cc_origen"] = mndi["cc1"]
    mndi["cc_destino"] = mndi["cc2"]

    na_values = {
        "n_grado_union_metro": mndi.n_grado_union_metro.max() * 2,
        "n_grado_union_cercanias": mndi.n_grado_union_cercanias.max() * 2,
        "n_grado_union_buses_EMT": mndi.n_grado_union_buses_EMT.max() * 2,
        "n_grado_union_buses_urb": mndi.n_grado_union_buses_urb.max() * 2,
    }

    return mndi.drop(columns=["cc1", "cc2"]).fillna(value=na_values)


def _no_mad_prepro(tabu, temp, mdir, mndi, no_mad: bool = False):
    """
    Filter out Madrid municipality data if specified.

    Removes all data related to Madrid (cc=28079) from all input DataFrames
    when no_mad flag is set to True.

    Args:
        tabu: Tabular data DataFrame
        temp: Temporal data DataFrame
        mdir: Directed migration data DataFrame
        mndi: Non-directed migration data DataFrame
        no_mad: Whether to exclude Madrid from data

    Returns:
        Tuple of filtered DataFrames
    """
    mad_cc = 28079
    if no_mad:
        tabu = tabu[tabu["cc"] != mad_cc]
        temp = temp[temp["cc"] != mad_cc]
        mdir = mdir[mdir["cc_origen"] != mad_cc]
        mdir = mdir[mdir["cc_destino"] != mad_cc]
        mndi = mndi[mndi["cc_origen"] != mad_cc]
        mndi = mndi[mndi["cc_destino"] != mad_cc]

    return tabu, temp, mdir, mndi


def _add_idea_emb_func(df, add_idea_emb: bool):
    """
    Add or remove idea embedding columns based on configuration.

    Either preserves or removes idea embedding columns (matching pattern 'idea_emb_*')
    from the DataFrame depending on the add_idea_emb flag.

    Args:
        df: Input DataFrame
        add_idea_emb: Whether to include idea embeddings

    Returns:
        DataFrame with appropriate idea embedding columns
    """
    if add_idea_emb:
        idea_cols = []
    else:
        idea_reg = re.compile(r"^idea_emb_\w*\d{1,3}$")
        idea_cols = [c for c in df.columns if idea_reg.match(c)]

    return df.drop(idea_cols, axis=1)


def global_prepro(tabu, temp, mdir, mndi, no_mad: bool = False, add_idea_emb: bool = True):
    """
    Apply global preprocessing pipeline to all input data sources.

    Comprehensive preprocessing function that handles null value imputation,
    Madrid filtering, and idea embedding management across all data sources.

    Args:
        tabu: Tabular data DataFrame
        temp: Temporal data DataFrame
        mdir: Directed migration data DataFrame
        mndi: Non-directed migration data DataFrame
        no_mad: Whether to exclude Madrid from data
        add_idea_emb: Whether to include idea embeddings

    Returns:
        Tuple of preprocessed DataFrames
    """

    mndi = _infer_matrix_null_values(mndi)
    tabu = _add_idea_emb_func(tabu, add_idea_emb)
    tabu, temp, mdir, mndi = _no_mad_prepro(tabu, temp, mdir, mndi, no_mad=no_mad)

    return tabu, temp, mdir, mndi
