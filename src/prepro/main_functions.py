"""
Dentro del presente fichero se comparten únicamente las funcionalidades consideradas como relevantes
utilizadas en el prepocesado de datos usados en el trabajo (data/sources). Código menos relevante como
depuración de datos base (tipados, duplicados, cruces, etc.) no se muestran en el repositorio.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import re
from collections import defaultdict

from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from sentence_transformers import SentenceTransformer


def prepro_shp_file(file_path: str):
    """
    Preprocess SHP file to extract adjacency relationships between municipalities.

    This function processes a shapefile containing Spanish municipalities to create:
    1. A dataframe with adjacency relationships between municipalities within Madrid province
    2. A dataframe indicating which Madrid municipalities border other autonomous communities

    Args:
        file_path (str): Path to the SHP file containing municipality data

    Returns:
        tuple: Two pandas DataFrames:
            - adjacency_df: Contains municipality pairs and their adjacency status (1/0)
            - external_borders_df: Contains municipalities bordering other autonomous communities
    """
    # Read the shapefile
    gdf = gpd.read_file(file_path)

    # Filter municipalities from Madrid province (code 341328)
    madrid_gdf = gdf[gdf["NATCODE"].str.startswith("341328")].copy()
    madrid_gdf = madrid_gdf.to_crs(epsg=25830)
    municipalities = madrid_gdf["NAMEUNIT"].tolist()

    # Create all possible municipality-municipality combinations
    combinations = [(a, b) for a in municipalities for b in municipalities]

    # Create DataFrame of adjacency relationships between Madrid municipalities
    results = []
    for mun_a, mun_b in combinations:
        geom_a = madrid_gdf[madrid_gdf["NAMEUNIT"] == mun_a].geometry.values[0]
        geom_b = madrid_gdf[madrid_gdf["NAMEUNIT"] == mun_b].geometry.values[0]
        adjacent = int(geom_a.touches(geom_b))  # 1 if adjacent, 0 if not
        results.append((mun_a, mun_b, adjacent))

    adjacency_df = pd.DataFrame(results, columns=["Municipio_A", "Municipio_B", "Colindantes"])

    # Filter municipalities from other autonomous communities
    other_communities_gdf = gdf[~gdf["NATCODE"].astype(str).str.startswith("341328")].copy()
    other_communities_gdf = other_communities_gdf.to_crs(epsg=25830)

    # Create DataFrame of Madrid municipalities bordering other autonomous communities
    external_results = []
    for _, row in madrid_gdf.iterrows():
        municipality_geom = row.geometry
        municipality_name = row["NAMEUNIT"]
        borders_external = False
        neighboring_communities = set()

        # Check if the municipality touches any municipality outside Madrid
        for _, other in other_communities_gdf.iterrows():
            if municipality_geom.touches(other.geometry):
                borders_external = True
                # Extract neighboring community code from first 6 digits of NATCODE
                community_code = other["NATCODE"][:6]
                neighboring_communities.add(community_code)

        if borders_external:
            result = (municipality_name, 1, ", ".join(sorted(neighboring_communities)))
        else:
            result = (municipality_name, 0, "")

        external_results.append(result)

    external_borders_df = pd.DataFrame(
        external_results, columns=["Municipio", "Colinda_con_otra_CCAA", "CCAA_vecina"]
    )

    return adjacency_df, external_borders_df


def impute_knn(df: pd.DataFrame, ind_dict, neig_grid=[3, 5, 7, 9]) -> pd.DataFrame:
    """
    Impute missing values using K-Nearest Neighbors with cross-validation.

    Performs KNN imputation for time series columns identified by prefix and year ranges.
    Uses cross-validation to select the optimal number of neighbors.

    Args:
        df: Input DataFrame with missing values
        ind_dict: Dictionary mapping prefixes to (start_year, end_year) tuples
        neig_grid: List of k values to test for KNN

    Returns:
        DataFrame with imputed values
    """
    df_impute = df.copy()
    for prefix, (start, end) in ind_dict.items():
        cols = [f"{prefix}_{year}" for year in range(start, end + 1)]
        missing_cols = [col for col in cols if col in df.columns]

        if not missing_cols:
            continue

        subdata = df_impute[missing_cols]

        # Máscara para identificar valores reales no nulos
        mask_notna = subdata.notna()

        # Enmascaramos aleatoriamente un 10% de los datos no nulos para validación
        np.random.seed(42)
        mask_for_cv = (np.random.rand(*subdata.shape) < 0.1) & mask_notna
        original_values = subdata[mask_for_cv]

        subdata_masked = subdata.mask(mask_for_cv)

        scores = {}
        for k in neig_grid:
            try:
                knn = KNNImputer(n_neighbors=k)
                imputado = knn.fit_transform(subdata_masked)
                imputado_df = pd.DataFrame(imputado, columns=missing_cols, index=subdata.index)
                error = mean_squared_error(original_values, imputado_df[mask_for_cv])
                scores[k] = error
            except Exception:
                scores[k] = np.inf

        best_k = min(scores, key=scores.get)

        # Imputar con el mejor valor encontrado
        knn_final = KNNImputer(n_neighbors=best_k)
        imputed = knn_final.fit_transform(subdata)
        df_impute[missing_cols] = imputed

    return df_impute


def impute_rf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using Random Forest regression.

    Uses Random Forest with grid search to predict missing values
    based on other property features. Handles categorical variables
    through one-hot encoding.

    Args:
        df: Input DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    var2input = "floor"

    # División entre train (con floor) y test (sin floor)
    train = df[df[var2input].notna()]
    test = df[df[var2input].isna()]

    # Variables predictoras
    features = [
        "price",
        "size",
        "rooms",
        "bathrooms",
        "exterior",
        "priceByArea",
        "status",
        "hasLift",
        "hasAirConditioning",
        "hasBoxRoom",
        "hasGarden",
        "hasSwimmingPool",
        "hasTerrace",
        "hasParkingSpace",
        "hasStaging",
        "municipality",
        "newDevelopment",
        "latitude",
        "longitude",
    ]

    # One-hot encoding
    X_train = pd.get_dummies(train[features])
    y_train = train[var2input]

    # Grid de hiperparámetros para Random Forest
    param_grid = {
        "n_estimators": [75, 200],
        "max_depth": [7, 15],
        "min_samples_leaf": [1, 5],
    }

    # Definimos el modelo base
    rf = RandomForestRegressor(random_state=42)

    # Grid search con validación cruzada
    grid = GridSearchCV(
        rf, param_grid, cv=2, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    # Mejor modelo encontrado
    best_model = grid.best_estimator_
    print(f"Mejor combinación: {grid.best_params_}")

    # Preparar datos de test para predicción
    X_test = pd.get_dummies(test[features])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Imputar valores faltantes
    df.loc[df[var2input].isna(), var2input] = np.round(best_model.predict(X_test)).astype(int)

    return df


def fill_and_cut_time_series(data: pd.DataFrame, ceil_year: int = 2022) -> pd.DataFrame:
    """
    Fill missing years in time series columns and trim future years.

    Identifies time series columns with pattern 'prefix_YYYY' and:
    1. Removes columns for years beyond the specified ceiling year
    2. Imputes missing intermediate years using linear regression per row
    3. Maintains complete time series from minimum available year to ceiling year

    Args:
        data: Input DataFrame with time series columns
        ceil_year: Maximum year to include (columns beyond this year + 1 are dropped)

    Returns:
        DataFrame with processed time series columns
    """
    ceil_plus = ceil_year + 1

    # Detect columns prefix_YYYY
    pattern = r"^(.*)_(\d{4})$"
    col_info = []

    for col in data.columns:
        match = re.match(pattern, col)
        if match:
            prefix, year = match.groups()
            year = int(year)
            col_info.append((prefix, year, col))

    # Group by prefix
    prefix_dict = defaultdict(list)
    for prefix, year, col in col_info:
        prefix_dict[prefix].append((year, col))

    columns_to_drop = []
    imputed_columns = {}

    for prefix, year_cols in prefix_dict.items():
        year_cols = sorted(year_cols)

        # Drop columns >= 2023
        for y, col in year_cols:
            if y >= ceil_plus:
                columns_to_drop.append(col)

        valid_year_cols = [(y, col) for y, col in year_cols if y < ceil_plus]
        if len(valid_year_cols) < 2:
            continue  # At least 2 points to impute

        valid_years = [y for y, _ in valid_year_cols]
        min_year = min(valid_years)
        target_years = list(range(min_year, ceil_plus))

        missing_years = sorted(set(target_years) - set(valid_years))
        if missing_years:
            imputed_columns[prefix] = missing_years

            # Train and impute by row
            valid_cols = [col for _, col in valid_year_cols]
            X = np.array(valid_years).reshape(-1, 1)

            for idx, row in data[valid_cols].iterrows():
                y = row.values.astype(float)
                reg = LinearRegression().fit(X, y)

                for year in missing_years:
                    pred = reg.predict(np.array([[year]]))[0]
                    imputed_col = f"{prefix}_{year}"
                    data.at[idx, imputed_col] = pred

    # Print summary
    for prefix, years in imputed_columns.items():
        print(f"Prefijo: {prefix}, años imputados: {years}")

    return data


def add_idea_text_emb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add text embeddings for idea titles and descriptions using E5-base-v2 model.

    Generates text embeddings for the 'title_description' column using the
    intfloat/e5-base-v2 sentence transformer model. This model achieves
    83.5% top-5 retrieval accuracy according to MTEB benchmarks.

    Based on:
    - Muennighoff, N. et al. (2025). Best Open-Source Embedding Models Benchmarked and Ranked
    - https://huggingface.co/intfloat/e5-base-v2
    - https://arxiv.org/pdf/2212.03533

    Args:
        df: Input DataFrame containing a 'title_description' column

    Returns:
        DataFrame with added 'title_desc_emb' column containing text embeddings
    """
    if "title_description" not in df.columns:
        raise ValueError("La columna 'title_description' no está en el DataFrame")

    model = SentenceTransformer("intfloat/e5-base-v2", device="cuda")

    texts = df["title_description"].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, device="cuda")
    df["title_desc_emb"] = embeddings.tolist()

    return df
