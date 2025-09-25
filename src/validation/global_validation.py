from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    average_precision_score,
)

from xgboost import XGBRegressor, XGBClassifier

from ..models.utils import global_prepro


def _is_binary(y: np.ndarray) -> bool:
    """
    Check if target array represents binary classification.

    Args:
        y: Target array

    Returns:
        Boolean indicating if target has exactly two classes
    """
    classes = np.unique(y[~pd.isna(y)])
    return len(classes) == 2


def _get_emb_cols(df: pd.DataFrame) -> List[str]:
    """
    Extract embedding column names from DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        List of column names starting with 'emb_'
    """
    return [c for c in df.columns if c.startswith("emb_")]


def _cv_iterator(task: str, y: np.ndarray, n_splits: int, random_state: int):
    """
    Get appropriate cross-validator based on task type.

    Args:
        task: Type of task ('classification' or 'regression')
        y: Target array
        n_splits: Number of folds
        random_state: Random seed

    Returns:
        Cross-validator object
    """
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _regression_models(random_state: int):
    """Create regression model instances with standardized configurations.

    Args:
        random_state: Random seed

    Returns:
        Dictionary of regression models (Linear Regression and XGBoost)
    """
    lin = Pipeline(
        [("scaler", StandardScaler(with_mean=True, with_std=True)), ("lr", LinearRegression())]
    )
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    return {"LR": lin, "XGB": xgb}


def _classification_models(random_state: int, binary: bool):
    """
    Create classification model instances with standardized configurations.

    Args:
        random_state: Random seed
        binary: Whether the task is binary classification

    Returns:
        Dictionary of classification models (Logistic Regression and XGBoost)
    """
    logreg = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logit",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=-1
                ),
            ),
        ]
    )
    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )
    if not binary:
        pass
    xgb = XGBClassifier(**xgb_params)
    return {"LR": logreg, "XGB": xgb}


def _regression_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of regression metrics (RMSE and R²)
    """
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R2": r2}


def _classification_metrics(y_true, y_pred, y_proba, binary: bool):
    """
    Calculate comprehensive classification evaluation metrics.

    Robust metrics handling class imbalance including balanced accuracy,
    MCC, and probability-based metrics when available.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        binary: Whether the task is binary classification

    Returns:
        Dictionary of classification metrics
    """
    acc = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    roc = np.nan
    ap = np.nan
    try:
        if y_proba is not None:
            if binary:
                pos_proba = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                roc = roc_auc_score(y_true, pos_proba)
                ap = average_precision_score(y_true, pos_proba)
            else:
                roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        pass

    return {
        "Accuracy": acc,
        "BalancedAcc": bac,
        "MCC": mcc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1,
        "ROC_AUC": roc,
        "AP": ap,
    }


def _probas_from_model(model, X):
    """
    Extract probabilities from model using available methods.

    Attempts to get probabilities through predict_proba, falls back to
    decision_function with appropriate scaling for binary and multi-class cases.

    Args:
        model: Trained model
        X: Input features

    Returns:
        Probability estimates or None if unavailable
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X)
            if dec.ndim == 1:
                # scale to [0,1]
                dec = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
                return dec
            # simple softmax
            expd = np.exp(dec - dec.max(axis=1, keepdims=True))
            return expd / (expd.sum(axis=1, keepdims=True) + 1e-12)
        except Exception:
            pass
    return None


def _best_threshold(y_true, pos_scores, grid=None):
    """
    Find optimal threshold that maximizes macro F1 score.

    Args:
        y_true: True labels
        pos_scores: Positive class scores/probabilities
        grid: Threshold grid to search

    Returns:
        Optimal threshold value
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (pos_scores >= t).astype(int)
        f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate_embeddings(
    embeddings: List[Tuple[str, pd.DataFrame]],
    validacion_df: pd.DataFrame,
    targets_dict: Dict[str, str],
    id_col: str = "cc",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate embedding quality using cross-validation with multiple models and metrics.

    Comprehensive evaluation framework that tests embeddings on various regression
    and classification tasks using both linear models and XGBoost with proper
    handling of class imbalance and cross-validation.

    Args:
        embeddings: List of (embedding_name, embedding_dataframe) tuples
        validacion_df: DataFrame with target variables for evaluation
        targets_dict: Dictionary mapping target columns to task types
        id_col: Node identifier column name
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with evaluation results across all embeddings and targets
    """
    # Validaciones mínimas
    assert id_col in validacion_df.columns, f"{id_col} no está en validacion_df"
    for name, df in embeddings:
        assert id_col in df.columns, f"{id_col} no está en {name}"

    results = {}

    for emb_name, emb_df in embeddings:
        print(f"Evaluating {emb_name}...")
        emb_cols = _get_emb_cols(emb_df)
        if not emb_cols:
            raise ValueError(f"No se encontraron columnas 'emb_' en el embedding {emb_name}")

        # Merge con validación
        merged = emb_df[[id_col] + emb_cols].merge(validacion_df, on=id_col, how="inner")

        for target_col, task in targets_dict.items():
            print(f"  · Target {target_col} ({task})")
            if target_col not in merged.columns:
                # Si la columna no está, saltamos
                continue

            # Preparar X, y
            X = merged[emb_cols].values
            y = merged[target_col].values

            # Drop NaNs en y y X correspondientes
            mask = ~pd.isna(y)
            X = X[mask]
            y = y[mask]

            if X.shape[0] == 0:
                continue

            if task == "classification":
                # asegurar codificación numérica de clases
                y_unique = pd.unique(y)

                # Mapear a ints garantizando que la clase minoritaria sea 1 (positiva)
                # 1) contador por etiqueta original
                label_counts = pd.Series(y_unique).map(lambda lbl: np.sum(y == lbl)).values
                # 2) ordenar labels por frecuencia ascendente -> minoritaria primero
                sorted_labels = [
                    lbl for _, lbl in sorted(zip(label_counts, y_unique), key=lambda t: t[0])
                ]
                if len(sorted_labels) == 2:
                    mapping = {sorted_labels[1]: 0, sorted_labels[0]: 1}  # minoritaria -> 1
                else:
                    # multiclase: mapeo ordenado estable
                    mapping = {
                        lbl: i for i, lbl in enumerate(sorted(sorted_labels, key=lambda z: str(z)))
                    }
                y = np.vectorize(mapping.get)(y)

                binary = _is_binary(y)

                # Reducción dinámica de n_splits si la minoritaria es pequeña (solo binario)
                if binary:
                    counts = np.bincount(y.astype(int))
                    # evitar zeros en bincount si faltan etiquetas
                    if counts.size < 2:
                        counts = np.pad(counts, (0, 2 - counts.size), constant_values=0)
                    min_class = counts[counts > 0].min() if counts.sum() > 0 else 0
                    eff_splits = min(n_splits, max(2, int(min_class)))  # al menos 2
                    cv = StratifiedKFold(
                        n_splits=eff_splits, shuffle=True, random_state=random_state
                    )
                else:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

                models = _classification_models(random_state, binary)

            elif task == "regression":
                cv = _cv_iterator(task, y, n_splits, random_state)
                models = _regression_models(random_state)

            else:
                raise ValueError(f"Tarea desconocida para '{target_col}': {task}")

            # Evaluación CV
            for pred_name, base_model in models.items():
                print(f"    · Model {pred_name}")
                fold_metrics = []

                for train_idx, test_idx in cv.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model = base_model

                    # Ajustar scale_pos_weight por fold si XGB binario
                    if task == "classification" and pred_name == "XGB" and _is_binary(y):
                        n_pos = int((y_train == 1).sum())
                        n_neg = int((y_train == 0).sum())
                        spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
                        model = clone(base_model)
                        try:
                            model.set_params(scale_pos_weight=spw)
                        except ValueError:
                            pass

                    model.fit(X_train, y_train)

                    if task == "regression":
                        y_pred = model.predict(X_test)
                        m = _regression_metrics(y_test, y_pred)

                    else:
                        # Clasificación
                        y_proba_test = _probas_from_model(model, X_test)

                        if _is_binary(y):
                            # buscar umbral óptimo en el train del fold
                            y_proba_train = _probas_from_model(model, X_train)
                            if y_proba_train is not None:
                                pos_scores_train = (
                                    y_proba_train
                                    if y_proba_train.ndim == 1
                                    else y_proba_train[:, 1]
                                )
                                thr = _best_threshold(y_train, pos_scores_train)
                                pos_scores_test = (
                                    y_proba_test
                                    if (y_proba_test is not None and y_proba_test.ndim == 1)
                                    else (None if y_proba_test is None else y_proba_test[:, 1])
                                )
                                if pos_scores_test is not None:
                                    y_pred = (pos_scores_test >= thr).astype(int)
                                else:
                                    y_pred = model.predict(X_test)
                            else:
                                y_pred = model.predict(X_test)
                        else:
                            # multiclase: predicción directa
                            y_pred = model.predict(X_test)

                        m = _classification_metrics(
                            y_test, y_pred, y_proba_test, binary=_is_binary(y)
                        )

                    fold_metrics.append(m)

                # Promedio de métricas sobre folds
                avg_metrics = {
                    k: np.nanmean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0].keys()
                }
                # std_metrics disponible si lo quieres: std_metrics = {k: np.nanstd([fm[k] for fm in fold_metrics]) for k in fold_metrics[0].keys()}

                # Guardamos media por métrica
                for metric_name in avg_metrics.keys():
                    row_key = (target_col, metric_name)
                    col_key = (emb_name, pred_name)
                    results.setdefault(row_key, {})[col_key] = avg_metrics[metric_name]

    # Construcción del DataFrame final
    if not results:
        return pd.DataFrame()

    # reunir todas las columnas vistas
    all_cols = set()
    for _, cols in results.items():
        all_cols.update(cols.keys())

    def _col_sort_key(col):
        emb, pred = col
        pred_order = {"LR": 0, "XGB": 1}
        return (emb, pred_order.get(pred, 99), pred)

    sorted_cols = sorted(all_cols, key=_col_sort_key)

    index = pd.MultiIndex.from_tuples(results.keys(), names=["target", "metric"])
    columns = pd.MultiIndex.from_tuples(sorted_cols, names=["embedding", "predictor"])

    out = pd.DataFrame(index=index, columns=columns, dtype=float)
    for row_key, cols in results.items():
        for col_key, val in cols.items():
            out.loc[row_key, col_key] = val

    # ordenar índice alfabéticamente por target y métrica
    out = out.sort_index(axis=0)
    return out


def global_validation(
    *,
    tabu: pd.DataFrame,
    temp: pd.DataFrame,
    mdir: pd.DataFrame,
    mndi: pd.DataFrame,
    no_mad: bool = False,
    add_idea_emb: bool = True,
    val_df: pd.DataFrame,
    val_n_splits: int = 3,
    node_id_col: str = "cc",
    year_col: str = "year",
    emb_year: int = 2022,
    embeddings: list,
    seed: int = 33,
) -> pd.DataFrame:
    """
    Perform comprehensive global validation of embeddings across multiple data sources.

    End-to-end validation pipeline that integrates tabular, temporal, and edge data,
    prepares validation targets, and evaluates embedding performance on both
    known and unknown variables using cross-validation.

    Args:
        tabu: Tabular data DataFrame
        temp: Temporal data DataFrame
        mdir: Directed migration data DataFrame
        mndi: Non-directed migration data DataFrame
        no_mad: Whether to exclude Madrid from data
        add_idea_emb: Whether to include idea embeddings
        val_df: Additional validation DataFrame
        val_n_splits: Number of validation folds
        node_id_col: Node identifier column
        year_col: Year column
        emb_year: Embedding target year
        embeddings: List of embeddings to evaluate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with comprehensive validation results
    """
    # Prepro
    tabu, temp, mdir, mndi = global_prepro(
        tabu, temp, mdir, mndi, no_mad=no_mad, add_idea_emb=add_idea_emb
    )

    # Variables de validación (conocidas del año emb_year)
    val_temp_col = ["eur_renta_b_xhab", "n_ss_general_por_hab", "poblacion", "n_migr_inter_por_hab", "n_bibliotecas"]
    temp_val = temp[temp[year_col] == emb_year][[node_id_col] + val_temp_col].set_index(node_id_col)
    val_tabu_col = ["idea_price_mean", "idea_size_mean", "colinda_con_19"]
    tabu_val = tabu[[node_id_col] + val_tabu_col].set_index(node_id_col)

    # DataFrame para validar
    validation_df = temp_val.join(tabu_val).join(val_df).reset_index()

    targets = {
        # Known vars
        "eur_renta_b_xhab": "regression",
        "n_ss_general_por_hab": "regression",
        "idea_price_mean": "regression",
        "idea_size_mean": "regression",
        "poblacion": "regression",
        "n_migr_inter_por_hab": "regression",
        "n_bibliotecas": "regression",
        "colinda_con_19": "classification",
        # Unknown vars
        "n_doc_cred_pre_hip": "regression",
        "n_ongs": "regression",
        "n_lin_tel_ATF": "regression",
    }

    val_results = evaluate_embeddings(
        embeddings=embeddings,
        validacion_df=validation_df,
        targets_dict=targets,
        id_col=node_id_col,
        n_splits=val_n_splits,
        random_state=seed,
    )

    return val_results
