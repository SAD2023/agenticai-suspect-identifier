"""

Random Forest was chosen as the base model because it:
    (a) Handles mixed numeric + categorical features with minimal tuning.
    (d) Is entirely open-source and free (scikit-learn, Apache 2.0 license).
    (e) Does not require a GPU and runs well on standard government hardware.

"""

import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)

from config import MODEL_CONFIG, FEATURE_CONFIG, FILE_CONFIG

logger = logging.getLogger(__name__)


def build_full_pipeline(
    numeric_features: list,
    categorical_features: list,
) -> Pipeline:
    """
    Constructs the full sklearn Pipeline:
      
    Args:
        numeric_features     (list): Column names of numeric input features.
        categorical_features (list): Column names of categorical input features.

    Returns:
        Pipeline: An unfitted sklearn Pipeline object.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",   # Silently ignore unseen categories at predict time.
            drop="first",             # Drop one dummy per feature to avoid collinearity.
            sparse_output=False,      # Return dense array for compatibility.
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric",      numeric_pipe,      numeric_features),
            ("categorical",  categorical_pipe,  categorical_features),
        ],
        remainder="drop",  # Any column not listed in either list is silently dropped.
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(**MODEL_CONFIG)),
    ])

    return pipeline



def train_model(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    categorical_columns: list = None,
    test_size: float = 0.20,
) -> tuple:
    """
    Fits the full pipeline on the provided labeled DataFrame.


    Args:
        df                 (pd.DataFrame): Full labeled dataset.
        feature_columns    (list):         Columns to use as model inputs.
        target_column      (str):          Column containing 0/1 labels.
        categorical_columns (list):        Subset of feature_columns that are
                                           categorical (will be one-hot encoded).
                                           Defaults to [] (all numeric).
        test_size          (float):        Fraction held out for evaluation (default 0.20).

    Returns:
        tuple: (fitted_pipeline, X_test, y_test)
          fitted_pipeline — sklearn Pipeline, ready to call .predict_proba() on.
          X_test          — Held-out feature DataFrame for evaluation.
          y_test          — Held-out label Series for evaluation.

    """
    if categorical_columns is None:
        categorical_columns = []

    # Sanity-check that all referenced columns exist in the DataFrame.
    all_required = feature_columns + [target_column]
    missing_cols = [c for c in all_required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the training dataset: {missing_cols}\n"
            "Update FEATURE_CONFIG in config.py to match your dataset's column names."
        )

    numeric_features = [c for c in feature_columns if c not in categorical_columns]

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    class_dist = y.value_counts().to_dict()
    logger.info(
        f"Training data: {X.shape[0]:,} rows, {X.shape[1]} features. "
        f"Class distribution: {class_dist}"
    )

    if len(y.unique()) < 2:
        raise ValueError(
            "The target column contains only one class. "
            "A binary classifier requires examples of both class 0 and class 1."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,   # Preserve class ratio in both splits.
    )

    pipeline = build_full_pipeline(numeric_features, categorical_features=categorical_columns)

    logger.info("Fitting model pipeline — this may take a minute for large datasets...")
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting complete.")

    return pipeline, X_test, y_test


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list,
) -> dict:
    """
    Evaluates the trained model on the held-out test set and logs key metrics.


    Args:
        pipeline        (Pipeline):     Fitted sklearn Pipeline.
        X_test          (pd.DataFrame): Held-out feature data.
        y_test          (pd.Series):    Held-out labels.
        feature_columns (list):         Original feature names (before any encoding).

    Returns:
        dict: {
            "classification_report": dict,
            "roc_auc": float,
            "top_features": dict   # {feature_name: importance_score}
        }
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    roc_auc     = roc_auc_score(y_test, y_prob)

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION ON HELD-OUT TEST SET")
    logger.info("=" * 50)
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

    # Feature importances from the Random Forest
    rf_model    = pipeline.named_steps["classifier"]
    importances = rf_model.feature_importances_

    try:
        feature_names_out = (
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
    except AttributeError:
        feature_names_out = [f"feature_{i}" for i in range(len(importances))]

    importance_series = (
        pd.Series(importances, index=feature_names_out)
        .sort_values(ascending=False)
    )

    logger.info("\nTop 10 most important features:")
    for feat, imp in importance_series.head(10).items():
        logger.info(f"  {feat:<40s}  {imp:.4f}")
    logger.info("=" * 50 + "\n")

    return {
        "classification_report": report_dict,
        "roc_auc":               round(roc_auc, 6),
        "top_features":          importance_series.head(10).to_dict(),
    }


def compute_training_stats(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Computes descriptive statistics for each numeric feature column.
    Args:
        df              (pd.DataFrame): Training DataFrame (before train/test split).
        feature_columns (list):         Feature column names.

    Returns:
        dict: { column_name: {"mean": float, "std": float, "median": float} }
    """
    stats = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_data = df[col].dropna()
            stats[col] = {
                "mean":   float(col_data.mean()),
                "std":    float(col_data.std()),
                "median": float(col_data.median()),
            }
    return stats



def save_artifacts(pipeline: Pipeline, metadata: dict = None) -> None:
    """
    Saves the fitted pipeline and metadata to disk using joblib.
    Args:
        pipeline (Pipeline): The fitted sklearn Pipeline to persist.
        metadata (dict):     Optional dict to save alongside the model.
                             Should include at minimum:
                               "feature_columns"    — list of feature column names.
                               "categorical_columns"— list of categorical column names.
                               "target_column"      — name of the label column.
                               "training_stats"     — output of compute_training_stats().
                               "eval_metrics"       — output of evaluate_model().
    """
    model_path = FILE_CONFIG["MODEL_SAVE_PATH"]
    joblib.dump(pipeline, model_path)
    logger.info(f"Model pipeline saved → {model_path}")

    if metadata:
        meta_path = FILE_CONFIG["MODEL_METADATA_PATH"]
        joblib.dump(metadata, meta_path)
        logger.info(f"Model metadata saved → {meta_path}")


def load_artifacts() -> tuple:
    """
    Loads a previously saved pipeline and its metadata from disk.

    Returns:
        tuple: (pipeline, metadata)
          pipeline — Fitted sklearn Pipeline.
          metadata — Dict with training configuration and stats,
                     or empty dict if no metadata file is found.
    """
    model_path = FILE_CONFIG["MODEL_SAVE_PATH"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Train the model first:  python main.py train --data <path_to_dataset>"
        )

    pipeline = joblib.load(model_path)
    logger.info(f"Model pipeline loaded ← {model_path}")

    meta_path = FILE_CONFIG["MODEL_METADATA_PATH"]
    metadata  = joblib.load(meta_path) if os.path.exists(meta_path) else {}
    if not metadata:
        logger.warning(
            "No model metadata file found. "
            "Explanation quality will be reduced."
        )

    return pipeline, metadata


def run_training_pipeline(
    training_data_path: str,
    feature_columns: list,
    target_column: str,
    categorical_columns: list = None,
) -> None:
    """
    Full training flow: load data → train → evaluate → save.

    Args:
        training_data_path  (str):  Path to the labeled training dataset.
        feature_columns     (list): Feature columns to use as model inputs.
        target_column       (str):  Label column name.
        categorical_columns (list): Subset of feature_columns that are categorical.
    """
    from data_loader import load_training_data, validate_dataframe

    if categorical_columns is None:
        categorical_columns = []

    # --- Load ---
    logger.info(f"Loading training data from: {training_data_path}")
    df = load_training_data(training_data_path)
    validate_dataframe(df, feature_columns + [target_column], context="training data")

    # --- Train ---
    pipeline, X_test, y_test = train_model(
        df,
        feature_columns,
        target_column,
        categorical_columns,
    )

    # --- Evaluate ---
    metrics = evaluate_model(pipeline, X_test, y_test, feature_columns)

    # --- Compute training stats for explanation generation ---
    training_stats = compute_training_stats(df, feature_columns)

    # --- Save ---
    metadata = {
        "feature_columns":     feature_columns,
        "categorical_columns": categorical_columns,
        "target_column":       target_column,
        "eval_metrics":        metrics,
        "training_stats":      training_stats,
    }
    save_artifacts(pipeline, metadata)

    logger.info("Training pipeline complete. Model is ready for predictions.")
