#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import clone
import joblib

from catboost import CatBoostRegressor


DATA_PATH = "./data/processed/pred.csv"
MODELS_DIR = "./src/models"
TEST_START = "2025-10-01"  # adjust if you want a different test cutoff


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["day"])
    # Sort by time and station to preserve temporal order
    df = df.sort_values(["day", "station_id"]).reset_index(drop=True)
    # ensure station_id is string / categorical
    df["station_id"] = df["station_id"].astype("string")
    return df


def make_train_test_split(df: pd.DataFrame, test_start: str):
    test_start_ts = pd.to_datetime(test_start)
    df_train = df[df["day"] < test_start_ts].copy()
    df_test = df[df["day"] >= test_start_ts].copy()
    return df_train, df_test


def build_models():
    """
    Return a dict of model_name -> (estimator, param_grid, needs_pipeline)
    needs_pipeline = True → we will wrap in a sklearn Pipeline with preprocessing
    """
    models = {}

    # 1. CatBoost (no sklearn preprocessing; it handles categorical features internally)
    catboost_est = CatBoostRegressor(
        loss_function="MAE",
        random_state=42,
        verbose=False,
    )
    catboost_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [300, 600],
        "l2_leaf_reg": [1, 3, 5],
    }
    models["catboost"] = (catboost_est, catboost_grid, False)

    # 2. Random Forest (needs OneHotEncoder for station_id)
    rf_est = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )
    rf_grid = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 5],
    }
    models["random_forest"] = (rf_est, rf_grid, True)

    # 3. Ridge Regression (linear, with scaling)
    ridge_est = Ridge(random_state=42)
    ridge_grid = {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
    }
    models["ridge"] = (ridge_est, ridge_grid, True)

    # 4. Lasso Regression
    lasso_est = Lasso(random_state=42, max_iter=5000)
    lasso_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
    }
    models["lasso"] = (lasso_est, lasso_grid, True)

    return models


def time_series_cv_splits(n_splits=5):
    """
    Use sklearn's TimeSeriesSplit (no shuffling) for time-respecting CV.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def get_feature_sets(target: str):
    base_features = ["station_id", "year", "month", "day_of_week", "is_weekend"]

    if target == "net_flow":
        lag_feats = ["net_flow_1d", "net_flow_7d", "net_flow_28d"]
    elif target == "departures":
        lag_feats = ["dep_1d", "dep_7d", "dep_28d"]
    elif target == "arrivals":
        lag_feats = ["arr_1d", "arr_7d", "arr_28d"]
    else:
        raise ValueError(f"Unknown target: {target}")

    return base_features + lag_feats


def run_model_selection_for_target(df_train: pd.DataFrame, df_test: pd.DataFrame, target: str):
    print(f"\n========== Target: {target} ==========")

    # 1. Select features
    features = get_feature_sets(target)
    all_cols_needed = ["day", target] + features
    df_train = df_train[all_cols_needed].dropna().copy()
    df_test = df_test[all_cols_needed].dropna().copy()

    X_train = df_train[features]
    y_train = df_train[target].values

    X_test = df_test[features]
    y_test = df_test[target].values

    # Identify categorical / numeric features
    cat_cols = ["station_id"]
    num_cols = [c for c in features if c not in cat_cols]

    # Preprocessing for sklearn models
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    models = build_models()
    tscv = time_series_cv_splits(n_splits=5)

    best_model_name = None
    best_model = None
    best_score = np.inf   # MAE (lower is better)
    best_params = None
    
    for name, (base_estimator, param_grid, needs_pipeline) in models.items():
        print(f"\n--- Model: {name} ---")

        if name == "catboost":
            est = clone(base_estimator)
            cat_feature_indices = [X_train.columns.get_loc("station_id")]

            grid = GridSearchCV(
                estimator=est,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                refit=True,
            )

            grid.fit(
                X_train,
                y_train,
                cat_features=cat_feature_indices,
            )

        else:
            est = clone(base_estimator)
            pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", est),
                ]
            )
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)


    ###########################################################
    # SAVE CV RESULTS AS JSON: ./src/models/cv_results/<target>/<model>.json
    ###########################################################
        from json import dump
        cv_results_json = {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in grid.cv_results_.items()
        }

        cv_dir = Path(MODELS_DIR) / "cv_results" / target
        cv_dir.mkdir(parents=True, exist_ok=True)

        json_path = cv_dir / f"{name}.json"
        with open(json_path, "w") as f:
            dump(cv_results_json, f, indent=2)

        print(f"Saved CV results to {json_path}")


        mean_mae = -grid.best_score_
        print(f"{name}: best CV MAE = {mean_mae:.4f}, params = {grid.best_params_}")

        if mean_mae < best_score:
            best_score = mean_mae
            best_model_name = name
            best_model = grid.best_estimator_
            best_params = grid.best_params_


    print(f"\n*** Best model for {target}: {best_model_name} ***")
    print(f"Best CV MAE: {best_score:.4f}")
    print("Best params:", best_params)

    # Evaluate best model on test set
    y_pred_test = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    print(f"Test MAE for {target}: {test_mae:.4f}")

    return best_model_name, best_model, best_score, test_mae


def main():
    df = load_data(DATA_PATH)
    df_train, df_test = make_train_test_split(df, TEST_START)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    results = []

    for target in ["net_flow", "departures", "arrivals"]:
        best_model_name, best_model, cv_mae, test_mae = run_model_selection_for_target(
            df_train, df_test, target
        )

        # Save model
        model_path = Path(MODELS_DIR) / f"best_{target}_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"Saved best {target} model ({best_model_name}) to {model_path}")

        results.append(
            {
                "target": target,
                "best_model": best_model_name,
                "cv_mae": cv_mae,
                "test_mae": test_mae,
                "model_path": str(model_path),
            }
        )

    # Optional: save summary of results
    results_df = pd.DataFrame(results)
    summary_path = Path(MODELS_DIR) / "model_selection_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print("\nModel selection summary saved to:", summary_path)
    print(results_df)


if __name__ == "__main__":
    main()
