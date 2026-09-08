"""
Retraining module triggered by Airflow when data drift is detected.
Trains a candidate model on updated data and logs to MLflow under the 'Staging' tag.
"""
import os
import sqlite3
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.mlflow_utils.tracking import setup_mlflow, log_model_with_metrics, compute_dataset_hash
from src.model.train_baseline import FEATURE_NAMES, save_baseline_distributions


def fetch_production_logs(db_path: str = "logs.db", limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetches recent logged feature vectors. Prioritizes those with actual ground truth feedback."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        # Prefer actual_value if available, fallback to prediction
        query = f"SELECT {', '.join(FEATURE_NAMES)}, COALESCE(actual_value, prediction) as MedHouseVal FROM inference_logs ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"Warning: Could not fetch production logs from {db_path}: {e}")
        return None


def run_retraining_pipeline(
    db_path: str = "logs.db",
    existing_model_path: str = "production_model.joblib",
    n_estimators: int = 50,
    max_depth: int = 6,
    random_state: int = 101,
    update_baseline_distribution: bool = True
) -> Tuple[Any, Dict[str, float], str]:
    """
    Runs incremental retraining on purely the newly logged data using XGBoost's xgb_model parameter,
    slashing compute costs compared to full baseline retraining.
    """
    logged_df = fetch_production_logs(db_path=db_path)
    
    if logged_df is None or len(logged_df) < 50:
        print("Not enough new data for incremental learning. Falling back to full baseline retrain...")
        # Fallback to full dataset for demonstration purposes
        base_data = fetch_california_housing(as_frame=True)
        df_train = base_data.frame
        existing_model = None
    else:
        print(f"Executing Incremental Learning on {len(logged_df)} new production records...")
        df_train = logged_df
        existing_model = joblib.load(existing_model_path) if os.path.exists(existing_model_path) else None

    X = df_train[FEATURE_NAMES]
    y = df_train["MedHouseVal"]
    
    dataset_hash = compute_dataset_hash(X.values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    print(f"Training candidate XGBRegressor incrementally...")
    candidate_model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    if existing_model:
        candidate_model.fit(X_train, y_train, xgb_model=existing_model)
    else:
        candidate_model.fit(X_train, y_train)
    
    # Evaluate candidate
    y_pred = candidate_model.predict(X_test)
    rmse = float(root_mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    }
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "retrain_samples": len(df_train),
        "incremental": bool(existing_model)
    }
    
    print(f"Candidate Evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Log candidate to MLflow under 'Staging'
    setup_mlflow(experiment_name="california_housing_drift")
    run_id = log_model_with_metrics(
        model=candidate_model,
        model_name="CaliforniaHousingRegressor",
        metrics=metrics,
        params=params,
        dataset_hash=dataset_hash,
        stage_tag="Staging"
    )
    
    # Save locally for FastAPI shadow deployment
    joblib.dump(candidate_model, "staging_model.joblib")
    print("Saved candidate model to staging_model.joblib for shadow evaluation")
    
    if update_baseline_distribution:
        save_baseline_distributions(X, output_path="baseline_distribution.json")
        print("Updated reference baseline feature distributions after retraining.")
        
    return candidate_model, metrics, run_id


if __name__ == "__main__":
    run_retraining_pipeline()
