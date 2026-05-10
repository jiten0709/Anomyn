import os
import joblib
import threading
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from app.core.rules_engine import ValidationResult, ValidationStatus

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

class MLEngine:
    """
    Handles probabilistic anomaly detection for incoming data payloads.
    Uses an Isolation Forest to score transactions based on historical patterns.
    """
    
    def __init__(self, model_dir: str = "data/models", model_name: str = "anomym_model.joblib"):
        self.model_dir = model_dir
        self.pipelines: Dict[str, Pipeline] = {} # Store multiple models by dataset name
        self._lock = threading.Lock()
        
        # ensure model directory exists for local testing
        os.makedirs(model_dir, exist_ok=True)
        # Load all existing models at startup
        self._load_all_models()

    def _load_all_models(self):
        """Scans the model directory and pre-loads all existing .joblib models."""
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".joblib"):
                dataset_name = filename.replace(".joblib", "")
                filepath = os.path.join(self.model_dir, filename)
                try:
                    pipeline = joblib.load(filepath)
                    self.pipelines[dataset_name] = pipeline
                    logger.info(f"✅ Loaded model for dataset: {dataset_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load model {filename}: {e}")

    def get_model(self, dataset_name: str) -> Optional[Pipeline]:
        """Thread-safe retrieval of a specific dataset's model."""
        with self._lock:
            return self.pipelines.get(dataset_name)

    def train_model(self, df: pd.DataFrame, dataset_name: str, numerical_cols: List[str], categorical_cols: List[str], contamination: float = 0.01) -> bool:
        """
        Trains the anomaly detection pipeline for a specific dataset.
        """
        logger.info(f"💬 [ml engine] Training Anomaly Detection model for '{dataset_name}' on {len(df)} records...")
        
        try:
            # 1. Feature engineering pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_cols),
                    ('cat', cat_pipeline, categorical_cols)
                ])

            # 2. Build pipeline
            new_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', IsolationForest(contamination=contamination, random_state=42, n_jobs=-1))
            ])

            # 3. Train
            new_pipeline.fit(df)
            
            # 4. Save to disk
            save_path = os.path.join(self.model_dir, f"{dataset_name}.joblib")
            joblib.dump(new_pipeline, save_path)
            
            # 5. Swap atomically in memory
            with self._lock:
                self.pipelines[dataset_name] = new_pipeline

            logger.info(f"💬 [ml engine] Model trained and saved to {save_path}")
            return True
            
        except Exception as e:
            logger.exception(f"🚨 [ml engine] Model training failed for {dataset_name}: {e}")
            return False
        
    def evaluate_payload(self, payload: Dict[str, Any], dataset_name: str) -> ValidationResult:
        """
        Scores a single incoming payload for anomalies.
        Returns a strongly-typed ValidationResult object.
        """
        rule_name = "ml_anomaly_detection"
        pipeline = self.get_model(dataset_name)

        if not pipeline:
            logger.debug("🔍 [ml engine] No model available. Skipping ML anomaly check.")
            return ValidationResult(
                rule_name=rule_name,
                status=ValidationStatus.PASS, # or WARNING depending on strictness
                message="Anomaly model not trained/loaded. Bypassing ML check.",
                severity="INFO"
            )

        try:
            # convert single payload to dataFrame 
            df_payload = pd.DataFrame([payload])
            
            # predict() returns 1 for normal, -1 for anomaly
            prediction = pipeline.predict(df_payload)[0]
            
            # decision_function() returns a score. lower/negative means more abnormal.
            anomaly_score = pipeline.decision_function(df_payload)[0]

            # extract feature importance for explainability if flagged
            explanation = {}
            if prediction == -1:
                # get the processed features (2D array)
                transformed_features = pipeline.named_steps['preprocessor'].transform(df_payload)
                # try to get human-friendly feature names from the preprocessor
                try:
                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                except Exception:
                    # fallback generic names
                    feature_names = [f"f{i}" for i in range(transformed_features.shape[1])]

                # flatten single-row transformed vector
                row = transformed_features[0] if transformed_features.ndim == 2 else transformed_features
                # compute absolute magnitude per transformed feature as a simple importance proxy
                abs_vals = np.abs(row)
                # pick top k contributors
                k = min(5, len(abs_vals))
                top_idx = np.argsort(-abs_vals)[:k]
                contributors = []
                for idx in top_idx:
                    contributors.append({
                        "feature": feature_names[idx],
                        "transformed_value": float(row[idx]),
                        "magnitude": float(abs_vals[idx])
                    })

                explanation = {
                    "reason": "Significant deviation detected in transformed feature space",
                    "confidence": float(abs(anomaly_score)),
                    "top_contributors": contributors
                }

            if prediction == 1:
                logger.debug(f"✅ [ml engine] Payload scored as normal with score {anomaly_score:.4f}.")
                return ValidationResult(
                    rule_name=rule_name,
                    status=ValidationStatus.PASS,
                    message=f"Normal transaction with anomaly score {anomaly_score:.4f}",
                    severity="INFO"
                )
            else:
                logger.warning(f"⚠️ [ml engine] Anomaly detected with score {anomaly_score:.4f}. Explanation: {explanation}")
                return ValidationResult(
                    rule_name=rule_name,
                    status=ValidationStatus.FAIL,
                    message=f"Anomaly detected with score {anomaly_score:.4f}",
                    severity="HIGH",
                    explanation=explanation
                )

        except Exception as e:
            logger.exception("🚨 [ml engine] Error during anomaly inference.")
            return ValidationResult(
                rule_name=rule_name,
                status=ValidationStatus.ERROR,
                message=f"ML Inference failure: {str(e)}",
                severity="WARNING"
            )
        