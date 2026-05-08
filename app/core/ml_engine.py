import os, joblib, threading, pandas as pd, numpy as np
from typing import Any, Dict, List
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
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)
        self.pipeline: Pipeline = None
        self._lock = threading.Lock()
        
        # ensure model directory exists for local testing
        os.makedirs(model_dir, exist_ok=True)
        
        self._load_model()

    def _load_model(self):
        """Loads the trained ML pipeline from disk into memory."""
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"✅ [ml engine] Successfully loaded anomaly detection model from {self.model_path}")
            except Exception as e:
                logger.error(f"❌ [ml engine] Failed to load model at {self.model_path}: {e}")
                self.pipeline = None
        else:
            logger.warning(f"⚠️ [ml engine] No trained model found at {self.model_path}. Engine in cold-start mode.")

    def train_model(self, df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str], contamination: float = 0.01) -> bool:
        """
        Trains the anomaly detection pipeline on historical data.
        
        Args:
            df (pd.DataFrame): Historical transaction data.
            numerical_cols (List[str]): Columns to scale (e.g., 'amount').
            categorical_cols (List[str]): Columns to encode (e.g., 'currency', 'transaction_type').
            contamination (float): The expected proportion of outliers in the dataset.
        """
        logger.info(f"💬 [ml engine] Training Anomaly Detection model on {len(df)} records...")
        
        try:
            # 1. feature engineering pipeline with Imputation to prevent NaN crashes
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

            # 2. build the full pipeline: preprocessing + isolation forest
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', IsolationForest(contamination=contamination, random_state=42, n_jobs=-1))
            ])

            # 3. train the model
            self.pipeline.fit(df)
            
            # 4. swap atomically via lock
            with self._lock:
                self.pipeline = self.pipeline
            
            # determine model filename/path
            save_name = self.model_name
            if not save_name.lower().endswith(".joblib"):
                save_name = f"{save_name}.joblib"
            save_path = os.path.join(self.model_dir, save_name)
            
            joblib.dump(self.pipeline, save_path)
            # update current model_path to the saved model
            with self._lock:
                self.model_path = save_path

            logger.info(f"💬 [ml engine] Model trained and saved to {save_path}")
            return True
            
        except Exception as e:
            logger.exception(f"🚨 [ml engine] Model training failed: {e}")
            return False

    def evaluate_payload(self, payload: Dict[str, Any]) -> ValidationResult:
        """
        Scores a single incoming payload for anomalies.
        Returns a strongly-typed ValidationResult object.
        """
        rule_name = "ml_anomaly_detection"
        with self._lock:
            pipeline = self.pipeline

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
        