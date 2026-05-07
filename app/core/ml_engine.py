import os
import joblib
import pandas as pd
from typing import Any, Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.core.rules_engine import ValidationResult, ValidationStatus

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

class MLEngine:
    """
    Handles probabilistic anomaly detection for incoming data payloads.
    Uses an Isolation Forest to score transactions based on historical patterns.
    """
    
    def __init__(self, model_dir: str = "data/models", model_name: str = "anomaly_model.joblib"):
        self.model_path = os.path.join(model_dir, model_name)
        self.pipeline: Pipeline = None
        
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
            # 1. feature engineering pipeline
            # we must handle unseen categorical variables gracefully in production
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])

            # 2. build the full pipeline: preprocessing + isolation forest
            self.pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', IsolationForest(contamination=contamination, random_state=42, n_jobs=-1))
            ])

            # 3. train the model
            self.pipeline.fit(df)
            
            # 4. persist to disk
            joblib.dump(self.pipeline, self.model_path)
            logger.info(f"💬 [ml engine] Model trained and saved to {self.model_path}")
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
        
        if not self.pipeline:
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
            prediction = self.pipeline.predict(df_payload)[0]
            
            # decision_function() returns a score. lower/negative means more abnormal.
            anomaly_score = self.pipeline.decision_function(df_payload)[0]

            if prediction == -1:
                logger.debug(f"🔍 [ml engine] Anomaly detected with score {anomaly_score:.3f} for payload: {payload}")
                return ValidationResult(
                    rule_name=rule_name,
                    status=ValidationStatus.FAIL,
                    message=f"Data Anomaly Detected. Statistical outlier score: {anomaly_score:.3f}",
                    severity="WARNING" 
                )
            else:
                logger.debug(f"🔍 [ml engine] Payload scored as normal with score {anomaly_score:.3f}: {payload}")
                return ValidationResult(
                    rule_name=rule_name,
                    status=ValidationStatus.PASS,
                    message=f"Payload within normal parameters. Score: {anomaly_score:.3f}",
                    severity="INFO"
                )

        except Exception as e:
            logger.exception("🚨 [ml engine] Error during anomaly inference.")
            return ValidationResult(
                rule_name=rule_name,
                status=ValidationStatus.ERROR,
                message=f"ML Inference failure: {str(e)}",
                severity="WARNING"
            )
        