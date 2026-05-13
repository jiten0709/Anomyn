from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime, timezone
import json
import hashlib
import uuid

# import our engines
from app.core.schema_engine import generate_dynamic_model
from app.core.rules_engine import RulesEngine, ValidationResult, ValidationStatus
from app.core.ml_engine import MLEngine

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="services.log")

# --- standardized output models ---

class ValidationReport(BaseModel):
    """
    The final, unified report generated for every single payload.
    This fulfills the assignment's requirement for "Generation of validation reports".
    """
    transaction_id: Optional[str] = Field(default=None, description="Unique identifier for the transaction, if available.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of when the validation was performed.")
    overall_status: str = Field(..., description="Overall compliance status: PASS, FAIL, or WARNING.")
    schema_valid: bool = Field(..., description="Indicates if the payload passed the schema validation step.")
    total_rules_executed: int = Field(..., description="Total number of validation rules (deterministic + ML) that were executed.")
    results: List[ValidationResult] = Field(..., description="Detailed results for each individual rule executed, including ML evaluations.")

class ValidationService:
    """
    Orchestrates the flow of data through the Schema, Rules, and ML engines.
    """
    def __init__(self, ml_engine: MLEngine, active_rules: List[Dict[str, Any]], schema_config: Dict[str, Any], schema_name: str):
        """
        In a true production environment, these parameters would be fetched dynamically from a database using a repository pattern. For this prototype, we inject them.
        """
        self.ml_engine = ml_engine
        self.rules_engine = RulesEngine(active_rules=active_rules)
        self.schema_config = schema_config
        self.schema_name = schema_name

    def _coerce_and_clean(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce simple type conversions (e.g., strings to numbers/dates) and clean the payload before validation.
        """
        cleaned: Dict[str, Any] = {}
        for field, meta in self.schema_config.items():
            # check for the field in original, lowercase, and uppercase 
            val = None
            possible_keys = [field, field.lower(), field.upper()]
            for key in possible_keys:
                if key in raw:
                    val = raw[key]
                    break
            
            if isinstance(val, str) and val.strip() == "":
                cleaned[field] = None  
                continue
            if val is None:
                cleaned[field] = None
                continue
            ftype = meta.get("type", "string").lower()
            try:
                if ftype in ('float', 'number', 'integer'):
                    if isinstance(val, str):
                        v = val.replace(",", "").strip()
                    else:
                        v = val
                    cleaned[field] = float(v) 
                    if ftype == 'integer':
                        cleaned[field] = int(cleaned[field])
                elif ftype in ('date', 'datetime'):
                    if isinstance(val, (int, float)):
                        cleaned[field] = datetime.fromtimestamp(float(val), tz=timezone.utc).isoformat()
                    else:
                        cleaned[field] = str(val).strip()
                else:
                    cleaned[field] = str(val).strip()
            except Exception as e:
                logger.debug(f"🔄 [validation svc] Coercion failed for field '{field}' with value '{val}': {e}. Treating as null.")
                cleaned[field] = None

        for k, v in raw.items():
            if k not in cleaned:
                cleaned[k] = v
        return cleaned
        
    def validate_record(self, raw_payload: Dict[str, Any]) -> ValidationReport:
        """
        Passes a single record through the entire compliance pipeline.
        """
        results: List[ValidationResult] = []
        overall_status = "PASS"

        # derive a stable, unique transaction identifier:
        # 1) prefer explicit id-like fields (common variants)
        # 2) fall back to the first column/value (many datasets put id in first column)
        # 3) deterministic hash of the whole payload (stable across re-process)
        # 4) final fallback: random UUID
        id_candidates = ["id", "ID", "Id"]
        transaction_id = None
        for k in id_candidates:
            if k in raw_payload and raw_payload.get(k) not in (None, ""):
                transaction_id = str(raw_payload.get(k))
                break

        if transaction_id is None and isinstance(raw_payload, dict) and raw_payload:
            first_key = next(iter(raw_payload.keys()))
            first_val = raw_payload.get(first_key)
            if first_val not in (None, ""):
                transaction_id = f"{first_key}:{first_val}"

        if transaction_id is None:
            try:
                payload_bytes = json.dumps(raw_payload, sort_keys=True, default=str).encode("utf-8")
                transaction_id = hashlib.sha256(payload_bytes).hexdigest()
            except Exception:
                transaction_id = str(uuid.uuid4())

        # ==========================================
        # STEP 1: dynamic schema validation (integrity check)
        # ==========================================
        DynamicModel = generate_dynamic_model(self.schema_name, self.schema_config)
        
        try:
            # pre-clean/coerce the payload to handle common data issues before strict validation
            prepared = self._coerce_and_clean(raw_payload)
            # this handles missing value detection, type checking
            validated_data = DynamicModel(**prepared)
            clean_payload = validated_data.model_dump()
            schema_valid = True
            
            results.append(ValidationResult(
                rule_name="schema_integrity",
                status=ValidationStatus.PASS,
                message="Data conforms to expected schema types and mandatory fields.",
                severity="CRITICAL"
            ))
            
        except ValidationError as e:
            # if the schema fails, the data is unprocessable. we must abort and fail fast.
            error_details = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            logger.warning(f"⚠️ [validation svc] Schema validation failed for {transaction_id}: {error_details}")
            
            return ValidationReport(
                transaction_id=transaction_id,
                overall_status="FAIL",
                schema_valid=False,
                total_rules_executed=1,
                results=[ValidationResult(
                    rule_name="schema_integrity",
                    status=ValidationStatus.FAIL,
                    message=f"Missing or invalid fields: {error_details}",
                    severity="CRITICAL"
                )]
            )

        # ==========================================
        # STEP 2: deterministic rules (thresholds & cross-field)
        # ==========================================
        rule_results = self.rules_engine.execute_all(clean_payload)
        results.extend(rule_results)
        
        # if any deterministic rule fails, the whole payload fails compliance.
        if any(r.status == ValidationStatus.FAIL for r in rule_results):
            logger.warning(f"⚠️ [validation svc] Deterministic rule failure for {transaction_id}. Marking as FAIL without ML evaluation.")
            overall_status = "FAIL"

        # ==========================================
        # STEP 3: probabilistic anomaly detection (ML)
        # ==========================================
        ml_result = self.ml_engine.evaluate_payload(clean_payload, self.schema_name)
        results.append(ml_result)
        
        # ML failures usually trigger a WARNING for human review, not an automatic FAIL, unless strict regulatory policy demands it.
        if ml_result.status == ValidationStatus.FAIL and overall_status == "PASS":
            logger.warning(f"⚠️ [validation svc] ML anomaly detected for {transaction_id}. Marking as WARNING for human review.")
            overall_status = "WARNING"

        # ==========================================
        # STEP 4: compile final report
        # ==========================================
        logger.debug(f"✅ [validation svc] Validation completed for {transaction_id} with overall status: {overall_status}. Total rules executed: {len(results)}.")
        return ValidationReport(
            transaction_id=transaction_id,
            overall_status=overall_status,
            schema_valid=schema_valid,
            total_rules_executed=len(results),
            results=results
        )
    