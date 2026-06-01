from typing import Any, Dict, List
from pydantic import BaseModel
from enum import Enum
import pandas as pd

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR" # used if the rule itself is misconfigured

class ValidationResult(BaseModel):
    rule_name: str
    status: ValidationStatus
    message: str
    severity: str

class RulesEngine:
    """
    Executes deterministic business and regulatory rules against validated data payloads.
    """
    
    def __init__(self, active_rules: List[Dict[str, Any]]):
        """
        Initializes the engine with rules fetched from the database.
        Expected rule format:
        {
            "rule_name": "high_value_limit",
            "rule_type": "THRESHOLD",
            "target_field": "amount",
            "parameters": {"operator": "<=", "value": 50000},
            "severity": "CRITICAL"
        }
        """
        self.rules = active_rules
        
        # strategy map to route evaluation to the correct function based on rule_type
        self._evaluators = {
            "THRESHOLD": self._evaluate_threshold,
            "CROSS_FIELD": self._evaluate_cross_field
        }

    def execute_all(self, payload: Dict[str, Any]) -> List[ValidationResult]:
        """
        Runs all active rules against a single data payload.
        """
        results = []
        for rule in self.rules:
            rule_type = rule.get("rule_type")
            evaluator = self._evaluators.get(rule_type)
            
            if not evaluator:
                logger.error(f"❌ [rules engine] Unsupported rule_type '{rule_type}' for rule '{rule['rule_name']}'")
                results.append(ValidationResult(
                    rule_name=rule["rule_name"],
                    status=ValidationStatus.ERROR,
                    message=f"Engine lacks evaluator for type: {rule_type}",
                    severity="WARNING"
                ))
                continue
                
            # execute the specific strategy
            try:
                result = evaluator(rule, payload)
                results.append(result)
            except Exception as e:
                logger.exception(f"🚨 [rules engine] Rule execution failed for {rule['rule_name']}")
                results.append(ValidationResult(
                    rule_name=rule["rule_name"],
                    status=ValidationStatus.ERROR,
                    message=f"Execution crash: {str(e)}",
                    severity=rule.get("severity", "WARNING")
                ))
                
        return results

    # --- RULE STRATEGIES ---

    def _evaluate_threshold(self, rule: Dict[str, Any], payload: Dict[str, Any]) -> ValidationResult:
        """Evaluates mathematical thresholds (>, <, >=, <=, ==)."""
        field = rule["target_field"]
        params = rule["parameters"]
        operator = params.get("operator")
        limit = params.get("value")
        
        # if field is completely missing (and not caught by schema), skip threshold check
        if field not in payload or payload[field] is None:
            logger.debug(f"🔍 [rules engine] Field '{field}' is null/missing in payload; skipping threshold check for rule '{rule['rule_name']}'")
            return ValidationResult(
                rule_name=rule["rule_name"],
                status=ValidationStatus.PASS,
                message=f"Field '{field}' null/missing; threshold not applicable.",
                severity="INFO"
            )

        actual_value = payload[field]
        passed = False
        
        try:
            if operator == ">": passed = actual_value > limit
            elif operator == ">=": passed = actual_value >= limit
            elif operator == "<": passed = actual_value < limit
            elif operator == "<=": passed = actual_value <= limit
            elif operator == "==": passed = actual_value == limit
            else:
                logger.error(f"❌ [rules engine] Unknown operator '{operator}' in rule '{rule['rule_name']}'")
                raise ValueError(f"🚨 [rules engine] Unknown operator {operator}")
        except TypeError:
             logger.error(f"🚨 [rules engine] Type error comparing {actual_value} with {limit} for rule '{rule['rule_name']}'")
             return ValidationResult(
                rule_name=rule["rule_name"],
                status=ValidationStatus.ERROR,
                message=f"Type mismatch: Cannot compare {type(actual_value)} with {type(limit)}",
                severity=rule["severity"]
            )

        if passed:
            logger.debug(f"🔍 [rules engine] Threshold passed for rule '{rule['rule_name']}' with value {actual_value} {operator} {limit}")
            return ValidationResult(
                rule_name=rule["rule_name"],
                status=ValidationStatus.PASS,
                message="Threshold check passed.",
                severity=rule["severity"]
            )
        else:
            logger.warning(f"⚠️ [rules engine] Threshold failed for rule '{rule['rule_name']}': {actual_value} {operator} {limit}")
            return ValidationResult(
                rule_name=rule["rule_name"],
                status=ValidationStatus.FAIL,
                message=f"Failure: {field} value ({actual_value}) violated '{operator} {limit}'.",
                severity=rule["severity"]
            )

    def _evaluate_cross_field(self, rule: Dict[str, Any], payload: Dict[str, Any]) -> ValidationResult:
        """
        Evaluates dependencies between fields.
        """
        primary_field = rule["target_field"]
        params = rule["parameters"]
        condition_val = params.get("condition_value")
        dependent_field = params.get("must_not_be_null")

        # check if the trigger condition is met
        if payload.get(primary_field) == condition_val:
            # the condition is met, so the dependent field MUST exist and not be null
            if payload.get(dependent_field) is None:
                return ValidationResult(
                    rule_name=rule["rule_name"],
                    status=ValidationStatus.FAIL,
                    message=f"Failure: Because {primary_field} is '{condition_val}', {dependent_field} cannot be null.",
                    severity=rule["severity"]
                )
            
        logger.debug(f"🔍 [rules engine] Cross-field condition not triggered or passed for rule '{rule['rule_name']}'")
        return ValidationResult(
            rule_name=rule["rule_name"],
            status=ValidationStatus.PASS,
            message="Cross-field validation passed.",
            severity=rule["severity"]
        )
    
    @staticmethod
    def generate_smart_threshold_rules(
        df: pd.DataFrame,
        field_name: str,
        common_thresholds: List[float] = None,
        min_samples: int = 30,
        sigma_multiplier: float = 3.0,
        max_fraction_tol: float = 0.99,
        min_std_ratio: float = 0.01
    ):
        """
        Generate a non-trivial upper threshold rule for `field_name`.
        Returns None when a meaningful rule should NOT be created (e.g., too few samples,
        near-zero variance, or computed threshold is essentially the observed max).

        Heuristics:
        - Require at least `min_samples` non-null observations.
        - Exclude extreme outliers via IQR before computing mean/std.
        - Use mean + sigma_multiplier * std for threshold.
        - Snap to a provided common_thresholds list (choose smallest >= computed limit).
        - Reject rule if the resulting threshold is within `max_fraction_tol` of observed max.
        - Reject rule if std is negligible relative to mean (controlled by min_std_ratio).
        """
        series = pd.to_numeric(df[field_name], errors='coerce').dropna()
        n = len(series)
        if n < min_samples:
            logger.debug(f"🔍 [rule-gen] Not enough samples for {field_name} ({n} < {min_samples}); skipping rule generation.")
            return None

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        # filter extreme IQR outliers
        filtered = series[~((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR)))]
        if filtered.empty:
            logger.debug(f"🔍 [rule-gen] After IQR filtering no data remains for {field_name}; skipping.")
            return None

        mean_val = filtered.mean()
        std_val = filtered.std()
        if std_val == 0:
            logger.debug(f"🔍 [rule-gen] Zero variance for {field_name}; skipping.")
            return None
        if abs(mean_val) > 0 and (std_val / abs(mean_val)) < min_std_ratio:
            logger.debug(f"🔍 [rule-gen] Low relative std for {field_name} (std/mean={std_val/abs(mean_val):.4f} < {min_std_ratio}); skipping.")
            return None

        upper_limit = mean_val + (sigma_multiplier * std_val)
        observed_max = series.max()

        # If computed limit is effectively the observed max (within tolerance), skip trivial rule
        if observed_max != 0 and upper_limit >= observed_max * max_fraction_tol:
            logger.debug(f"🔍 [rule-gen] Computed limit {upper_limit:.2f} is near observed max {observed_max:.2f}; skipping trivial rule.")
            return None

        # Snap to common thresholds if provided (choose smallest >= upper_limit)
        if common_thresholds:
            valid_thresholds = sorted([t for t in common_thresholds if t >= upper_limit])
            if valid_thresholds:
                snapped = valid_thresholds[0]
                if observed_max != 0 and snapped >= observed_max * max_fraction_tol:
                    logger.debug(f"🔍 [rule-gen] Snapped threshold {snapped} is near observed max {observed_max:.2f}; skipping.")
                    return None
                upper_limit = snapped

        rule_value = round(float(upper_limit), 2)
        logger.info(f"✅ [rule-gen] Generated threshold for {field_name}: <= {rule_value} (samples={n})")
        return {
            "rule_name": f"{field_name}_upper_limit",
            "rule_type": "THRESHOLD",
            "target_field": field_name,
            "parameters": {"operator": "<=", "value": rule_value},
            "severity": "CRITICAL"
        }
    