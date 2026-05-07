from typing import Any, Dict, List
from pydantic import BaseModel
from enum import Enum

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
            logger.info(f"💬 [rules engine] Field '{field}' is null/missing in payload; skipping threshold check for rule '{rule['rule_name']}'")
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
        Example: IF 'transaction_type' == 'INTERNATIONAL', THEN 'destination_country' MUST NOT BE NULL.
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
    