import pytest
from app.core.rules_engine import RulesEngine, ValidationStatus

@pytest.fixture
def rules_engine():
    """Fixture providing a configured RulesEngine."""
    rules = [
        {
            "rule_name": "max_limit",
            "rule_type": "THRESHOLD",
            "target_field": "amount",
            "parameters": {"operator": "<=", "value": 50000},
            "severity": "CRITICAL"
        },
        {
            "rule_name": "international_check",
            "rule_type": "CROSS_FIELD",
            "target_field": "transaction_type",
            "parameters": {
                "condition_value": "INTERNATIONAL",
                "must_not_be_null": "destination_country"
            },
            "severity": "CRITICAL"
        }
    ]
    return RulesEngine(active_rules=rules)

def test_threshold_rule_pass(rules_engine):
    """Test amount under the threshold passes."""
    payload = {"amount": 10000.0}
    results = rules_engine.execute_all(payload)
    
    # find the specific rule result
    threshold_result = next(r for r in results if r.rule_name == "max_limit")
    assert threshold_result.status == ValidationStatus.PASS

def test_threshold_rule_fail(rules_engine):
    """Test amount over the threshold fails."""
    payload = {"amount": 75000.0}
    results = rules_engine.execute_all(payload)
    
    threshold_result = next(r for r in results if r.rule_name == "max_limit")
    assert threshold_result.status == ValidationStatus.FAIL

def test_cross_field_rule_fail(rules_engine):
    """Test missing dependent field triggers a failure."""
    payload = {
        "transaction_type": "INTERNATIONAL",
        "destination_country": None  # this violates the rule
    }
    results = rules_engine.execute_all(payload)
    
    cf_result = next(r for r in results if r.rule_name == "international_check")
    assert cf_result.status == ValidationStatus.FAIL

def test_cross_field_rule_irrelevant_condition(rules_engine):
    """Test that if the trigger condition isn't met, the rule passes safely."""
    payload = {
        "transaction_type": "DOMESTIC",
        "destination_country": None  # permitted for domestic
    }
    results = rules_engine.execute_all(payload)
    
    cf_result = next(r for r in results if r.rule_name == "international_check")
    assert cf_result.status == ValidationStatus.PASS

def test_engine_handles_corrupted_rule_gracefully():
    """Test that a bad rule configuration doesn't crash the engine."""
    bad_rules = [{
        "rule_name": "broken_rule",
        "rule_type": "UNKNOWN_TYPE",
        "target_field": "amount",
        "severity": "WARNING"
    }]
    engine = RulesEngine(active_rules=bad_rules)
    results = engine.execute_all({"amount": 10})
    
    assert results[0].status == ValidationStatus.ERROR
    assert "Engine lacks evaluator" in results[0].message