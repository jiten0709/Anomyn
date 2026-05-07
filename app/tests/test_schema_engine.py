import pytest
from pydantic import ValidationError
from app.core.schema_engine import generate_dynamic_model, clear_schema_cache, _MODEL_CACHE

@pytest.fixture(autouse=True)
def reset_cache():
    """Fixture to ensure a clean cache before every test."""
    clear_schema_cache()
    yield
    clear_schema_cache()

def test_generate_valid_dynamic_model():
    """Test that a valid schema config generates a working Pydantic model."""
    schema_config = {
        "transaction_id": {"type": "string", "required": True},
        "amount": {"type": "float", "required": True, "gt": 0},
        "destination_country": {"type": "string", "required": False}
    }
    
    DynamicModel = generate_dynamic_model("TestSchema_v1", schema_config)
    
    # test valid data
    valid_data = {"transaction_id": "tx-1", "amount": 100.50}
    instance = DynamicModel(**valid_data)
    
    assert instance.transaction_id == "tx-1"
    assert instance.amount == 100.50
    assert instance.destination_country is None  # optional field defaults to None

def test_missing_required_field_raises_error():
    """Test that missing mandatory fields throw a validation error."""
    schema_config = {"amount": {"type": "float", "required": True}}
    DynamicModel = generate_dynamic_model("TestSchema_v2", schema_config)
    
    with pytest.raises(ValidationError):
        DynamicModel(wrong_field="tx-1")

def test_schema_caching_mechanism():
    """Test that the engine caches models to save CPU cycles."""
    schema_config = {"name": {"type": "string", "required": True}}
    
    # generate once
    Model1 = generate_dynamic_model("CacheTest", schema_config)
    assert "CacheTest" in _MODEL_CACHE
    
    # generate again - should return the exact same class object from memory
    Model2 = generate_dynamic_model("CacheTest", schema_config)
    assert Model1 is Model2

def test_unsupported_data_type_raises_error():
    """Test that the engine fails safely on bad configuration."""
    schema_config = {"weird_field": {"type": "magic_type", "required": True}}
    
    with pytest.raises(ValueError, match="Unsupported data type"):
        generate_dynamic_model("BadSchema", schema_config)