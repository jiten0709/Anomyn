from datetime import datetime
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field, create_model

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

# 1. safe type mapping registry
# maps JSON/DB string types to actual python objects safely.
TYPE_REGISTRY: Dict[str, Type] = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "datetime": datetime,
    "dict": dict,
    "list": list,
}

# 2. in-memory model cache
# prevents expensive recompilation of the same pydantic model.
_MODEL_CACHE: Dict[str, Type[BaseModel]] = {}

def clear_schema_cache(schema_name: Optional[str] = None) -> None:
    """
    Clears the cached dynamic models. 
    Call this via API when a compliance officer updates a schema.
    """
    if schema_name and schema_name in _MODEL_CACHE:
        del _MODEL_CACHE[schema_name]
        logger.info(f"💬 [schema_engine] Cleared cache for schema: {schema_name}")
    else:
        _MODEL_CACHE.clear()
        logger.info("💬 [schema_engine] Cleared entire schema cache.")

def generate_dynamic_model(schema_name: str, field_definitions: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
    """
    Dynamically generates a Pydantic BaseModel based on a dictionary configuration.
    
    Args:
        schema_name (str): The unique name of the schema (e.g., "FinancialTransaction_v1")
        field_definitions (dict): A dictionary defining the fields.
            Format expected from DB:
            {
                "amount": {"type": "float", "required": True, "description": "Transaction amount"},
                "destination": {"type": "string", "required": False}
            }
            
    Returns:
        Type[BaseModel]: A compiled Pydantic model ready for data validation.
    """
    
    # 3. check cache first (O(1) lookup)
    if schema_name in _MODEL_CACHE:
        logger.debug(f"🔍 [schema_engine] Cache hit for schema: {schema_name}")
        return _MODEL_CACHE[schema_name]

    logger.info(f"💬 [schema_engine] Compiling new dynamic schema: {schema_name}")
    pydantic_fields: Dict[str, Any] = {}

    for field_name, config in field_definitions.items():
        # safely extract type
        raw_type = config.get("type", "string").lower()
        if raw_type not in TYPE_REGISTRY:
            error_msg = f"❌ [schema_engine] Unsupported data type '{raw_type}' for field '{field_name}' in schema '{schema_name}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        python_type = TYPE_REGISTRY[raw_type]
        
        # handle nullability (optionals)
        is_required = config.get("required", True)
        if not is_required:
            python_type = Optional[python_type]
            default_value = None
        else:
            default_value = ... 

        # build the Field object with metadata
        field_kwargs = {
            "description": config.get("description", ""),
        }
        
        # add numeric constraints if they exist in the config
        if "gt" in config: field_kwargs["gt"] = config["gt"]
        if "lt" in config: field_kwargs["lt"] = config["lt"]

        # 4. Construct the tuple Pydantic expects: (Type, FieldInfo)
        pydantic_fields[field_name] = (python_type, Field(default_value, **field_kwargs))

    try:
        # 5. compile the model
        dynamic_model = create_model(schema_name, **pydantic_fields)
        
        # 6. store in cache
        _MODEL_CACHE[schema_name] = dynamic_model
        return dynamic_model
        
    except Exception as e:
        logger.exception(f"🚨 [schema_engine] Failed to generate dynamic model '{schema_name}'.")
        raise RuntimeError(f"❌ [schema_engine] Schema generation failed: {str(e)}")
    