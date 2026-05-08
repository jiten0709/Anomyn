from typing import Any, Dict, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import io

# import internal modules
from app.utils.file_handler import SafeFileParser
from app.core.profiler import analyze_dataset
from app.core.schema_engine import clear_schema_cache
from app.core.ml_engine import MLEngine
from app.services.validation import ValidationService, ValidationReport

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="api.log")

router = APIRouter(prefix="/api/v1", tags=["Anomyn"])

# --- mock database for prototype ---
# in production, these would be fetched via a db session dependency (e.g., SQLAlchemy)
MOCK_DB_SCHEMA_CONFIG = {}
MOCK_DB_RULES = [
    {
        "rule_name": "high_value_transaction_limit",
        "rule_type": "THRESHOLD",
        "target_field": "amount",
        "parameters": {"operator": "<=", "value": 50000},
        "severity": "CRITICAL"
    }
]

# initialize ml engine globally (singleton pattern for api)
ml_engine_instance = MLEngine()

# --- request/response pydantic models ---

class SchemaConfirmationRequest(BaseModel):
    schema_name: str = Field(..., description="A unique name for the schema configuration")
    field_definitions: Dict[str, Dict[str, Any]] = Field(..., description="The confirmed field definitions with types and constraints")

class SchemaConfirmationResponse(BaseModel):
    message: str = Field(..., description="Confirmation message about the schema")
    schema_name: str = Field(..., description="The name of the confirmed schema")

class ValidationSummaryResponse(BaseModel):
    total_processed: int = Field(..., description="Total number of records processed")
    total_passed: int = Field(..., description="Total number of records that passed all validations")
    total_failed: int = Field(..., description="Total number of records that failed any validation")
    total_warnings: int = Field(..., description="Total number of records that triggered warnings")
    reports: List[ValidationReport] = Field(..., description="Detailed validation reports for each record")

# --- endpoints ---

@router.post("/profile-dataset/", response_model=Dict[str, Any])
async def profile_uploaded_dataset(file: UploadFile = File(...)):
    """
    Step 1: Upload a dataset to infer its schema automatically.
    This fulfills the Zero-Shot / Human-in-the-Loop requirement.
    """
    await SafeFileParser.validate_file_size(file)
    
    try:
        contents = await file.read()
        
        # load into pandas for profiling (handling csv or json)
        if file.filename.endswith('.csv'):
            logger.info(f"💬 [routes] Profiling dataset from file: {file.filename}")
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            logger.info(f"💬 [routes] Profiling dataset from file: {file.filename}")
            df = pd.read_json(io.BytesIO(contents))
        else:
            logger.warning(f"⚠️ [routes] Unsupported file format attempted: {file.filename}")
            raise HTTPException(status_code=415, detail="Unsupported file format.")

        # profile the dataset to infer schema
        inferred_schema = analyze_dataset(df)
        
        return {
            "message": "Dataset profiled successfully. Please review and confirm the schema.",
            "inferred_schema": inferred_schema
        }
        
    except Exception as e:
        logger.error(f"❌ [routes] Error profiling dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")


@router.post("/confirm-schema/", response_model=SchemaConfirmationResponse)
async def confirm_schema(payload: SchemaConfirmationRequest):
    """
    Step 2: Human-in-the-loop confirms the inferred schema.
    Saves it to the "database" and clears the dynamic model cache.
    """
    schema_name = payload.schema_name
    
    # save to mock database
    MOCK_DB_SCHEMA_CONFIG[schema_name] = payload.field_definitions
    
    # critical production step: flush the schema cache so the engine recompiles the new rules
    clear_schema_cache(schema_name)
    
    logger.info(f"💬 [routes] Schema '{schema_name}' confirmed and saved.")
    
    return SchemaConfirmationResponse(
        message="Schema confirmed successfully. Ready for validation.",
        schema_name=schema_name
    )


@router.post("/validate/{schema_name}", response_model=ValidationSummaryResponse)
async def run_compliance_validation(schema_name: str, file: UploadFile = File(...)):
    """
    Step 3: Upload a dataset to validate it against the confirmed schema, deterministic rules, and ML anomaly detection.
    """
    # check if schema exists
    if schema_name not in MOCK_DB_SCHEMA_CONFIG:
        logger.warning(f"⚠️ [routes] Validation attempted with non-existent schema: {schema_name}")
        raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' not found. Please confirm it first.")
        
    schema_config = MOCK_DB_SCHEMA_CONFIG[schema_name]
    
    # instantiate the validation service
    validation_svc = ValidationService(
        ml_engine=ml_engine_instance,
        active_rules=MOCK_DB_RULES,
        schema_config=schema_config,
        schema_name=schema_name
    )

    summary = {
        "total_processed": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_warnings": 0,
        "reports": []
    }

    try:
        # stream the file securely using our SafeFileParser
        
        async for raw_row in SafeFileParser.process_upload(file):
            summary["total_processed"] += 1
            
            # execute full compliance pipeline
            report = validation_svc.validate_record(raw_row)
            summary["reports"].append(report)
            
            # tally metrics
            if report.overall_status == "PASS":
                logger.debug(f"🔍 Record {summary['total_processed']} passed all validations.")
                summary["total_passed"] += 1
            elif report.overall_status == "FAIL":
                logger.debug(f"🔍 Record {summary['total_processed']} failed validation.")
                summary["total_failed"] += 1
            elif report.overall_status == "WARNING":
                logger.debug(f"🔍 Record {summary['total_processed']} triggered a warning.")
                summary["total_warnings"] += 1

        return summary

    except HTTPException as http_ex:
        logger.error(f"❌ [routes] HTTP error during validation: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logger.exception("🚨 [routes] Validation execution failed.")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    
@router.post("/train-model/{schema_name}", response_model=Dict[str, Any])
async def auto_train_ml_model(schema_name: str, file: UploadFile = File(...)):
    """
    Enterprise Zero-Shot Training Pipeline.
    Automatically parses the schema, builds the feature lists, trains the Isolation Forest,
    and hot-reloads the model into live memory.
    """
    # 1. verify the schema exists in our "database"
    if schema_name not in MOCK_DB_SCHEMA_CONFIG:
        logger.exception(f"🚨 [routes] ML training attempted with non-existent schema: {schema_name}")
        raise HTTPException(
            status_code=404, 
            detail=f"Schema '{schema_name}' not found. Please profile and confirm it first."
        )
        
    schema_config = MOCK_DB_SCHEMA_CONFIG[schema_name]
    
    # 2. automated feature engineering (The zero-shot magic)
    numerical_cols = []
    categorical_cols = []
    
    for field_name, config in schema_config.items():
        field_type = config.get("type", "string")
        
        # we don't want to train the ml model on unique IDs (it ruins the math)
        # so we safely ignore fields with 'id' in the name
        if "id" in field_name.lower():
            continue
            
        if field_type in ["float", "integer"]:
            numerical_cols.append(field_name)
        else:
            categorical_cols.append(field_name)
            
    logger.info(f"💬 [routes] Auto-extracted Numerical features: {numerical_cols}")
    logger.info(f"💬 [routes] Auto-extracted Categorical features: {categorical_cols}")

    if not numerical_cols and not categorical_cols:
        logger.exception("🚨 [routes] No usable features found for ML training after schema analysis.")
        raise HTTPException(status_code=400, detail="No usable features found for ML training.")

    # 3. Securely load the dataset into memory
    await SafeFileParser.validate_file_size(file)
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format.")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read training data: {str(e)}")

    # 4. train and hot-reload!
    # by calling this on our global ml_engine_instance, it overwrites the model in memory so the very next api call to /validate will use the newly trained brain.
    success = ml_engine_instance.train_model(
        df=df,
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
        contamination=0.05 # Assuming 5% anomaly rate
    )

    if success:
        logger.info(f"✅ [routes] ML Model trained and hot-reloaded successfully for schema '{schema_name}'.")
        return {
            "message": "Model successfully trained and hot-reloaded into memory.",
            "features_used": {
                "numerical": numerical_cols,
                "categorical": categorical_cols
            },
            "training_samples": len(df)
        }
    else:
        logger.error(f"❌ [routes] ML Model training failed for schema '{schema_name}'. Check server logs for details.")
        raise HTTPException(status_code=500, detail="ML Model training failed. Check server logs.")