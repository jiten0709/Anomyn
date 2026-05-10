from typing import Any, Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Form
from pydantic import BaseModel, Field
import pandas as pd
import io
import os
import time
import json

# import internal modules
from app.utils.file_handler import SafeFileParser
from app.core.profiler import analyze_dataset
from app.core.schema_engine import clear_schema_cache
from app.core.ml_engine import MLEngine
from app.services.validation import ValidationService, ValidationReport

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="api.log")

# env variables
from dotenv import load_dotenv
load_dotenv()
ROW_LIMIT = int(os.getenv("ROW_LIMIT", 1000))

router = APIRouter(prefix="/api/v1", tags=["Anomyn"])

MOCK_DB_RULES = [
    {
        "rule_name": "high_value_transaction_limit",
        "rule_type": "THRESHOLD",
        "target_field": "amount",
        "parameters": {"operator": "<=", "value": 50000},
        "severity": "CRITICAL"
    }
]

# persist/load path for schemas (simple persistence for prototype)
SCHEMA_DIR = os.path.join(os.getcwd(), "data", "schemas")
os.makedirs(SCHEMA_DIR, exist_ok=True)

def load_schema_on_demand(schema_name: str) -> Optional[Dict[str, Any]]:
    """Loads a schema directly from disk if it exists, otherwise returns None."""
    schema_path = os.path.join(SCHEMA_DIR, f"{schema_name}.json")
    if os.path.exists(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                logger.info(f"💬 Loading schema '{schema_name}' from disk.")
                return json.load(fh)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load schema '{schema_name}' from disk: {e}")
            return None
    return None
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
    
    # derive suggested schema name from filename
    base_name = os.path.splitext(file.filename)[0]
    suggested_schema_name = base_name.strip().replace(" ", "_")

    # check if schema already exists (load directly from disk)
    existing_schema = load_schema_on_demand(suggested_schema_name)
    if existing_schema is not None:
        logger.info(f"💬 [routes] Existing schema '{suggested_schema_name}' loaded from disk.")
        return {
            "message": f"Existing schema '{suggested_schema_name}' loaded successfully. No profiling needed.",
            "schema_name": suggested_schema_name,
            "inferred_schema": existing_schema,
            "row_count": None,
            "column_count": len(existing_schema)
        }
    
    # no existing schema, proceed with profiling
    try:
        contents = await file.read()
        
        # load into pandas for profiling
        if file.filename.endswith('.csv'):
            logger.info(f"💬 [routes] Profiling new dataset from CSV file: {file.filename}")
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            logger.info(f"💬 [routes] Profiling new dataset from JSON file: {file.filename}")
            df = pd.read_json(io.BytesIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            logger.info(f"💬 [routes] Profiling new dataset from Excel file: {file.filename}")
            df = pd.read_excel(io.BytesIO(contents))
        else:
            logger.warning(f"⚠️ [routes] Unsupported file format attempted: {file.filename}")
            raise HTTPException(status_code=415, detail="Unsupported file format. Supported: CSV, JSON, XLSX.")

        if df.empty:
            logger.warning("⚠️ [routes] Uploaded file is empty or contains no data rows.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # profile the dataset to infer schema
        inferred_schema = analyze_dataset(df)
        
        return {
            "message": "New dataset profiled successfully. Please review and confirm the schema.",
            "suggested_schema_name": suggested_schema_name,
            "inferred_schema": inferred_schema,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    
    except pd.errors.EmptyDataError:
        logger.error("❌ [routes] File appears to be empty or malformed.")
        raise HTTPException(status_code=400, detail="File is empty or malformed.")
    except Exception as e:
        logger.error(f"❌ [routes] Error profiling dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@router.post("/confirm-schema/", response_model=SchemaConfirmationResponse)
async def confirm_schema(payload: SchemaConfirmationRequest):
    """
    Step 2: Human-in-the-loop confirms the inferred schema.
    Saves it to the "database" and clears the dynamic model cache.
    """
    schema_name = payload.schema_name.strip().replace(" ", "_")
    
    if not schema_name:
        logger.error("❌ [routes] Schema name cannot be empty.")
        raise HTTPException(status_code=400, detail="Schema name cannot be empty.")
    
    if not payload.field_definitions:
        logger.error("❌ [routes] Field definitions cannot be empty.")
        raise HTTPException(status_code=400, detail="Field definitions cannot be empty.")
    
    # Persist to disk only
    try:
        os.makedirs(SCHEMA_DIR, exist_ok=True)
        schema_path = os.path.join(SCHEMA_DIR, f"{schema_name}.json")
        with open(schema_path, "w", encoding="utf-8") as fh:
            json.dump(payload.field_definitions, fh, indent=2)
        logger.info(f"💬 Persisted schema to disk: {schema_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to persist schema to disk: {e}")
        raise HTTPException(status_code=500, detail="Failed to save schema.")

    # flush the schema cache so the engine recompiles the new rules
    try:
        clear_schema_cache(schema_name)
        logger.info(f"💬 [routes] Schema cache cleared for '{schema_name}'.")
    except Exception as e:
        logger.warning(f"⚠️ [routes] Failed to clear schema cache: {e}")
    
    logger.info(f"✅ [routes] Schema '{schema_name}' confirmed and saved.")
    
    return SchemaConfirmationResponse(
        message="Schema confirmed successfully. Ready for validation.",
        schema_name=schema_name
    )

@router.post("/validate/{schema_name}", response_model=ValidationSummaryResponse)
async def run_compliance_validation(schema_name: str, file: UploadFile = File(...)):
    """
    Step 3: Upload a dataset to validate it against the confirmed schema, deterministic rules, and ML anomaly detection.
    """
    # load schema directly from disk
    schema_config = load_schema_on_demand(schema_name)
    if schema_config is None:
        logger.warning(f"⚠️ [routes] Validation attempted with non-existent schema: {schema_name}")
        raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' not found. Please confirm it first.")
    
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
        "reports": [],
        "processing_time_seconds": 0.0
    }
    start_time = time.time() 

    try:
        # stream the file securely using our SafeFileParser
        async for raw_row in SafeFileParser.process_upload(file):
            if summary["total_processed"] >= ROW_LIMIT:
                logger.warning(f"⚠️ [routes] Row limit ({ROW_LIMIT}) reached. Stopping validation.")
                break

            summary["total_processed"] += 1
            try:
                # execute full compliance pipeline
                report = validation_svc.validate_record(raw_row)
                summary["reports"].append(report)
                
                # tally metrics
                if report.overall_status == "PASS":
                    summary["total_passed"] += 1
                elif report.overall_status == "FAIL":
                    summary["total_failed"] += 1
                elif report.overall_status == "WARNING":
                    summary["total_warnings"] += 1
            except Exception as row_e:
                logger.error(f"❌ [routes] Error validating row {summary['total_processed']}: {row_e}")

        summary["processing_time_seconds"] = time.time() - start_time
        logger.info(f"✅ [routes] Validation completed for {summary['total_processed']} records in {summary['processing_time_seconds']:.2f}s.")
        return summary

    except HTTPException as http_ex:
        logger.error(f"❌ [routes] HTTP error during validation: {http_ex.detail}")
        raise http_ex
    except Exception as e:
        logger.exception("🚨 [routes] Validation execution failed.")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
    
@router.post("/train-model/", response_model=Dict[str, Any])
async def train_model(
    file: UploadFile = File(...),
    numerical_cols: str = Form(""),
    categorical_cols: str = Form(""),
    force: bool = Form(False)
):
    """
    Trains an ML model based on the uploaded file name.
    Behavior:
    - Derives schema_name / dataset_name from uploaded filename (filename without ext).
    - Validates that a confirmed schema exists for that name.
    - Auto-extracts features from the confirmed schema unless explicit numerical_cols / categorical_cols are provided.
    - If a model for that dataset already exists and force==False, returns early.
    - Otherwise trains, saves and hot-reloads the model for that dataset_name.
    """
    # derive dataset / schema name from filename
    base_name = os.path.splitext(file.filename)[0]
    schema_name = base_name.strip().replace(" ", "_")
    dataset_name = schema_name  # model will be saved as <dataset_name>.joblib

    # verify the schema exists (load directly from disk)
    schema_config = load_schema_on_demand(schema_name)
    if schema_config is None:
        logger.exception(f"🚨 [routes] ML training attempted with non-existent schema: {schema_name}")
        raise HTTPException(
            status_code=404,
            detail=f"Schema '{schema_name}' not found. Please profile and confirm it first."
        )
    schema_fields = set(schema_config.keys())

    # if user provided explicit column lists via form, parse and use them (overrides auto-extract)
    def parse_cols_param(s: str) -> List[str]:
        return [c.strip() for c in s.split(",") if c.strip()]

    explicit_num_cols = parse_cols_param(numerical_cols) if numerical_cols else []
    explicit_cat_cols = parse_cols_param(categorical_cols) if categorical_cols else []

    if explicit_num_cols or explicit_cat_cols:
        # respect user override but ensure columns exist in the confirmed schema
        num_cols = [c for c in explicit_num_cols if c in schema_fields]
        cat_cols = [c for c in explicit_cat_cols if c in schema_fields]

        dropped_num = set(explicit_num_cols) - set(num_cols)
        dropped_cat = set(explicit_cat_cols) - set(cat_cols)
        if dropped_num or dropped_cat:
            logger.warning(f"⚠️ [routes] Some explicitly provided columns not in schema and will be ignored: numeric_dropped={dropped_num}, cat_dropped={dropped_cat}")
    else:
        # automated feature engineering (zero-shot)
        num_cols = []
        cat_cols = []
        for field_name, config in schema_config.items():
            field_type = config.get("type", "string")
            if "id" in field_name.lower():
                continue
            if field_type in ["float", "integer", "number"]:
                num_cols.append(field_name)
            else:
                cat_cols.append(field_name)

    logger.info(f"💬 [routes] Features chosen for training -> numerical: {num_cols}, categorical: {cat_cols}")

    if not num_cols and not cat_cols:
        logger.exception("🚨 [routes] No usable features found for ML training after schema analysis / overrides.")
        raise HTTPException(status_code=400, detail="No usable features found for ML training.")

    # check existing model
    existing = ml_engine_instance.get_model(dataset_name)
    if existing and not force:
        logger.info(f"💬 [routes] Model for dataset '{dataset_name}' already exists. Use force=True to retrain.")
        return {
            "message": f"Model for '{dataset_name}' already exists. Set force=true to retrain.",
            "model_exists": True,
            "features_used": {"numerical": num_cols, "categorical": cat_cols}
        }

    # load dataset securely into memory
    await SafeFileParser.validate_file_size(file)
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            logger.info(f"💬 [routes] Loading training dataset from file: {file.filename}")
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith('.json'):
            logger.info(f"💬 [routes] Loading training dataset from file: {file.filename}")
            df = pd.read_json(io.BytesIO(contents.decode('utf-8')))
        else:
            logger.exception(f"🚨 [routes] Unsupported file format attempted for training: {file.filename}")
            raise HTTPException(status_code=415, detail="Unsupported file format.")
    except Exception as e:
        logger.exception("🚨 [routes] Failed to read training data.")
        raise HTTPException(status_code=500, detail=f"Failed to read training data: {str(e)}")

    if df.empty:
        logger.exception("🚨 [routes] Uploaded training file contained no rows.")
        raise HTTPException(status_code=400, detail="Training dataset is empty.")

    # ensure the dataset actually contains the selected features
    missing_cols = [c for c in (num_cols + cat_cols) if c not in df.columns]
    if missing_cols:
        logger.exception(f"🚨 [routes] Training file missing required columns: {missing_cols}")
        raise HTTPException(status_code=400, detail=f"Training file missing required columns: {missing_cols}")

    # train and hot-reload model (dataset-scoped)
    success = ml_engine_instance.train_model(
        df=df,
        dataset_name=dataset_name,
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        contamination=0.05  # default; could be parameterized later
    )

    if success:
        logger.info(f"✅ [routes] ML Model trained and hot-reloaded successfully for schema '{schema_name}'.")
        return {
            "message": "Model successfully trained and hot-reloaded into memory.",
            "dataset": dataset_name,
            "features_used": {
                "numerical": num_cols,
                "categorical": cat_cols
            },
            "training_samples": len(df),
            "model_path": os.path.join(ml_engine_instance.model_dir, f"{dataset_name}.joblib")
        }

    logger.error(f"❌ [routes] ML Model training failed for schema '{schema_name}'. Check server logs for details.")
    raise HTTPException(status_code=500, detail="ML Model training failed. Check server logs.")
