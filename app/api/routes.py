from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="routes.log")

router = APIRouter()

@router.post("/upload-dataset/")
async def upload_and_profile_dataset(file: UploadFile = File(...)):
    """
    Accepts a user dataset, parses it, and returns an inferred schema 
    for the user to confirm.
    """
    logger.info(f"📂 Received file upload: {file.filename} ({file.content_type})")
    if not file.filename.endswith(('.csv', '.json')):
        raise HTTPException(status_code=400, detail="Only CSV or JSON files are supported.")
    
    # Read file content securely into memory
    contents = await file.read()
    
    try:
        # Pass to pandas for initial profiling
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_json(io.BytesIO(contents))
            
        # In a real app, you would pass 'df' to app/core/profiler.py here
        inferred_columns = list(df.columns)
        row_count = len(df)
        logger.info(f"✅ Successfully processed file: {file.filename} with {row_count} rows and columns: {inferred_columns}")
        return {
            "message": "File uploaded successfully.",
            "rows_detected": row_count,
            "inferred_columns": inferred_columns,
            "next_step": "Please confirm column data types to generate validation schema."
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")