import csv
import json
from typing import Any, Dict, AsyncIterator, Iterator
from fastapi import UploadFile, HTTPException

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="utils.log")

# hard limits to prevent Out-Of-Memory (OOM) crashes
MAX_FILE_SIZE_MB = 50
MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

class SafeFileParser:
    """
    Secure utility to parse user-uploaded CSV and JSON datasets.
    Implements memory safeguards and streaming where possible.
    """

    @staticmethod
    async def validate_file_size(file: UploadFile) -> None:
        """
        Validates that the file does not exceed the maximum allowed size without loading the entire file into memory at once.
        """
        # SpooledTemporaryFile allows us to seek to the end to get the size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        
        # reset the cursor back to the beginning for actual reading
        file.file.seek(0)

        if file_size > MAX_BYTES:
            logger.warning(f"⚠️ [file handler] Rejected large file upload: {file.filename} ({file_size} bytes)")
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )
        
        if file_size == 0:
            logger.warning(f"⚠️ [file handler] Rejected empty file upload: {file.filename}")
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    @staticmethod
    async def stream_csv(file: UploadFile) -> AsyncIterator[Dict[str, Any]]:
        """
        Streams a CSV file row by row. 
        Highly memory efficient for large files.
        """
        # we decode the byte stream into a string stream on the fly
        text_stream = (line.decode("utf-8") for line in file.file)
        
        try:
            reader = csv.DictReader(text_stream)
            for row in reader:
                # DictReader values are all strings; we yield them raw
                # our schema_engine and pydantic will handle the type casting!
                yield dict(row)
        except csv.Error as e:
            logger.error(f"❌ [file handler] CSV parsing error: {e}")
            raise HTTPException(status_code=400, detail="Malformed CSV file structure.")

    @staticmethod
    async def parse_json(file: UploadFile) -> AsyncIterator[Dict[str, Any]]:
        """
        Parses a JSON file. Expects an array of JSON objects.
        """
        try:
            content = await file.read()
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.error(f"🚨 [file handler] JSON structure error: Expected an array of objects, got {type(data).__name__}")
                raise HTTPException(
                    status_code=400, 
                    detail="JSON file must contain an array of objects."
                )
                
            for row in data:
                if not isinstance(row, dict):
                    logger.error(f"🚨 [file handler] JSON element error: Expected object, got {type(row).__name__}")
                    raise HTTPException(status_code=400, detail="JSON array elements must be objects.")
                yield row
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ [file handler] JSON decoding error: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON format.")

    @classmethod
    async def process_upload(cls, file: UploadFile) -> AsyncIterator[Dict[str, Any]]:
        """
        Master entry point to validate and route the file to the correct parser.
        Yields one row at a time as a dictionary (async generator).
        """
        await cls.validate_file_size(file)

        filename = file.filename.lower()
        if filename.endswith('.csv'):
            logger.info(f"💬 [file handler] Processing CSV upload: {filename}")
            async for row in cls.stream_csv(file):
                yield row

        elif filename.endswith('.json'):
            logger.info(f"💬 [file handler] Processing JSON upload: {filename}")
            async for row in cls.parse_json(file):
                yield row
            
        else:
            logger.warning(f"⚠️ [file handler] Unsupported file type upload: {filename}")
            raise HTTPException(
                status_code=415, 
                detail="Unsupported media type. Please upload a .csv or .json file."
            )

    @staticmethod
    def _sync_stream_csv(file: UploadFile) -> Iterator[Dict[str, Any]]:
         # synchronous generator for the API to loop through without async overhead on each row
         file.file.seek(0)
         text_stream = (line.decode("utf-8") for line in file.file)
         reader = csv.DictReader(text_stream)
         for row in reader:
             yield dict(row)