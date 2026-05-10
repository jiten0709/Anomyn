import pandas as pd
import os
from typing import Any, Dict

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="core.log")

# env variables
from dotenv import load_dotenv
load_dotenv()

PROFILER_SAMPLE_SIZE = int(os.getenv("PROFILER_SAMPLE_SIZE", 100))

def map_pandas_dtype_to_engine_type(pd_dtype: str) -> str:
    """
    Maps pandas data types to our internal engine's string representations.
    Fallback is always 'string' for safety.
    """
    dtype_str = str(pd_dtype).lower()
    
    if 'int' in dtype_str:
        return 'integer'
    elif 'float' in dtype_str:
        return 'float'
    elif 'bool' in dtype_str:
        return 'boolean'
    elif 'datetime' in dtype_str:
        return 'datetime'
    else:
        return 'string'

def analyze_dataset(df: pd.DataFrame, sample_size: int = PROFILER_SAMPLE_SIZE) -> Dict[str, Dict[str, Any]]:
    """
    Profiles a pandas DataFrame to infer a base schema configuration.
    
    Args:
        df (pd.DataFrame): The ingested dataset.
        sample_size (int): Max rows to analyze to save memory/CPU on massive files.
        
    Returns:
        dict: A proposed schema configuration ready for user confirmation.
    """
    if df.empty:
        logger.error("🚨 [profiler] Attempted to profile an empty dataset.")
        raise ValueError("Cannot profile an empty dataset.")

    # 1. optimize performance: profile only a sample if the dataset is huge
    if len(df) > sample_size:
        logger.info(f"💬 [profiler] Dataset large ({len(df)} rows). Profiling top {sample_size} rows.")
        df_sample = df.head(sample_size).copy()
    else:
        df_sample = df.copy()

    # 2. attempt to parse dates automatically (pandas often reads dates as strings initially)
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object':
            try:
                # use a fast subset check before applying to the whole column
                pd.to_datetime(df_sample[col].dropna().head(10))
                df_sample[col] = pd.to_datetime(df_sample[col])
                logger.debug(f"🔍 [profiler] Column '{col}' successfully parsed as datetime.")
            except (ValueError, TypeError):
                logger.debug(f"🔍 [profiler] Column '{col}' does not appear to be datetime. Keeping as string.")
                pass # if it fails, leave it as a string/object

    inferred_schema: Dict[str, Dict[str, Any]] = {}

    # 3. build the schema dict
    for column in df_sample.columns:
        series = df_sample[column]
        
        # determine nullability
        null_count = series.isna().sum()
        is_required = bool(null_count == 0)
        
        # determine base type
        engine_type = map_pandas_dtype_to_engine_type(series.dtype)

        # build field configuration
        field_config: Dict[str, Any] = {
            "type": engine_type,
            "required": is_required,
            "description": f"Inferred type: {engine_type}. Nulls detected: {null_count}"
        }

        # 4. optional: infer basic numerical bounds for thresholds
        if engine_type in ['integer', 'float'] and not series.isna().all():
            min_val = series.min()
            # if the minimum value is > 0, propose a "greater than 0" threshold
            if min_val >= 0:
                field_config["gt"] = 0

        inferred_schema[column] = field_config

    logger.info(f"✅ [profiler] Successfully profiled {len(inferred_schema)} columns.")
    return inferred_schema
