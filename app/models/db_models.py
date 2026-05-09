import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Boolean, JSON, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# --- configuration tables ---

class DataDictionary(Base):
    """
    Stores the dynamically inferred schemas confirmed by the Human-in-the-Loop.
    Replaces hardcoded Pydantic classes.
    """
    __tablename__ = "data_dictionaries"

    id = Column(Integer, primary_key=True, index=True)
    schema_name = Column(String(100), unique=True, nullable=False, index=True)
    
    # stores the JSON object: {"amount": {"type": "float", "required": true}, ...}
    field_definitions = Column(JSON, nullable=False) 
    
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))


class ValidationRule(Base):
    """
    stores deterministic regulatory rules (thresholds, cross-field dependencies).
    """
    __tablename__ = "validation_rules"

    id = Column(Integer, primary_key=True, index=True)
    rule_name = Column(String(100), unique=True, nullable=False, index=True)
    rule_type = Column(String(50), nullable=False) # e.g., "THRESHOLD", "CROSS_FIELD"
    
    # the schema this rule applies to (e.g., "financial_v1")
    target_schema = Column(String(100), ForeignKey("data_dictionaries.schema_name"), nullable=True)
    target_field = Column(String(50), nullable=False)
    
    # JSON configuration: {"operator": "<=", "value": 50000}
    parameters = Column(JSON, nullable=False)
    
    severity = Column(String(20), nullable=False) # "WARNING", "CRITICAL"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# --- audit & reporting tables ---

class ValidationLog(Base):
    """
    The master audit trail. Every payload that passes through the ValidationService gets a record here, regardless of whether it passed or failed.
    """
    __tablename__ = "validation_logs"

    # UUIDs are better for distributed, high-throughput systems than sequential IDs
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # the identifier from the upstream operational system (if available)
    transaction_id = Column(String(100), nullable=True, index=True)
    
    schema_name = Column(String(100), nullable=False)
    overall_status = Column(String(20), nullable=False, index=True) # "PASS", "FAIL", "WARNING"
    
    # metric to monitor the performance of our python engines
    execution_time_ms = Column(Float, nullable=True) 
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # relationship to specific failures
    failures = relationship("ValidationFailure", back_populates="log", cascade="all, delete-orphan")


class ValidationFailure(Base):
    """
    Child table to ValidationLog.
    If a payload fails 3 different rules, 3 rows are inserted here pointing to the same log_id.
    """
    __tablename__ = "validation_failures"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(36), ForeignKey("validation_logs.id"), nullable=False, index=True)
    
    rule_name = Column(String(100), nullable=False) # e.g., "schema_integrity", "max_transfer_limit", "ml_anomaly_detection"
    severity = Column(String(20), nullable=False)
    
    # The specific reason for failure (e.g., "Missing field: destination_country")
    error_message = Column(String(500), nullable=False)
    
    # Workflow flag for compliance officers
    is_resolved = Column(Boolean, default=False, nullable=False) 
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    log = relationship("ValidationLog", back_populates="failures")
    