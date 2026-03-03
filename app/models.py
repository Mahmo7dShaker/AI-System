from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from datetime import datetime
from .db import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    original_name = Column(String, nullable=False)
    stored_name = Column(String, nullable=False)
    stored_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class DatasetMapping(Base):
    __tablename__ = "dataset_mappings"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    mapping_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ProcessedRun(Base):
    __tablename__ = "processed_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    mapping_id = Column(Integer, ForeignKey("dataset_mappings.id"), nullable=False, index=True)
    processed_path = Column(String, nullable=False)
    rows_before = Column(Integer, nullable=False)
    rows_after = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)