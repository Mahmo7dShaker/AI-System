import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

RUNTIME_DIR = "runtime"
DB_PATH = os.path.join(RUNTIME_DIR, "app.db")
os.makedirs(RUNTIME_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()