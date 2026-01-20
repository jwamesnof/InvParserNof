# db.py  # pragma: no cover
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Handle both old and new SQLAlchemy versions
try:
    from sqlalchemy.orm import declarative_base
except ImportError:  # pragma: no cover
    from sqlalchemy.ext.declarative import declarative_base  # pragma: no cover

def configure_engine():  # pragma: no cover
    backend = os.getenv("DB_BACKEND", "sqlite").lower()  # pragma: no cover

    if backend == "postgres":
        db_user = os.getenv("POSTGRES_USER", "postgres")
        db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db_host = os.getenv("POSTGRES_HOST", "localhost")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "invparser")

        database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine_obj = create_engine(database_url)
    else:
        database_url = "sqlite:///./invoices.db"
        engine_obj = create_engine(database_url, connect_args={"check_same_thread": False})

    return backend, database_url, engine_obj


DB_BACKEND, DATABASE_URL, engine = configure_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # import models so SQLAlchemy knows them before create_all
    from app.models.invoice_model import Invoice  # noqa: F401
    from app.models.item_model import Item        # noqa: F401
    from app.models.confidence_model import Confidence  # noqa: F401

    Base.metadata.create_all(bind=engine)
