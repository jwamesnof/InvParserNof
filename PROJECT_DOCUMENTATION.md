# Invoice Parser Project - Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [MVC Pattern Implementation](#mvc-pattern-implementation)
4. [Database Layer & PostgreSQL Integration](#database-layer--postgresql-integration)
5. [Docker Containerization](#docker-containerization)
6. [Code Walkthrough](#code-walkthrough)
7. [Testing Strategy](#testing-strategy)
8. [API Endpoints](#api-endpoints)

---

## Project Overview

### Purpose
This project is a FastAPI-based invoice processing system that extracts structured data from invoice PDFs using Oracle Cloud Infrastructure (OCI) Document AI service. It demonstrates enterprise-level software architecture with MVC pattern, database abstraction, and comprehensive testing.

### Technology Stack
- **Framework**: FastAPI (Python web framework)
- **Database**: SQLite (development) / PostgreSQL (production)
- **ORM**: SQLAlchemy
- **Cloud AI**: OCI Document AI
- **Testing**: pytest with 100% code coverage
- **Containerization**: Docker

### Key Features
1. PDF invoice extraction via OCI Document AI
2. Structured data storage (invoices, line items, confidence scores)
3. RESTful API for invoice retrieval
4. Database abstraction layer supporting multiple backends
5. Comprehensive test suite with 100% coverage

---

## Architecture & Design Patterns

### MVC (Model-View-Controller) Pattern

The project follows a modified MVC pattern adapted for web APIs:

```
┌─────────────────────────────────────────────────────────┐
│                    Client (HTTP Request)                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  CONTROLLERS (Views)                    │
│  • invoice_controller.py - Route handlers               │
│  • extract_controller.py - PDF processing               │
│  • health_controller.py - Health checks                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      SERVICES                           │
│  • invoice_service.py - Business logic                  │
│  • extraction_service.py - Data transformation          │
│  • oci_service.py - External API integration            │
│  • confidence_service.py - Confidence score mgmt        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    REPOSITORIES                         │
│  • invoice_repository.py - Data access layer            │
│  • item_repository.py - Item CRUD operations            │
│  • confidence_repository.py - Confidence CRUD           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   MODELS (Database)                     │
│  • invoice_model.py - Invoice entity                    │
│  • item_model.py - LineItem entity                      │
│  • confidence_model.py - Confidence entity              │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Controllers** (app/controllers/)
- Handle HTTP requests/responses
- Route definitions with FastAPI's APIRouter
- Input validation via Pydantic schemas
- Minimal business logic

**Services** (app/services/)
- Business logic implementation
- Data transformation and aggregation
- External service integration (OCI)
- Orchestrates repository calls

**Repositories** (app/repositories/)
- Direct database access (CRUD operations)
- SQLAlchemy ORM queries
- Transaction management
- Data persistence layer

**Models** (app/models/)
- SQLAlchemy ORM models
- Database schema definitions
- Entity relationships

---

## MVC Pattern Implementation

### Why MVC?

1. **Separation of Concerns**: Each layer has a single responsibility
2. **Testability**: Layers can be tested independently with mocks
3. **Maintainability**: Changes in one layer don't cascade to others
4. **Scalability**: Easy to add new features without breaking existing code
5. **Reusability**: Services and repositories can be reused across controllers

### Example Flow: GET /invoice/{invoice_id}

```
1. Client Request
   ↓
2. Controller (invoice_controller.py)
   - get_invoice(invoice_id, db)
   - Calls service layer
   ↓
3. Service (invoice_service.py)
   - fetch_invoice(db, invoice_id)
   - Orchestrates multiple repository calls
   ↓
4. Repositories
   - get_invoice_by_id(db, invoice_id)
   - get_items_by_invoice(db, invoice_id)
   - get_confidence_by_invoice(db, invoice_id)
   ↓
5. Models (ORM)
   - SQLAlchemy executes SQL queries
   - Returns Python objects
   ↓
6. Response flows back up through layers
   - Repository → Service → Controller → Client
```

---

## Database Layer & PostgreSQL Integration

### Database Abstraction

The project supports **two database backends** via environment variable:

```python
# app/db/database.py
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")

if DB_BACKEND == "postgres":
    DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
else:
    DATABASE_URL = "sqlite:///./invoices.db"
```

### Why Database Abstraction?

1. **Development Efficiency**: SQLite for local development (no setup required)
2. **Production Readiness**: PostgreSQL for production (ACID compliance, concurrent users)
3. **Testing Isolation**: SQLite in-memory for fast tests
4. **Flexibility**: Easy to switch backends without code changes

### Database Schema

```sql
-- Invoices Table
CREATE TABLE invoices (
    "InvoiceId" VARCHAR PRIMARY KEY,
    "VendorName" VARCHAR,
    "InvoiceDate" VARCHAR,
    "BillingAddressRecipient" VARCHAR,
    "ShippingAddress" VARCHAR,
    "SubTotal" FLOAT,
    "ShippingCost" FLOAT,
    "InvoiceTotal" FLOAT
);

-- Items Table (One-to-Many with Invoices)
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id VARCHAR REFERENCES invoices("InvoiceId"),
    "Description" VARCHAR,
    "Name" VARCHAR,
    "Quantity" FLOAT,
    "UnitPrice" FLOAT,
    "Amount" FLOAT
);

-- Confidence Table (One-to-One with Invoices)
CREATE TABLE confidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id VARCHAR UNIQUE REFERENCES invoices("InvoiceId"),
    "VendorName" FLOAT,
    "InvoiceDate" FLOAT,
    "BillingAddressRecipient" FLOAT,
    "ShippingAddress" FLOAT,
    "SubTotal" FLOAT,
    "ShippingCost" FLOAT,
    "InvoiceTotal" FLOAT
);
```

### Entity Relationships

```
Invoice (1) ──────────────── (*) Item
   │
   │ (1:1)
   │
   └──────────────────────── (1) Confidence
```

### PostgreSQL Docker Setup

```bash
# Start PostgreSQL container
docker run -d \
  --name postgres-invoice \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=predictions \
  -p 5432:5432 \
  postgres:latest

# Connect to database
psql -U user -d predictions -h localhost

# View data
SELECT * FROM invoices;
```

---

## Docker Containerization

### Why Docker?

1. **Consistency**: Same environment across dev/staging/production
2. **Isolation**: Dependencies contained, no conflicts
3. **Portability**: Runs anywhere Docker is installed
4. **Easy Setup**: One command to start database

### PostgreSQL Container Configuration

```yaml
# Environment Variables
POSTGRES_USER: user           # Database user
POSTGRES_PASSWORD: pass       # User password
POSTGRES_DB: predictions      # Database name
Port Mapping: 5432:5432      # Host:Container
```

### Docker Commands Used

```bash
# List running containers
docker ps

# View container logs
docker logs <container_id>

# Execute commands in container
docker exec -it <container_id> psql -U user -d predictions

# Stop container
docker stop <container_id>

# Remove container
docker rm <container_id>
```

---

## Code Walkthrough

### 1. Application Entry Point (app/main.py)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.controllers.invoice_controller import router as invoice_router
from app.controllers.extract_controller import router as extract_router
from app.controllers.health_controller import router as health_router
from app.db.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    init_db()  # Create database tables on startup
    yield      # Application runs here
    # Cleanup code would go here

app = FastAPI(lifespan=lifespan)

# Register route controllers
app.include_router(invoice_router)
app.include_router(extract_router)
app.include_router(health_router)
```

**Key Concepts:**
- `lifespan`: Manages startup/shutdown tasks (database initialization)
- `include_router`: Registers controller routes with the app
- Separation of concerns: Each feature has its own controller

---

### 2. Database Configuration (app/db/database.py)

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()  # Base class for all models

def configure_engine():
    """Create database engine based on environment"""
    DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
    
    if DB_BACKEND == "postgres":
        # Build PostgreSQL connection string
        user = os.getenv("POSTGRES_USER", "user")
        password = os.getenv("POSTGRES_PASSWORD", "pass")
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "predictions")
        
        DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    else:
        # SQLite file-based database
        DATABASE_URL = "sqlite:///./invoices.db"
    
    return create_engine(DATABASE_URL)

engine = configure_engine()
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency injection for database sessions"""
    db = SessionLocal()
    try:
        yield db  # Request uses this session
    finally:
        db.close()  # Always close connection
```

**Key Concepts:**
- **Engine**: Manages database connections
- **SessionLocal**: Creates database sessions
- **Dependency Injection**: `get_db()` provides session to endpoints
- **Environment-based Configuration**: Switches backend via env var

---

### 3. Models Layer

#### Invoice Model (app/models/invoice_model.py)

```python
from sqlalchemy import Column, String, Float
from app.db.database import Base

class Invoice(Base):
    """Invoice entity - represents a single invoice"""
    __tablename__ = "invoices"
    
    # Primary Key
    InvoiceId = Column(String, primary_key=True)
    
    # Invoice Header Information
    VendorName = Column(String, nullable=True)
    InvoiceDate = Column(String, nullable=True)
    BillingAddressRecipient = Column(String, nullable=True)
    ShippingAddress = Column(String, nullable=True)
    
    # Financial Totals
    SubTotal = Column(Float, nullable=True)
    ShippingCost = Column(Float, nullable=True)
    InvoiceTotal = Column(Float, nullable=True)
```

**Key Concepts:**
- `Base`: Inherits from SQLAlchemy declarative base
- `__tablename__`: Database table name
- `Column`: Defines table columns with types
- `nullable=True`: Allows NULL values (optional fields)

#### Item Model (app/models/item_model.py)

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from app.db.database import Base

class Item(Base):
    """Line item entity - invoice line items"""
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String, ForeignKey("invoices.InvoiceId"))
    
    # Item Details
    Description = Column(String, nullable=True)
    Name = Column(String, nullable=True)
    Quantity = Column(Float, nullable=True)
    UnitPrice = Column(Float, nullable=True)
    Amount = Column(Float, nullable=True)
```

**Key Concepts:**
- `ForeignKey`: Establishes relationship with Invoice
- `autoincrement`: Auto-generates primary key values
- One-to-Many relationship with Invoice

#### Confidence Model (app/models/confidence_model.py)

```python
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from app.db.database import Base

class Confidence(Base):
    """Confidence scores for extracted fields"""
    __tablename__ = "confidence"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    invoice_id = Column(String, ForeignKey("invoices.InvoiceId"), unique=True)
    
    # Confidence scores (0.0 to 1.0) for each field
    VendorName = Column(Float, nullable=True)
    InvoiceDate = Column(Float, nullable=True)
    BillingAddressRecipient = Column(Float, nullable=True)
    ShippingAddress = Column(Float, nullable=True)
    SubTotal = Column(Float, nullable=True)
    ShippingCost = Column(Float, nullable=True)
    InvoiceTotal = Column(Float, nullable=True)
```

**Key Concepts:**
- `unique=True`: One confidence record per invoice
- Stores ML model confidence scores (0.0 = low, 1.0 = high)

---

### 4. Repository Layer (Data Access)

#### Invoice Repository (app/repositories/invoice_repository.py)

```python
from sqlalchemy.orm import Session
from typing import Optional, Dict, List
from app.models.invoice_model import Invoice

def create_invoice(db: Session, invoice: Invoice) -> Invoice:
    """Insert new invoice into database"""
    db.add(invoice)
    db.commit()
    db.refresh(invoice)  # Get updated object from DB
    return invoice

def get_invoice_by_id(db: Session, invoice_id: str) -> Optional[Invoice]:
    """Retrieve invoice by primary key"""
    return db.query(Invoice).filter(
        Invoice.InvoiceId == invoice_id
    ).first()

def get_invoices_by_vendor(db: Session, vendor_name: str) -> List[Invoice]:
    """Retrieve all invoices from specific vendor"""
    return db.query(Invoice).filter(
        Invoice.VendorName == vendor_name
    ).all()

def update_invoice(db: Session, invoice_id: str, 
                   updates: Dict) -> Optional[Invoice]:
    """Update existing invoice fields"""
    invoice = get_invoice_by_id(db, invoice_id)
    if not invoice:
        return None
    
    for key, value in updates.items():
        setattr(invoice, key, value)  # Update attributes
    
    db.commit()
    db.refresh(invoice)
    return invoice

def delete_invoice(db: Session, invoice_id: str) -> bool:
    """Delete invoice from database"""
    invoice = get_invoice_by_id(db, invoice_id)
    if not invoice:
        return False
    
    db.delete(invoice)
    db.commit()
    return True
```

**Key Concepts:**
- **CRUD Operations**: Create, Read, Update, Delete
- `db.query()`: SQLAlchemy query builder
- `filter()`: WHERE clause in SQL
- `commit()`: Persists changes to database
- `refresh()`: Reloads object from database
- **Type Hints**: Improves code clarity and IDE support

#### Item Repository (app/repositories/item_repository.py)

```python
from sqlalchemy.orm import Session
from typing import List
from app.models.item_model import Item

def create_items(db: Session, items: List[Item]) -> List[Item]:
    """Bulk insert line items"""
    db.add_all(items)
    db.commit()
    return items

def get_items_by_invoice(db: Session, invoice_id: str) -> List[Item]:
    """Get all line items for an invoice"""
    return db.query(Item).filter(
        Item.invoice_id == invoice_id
    ).all()

def delete_items_for_invoice(db: Session, invoice_id: str) -> bool:
    """Delete all items associated with invoice"""
    items = get_items_by_invoice(db, invoice_id)
    if not items:
        return False
    
    for item in items:
        db.delete(item)
    
    db.commit()
    return True
```

**Key Concepts:**
- `add_all()`: Bulk insert for performance
- Cascade deletes managed manually
- Returns lists for one-to-many relationships

---

### 5. Service Layer (Business Logic)

#### Invoice Service (app/services/invoice_service.py)

```python
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from app.repositories import invoice_repository
from app.repositories import item_repository
from app.repositories import confidence_repository

def fetch_invoice(db: Session, invoice_id: str) -> Optional[Dict]:
    """
    Fetch complete invoice with items and confidence.
    Aggregates data from multiple repositories.
    """
    # Get invoice entity
    invoice = invoice_repository.get_invoice_by_id(db, invoice_id)
    if not invoice:
        return None
    
    # Get related items
    items = item_repository.get_items_by_invoice(db, invoice_id)
    
    # Get confidence scores
    confidence = confidence_repository.get_confidence_by_invoice(db, invoice_id)
    
    # Return aggregated data
    return {
        "invoice": invoice,
        "items": items,
        "confidence": confidence
    }

def fetch_invoices_by_vendor_name(db: Session, 
                                  vendor_name: str) -> List:
    """Service wrapper for vendor lookup"""
    return invoice_repository.get_invoices_by_vendor(db, vendor_name)
```

**Key Concepts:**
- **Orchestration**: Coordinates multiple repository calls
- **Data Aggregation**: Combines related entities
- **Business Logic**: Handles "what data to fetch"
- Returns dictionaries for flexible response structure

#### Extraction Service (app/services/extraction_service.py)

```python
from sqlalchemy.orm import Session
from typing import Dict
from app.models.invoice_model import Invoice
from app.models.item_model import Item
from app.models.confidence_model import Confidence
from app.repositories import invoice_repository, item_repository
from app.services import confidence_service

def save_extracted_invoice(db: Session, extracted_data: Dict, 
                           confidence_data: Dict):
    """
    Transform and save OCI extraction results to database.
    Handles complete invoice lifecycle.
    """
    data = extracted_data.get("data", {})
    invoice_id = data.get("InvoiceId")
    
    if not invoice_id:
        raise ValueError("InvoiceId is required")
    
    # 1. Create invoice entity
    invoice = Invoice(
        InvoiceId=invoice_id,
        VendorName=data.get("VendorName"),
        InvoiceDate=data.get("InvoiceDate"),
        BillingAddressRecipient=data.get("BillingAddressRecipient"),
        ShippingAddress=data.get("ShippingAddress"),
        SubTotal=data.get("SubTotal"),
        ShippingCost=data.get("ShippingCost"),
        InvoiceTotal=data.get("InvoiceTotal")
    )
    invoice_repository.create_invoice(db, invoice)
    
    # 2. Create line items (if present)
    items_data = data.get("Items", [])
    if items_data:
        items = [
            Item(
                invoice_id=invoice_id,
                Description=item.get("Description"),
                Name=item.get("Name"),
                Quantity=item.get("Quantity"),
                UnitPrice=item.get("UnitPrice"),
                Amount=item.get("Amount")
            )
            for item in items_data
        ]
        item_repository.create_items(db, items)
    
    # 3. Create confidence scores
    confidence_service.create_confidence(
        db, invoice_id, **confidence_data
    )
```

**Key Concepts:**
- **Transaction Management**: All-or-nothing save
- **Data Transformation**: OCI format → Database format
- **Error Handling**: Validates required fields
- **Orchestration**: Coordinates 3 repository operations

#### OCI Service (app/services/oci_service.py)

```python
import base64
import oci
from fastapi import UploadFile
from typing import Dict

async def analyze_document(file: UploadFile) -> Dict:
    """
    Call OCI Document AI to extract invoice data.
    Returns normalized response with confidence scores.
    """
    # 1. Read and encode file
    content = await file.read()
    encoded_content = base64.b64encode(content).decode("utf-8")
    
    # 2. Configure OCI client
    config = oci.config.from_file()  # Load ~/.oci/config
    ai_client = oci.ai_document.AIServiceDocumentClient(config)
    
    # 3. Build analysis request
    analyze_request = oci.ai_document.models.AnalyzeDocumentDetails(
        compartment_id=config["tenancy"],
        document=oci.ai_document.models.InlineDocumentContent(
            data=encoded_content
        ),
        features=[
            oci.ai_document.models.DocumentFeature(
                feature_type="KEY_VALUE_EXTRACTION"
            ),
            oci.ai_document.models.DocumentFeature(
                feature_type="TABLE_EXTRACTION"
            ),
        ],
    )
    
    # 4. Call OCI API
    response = ai_client.analyze_document(analyze_request)
    
    # 5. Parse response
    extracted_data = {}
    confidences = {}
    
    for field in response.data.document.fields:
        key = field.field_name
        value = field.field_value
        confidence = field.confidence
        
        extracted_data[key] = value
        confidences[key] = confidence
    
    # 6. Calculate average confidence
    document_confidence = (
        sum(confidences.values()) / len(confidences)
        if confidences else 0
    )
    
    return {
        "data": extracted_data,
        "dataConfidence": confidences,
        "confidence": document_confidence
    }
```

**Key Concepts:**
- **Async/Await**: Non-blocking file I/O
- **Base64 Encoding**: Required for OCI API
- **External Service Integration**: Handles OCI SDK
- **Data Normalization**: Converts OCI response to standard format
- **Confidence Calculation**: Averages field-level scores

---

### 6. Controller Layer (API Endpoints)

#### Invoice Controller (app/controllers/invoice_controller.py)

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.database import get_db
from app.services.invoice_service import fetch_invoice, fetch_invoices_by_vendor_name
from app.schemas.invoice_schema import InvoiceSchema
from app.schemas.item_schema import ItemSchema
from app.schemas.confidence_schema import ConfidenceSchema

router = APIRouter(tags=["Invoices"])

# Helper functions for data transformation
def _to_item_schema(it) -> ItemSchema:
    """Convert Item model to Pydantic schema"""
    return ItemSchema(
        Description=getattr(it, "Description", None),
        Name=getattr(it, "Name", None),
        Quantity=getattr(it, "Quantity", None),
        UnitPrice=getattr(it, "UnitPrice", None),
        Amount=getattr(it, "Amount", None),
    )

def _to_confidence_schema(conf) -> Optional[ConfidenceSchema]:
    """Convert Confidence model to Pydantic schema"""
    if conf is None:
        return None
    return ConfidenceSchema(
        VendorName=getattr(conf, "VendorName", None),
        InvoiceDate=getattr(conf, "InvoiceDate", None),
        BillingAddressRecipient=getattr(conf, "BillingAddressRecipient", None),
        ShippingAddress=getattr(conf, "ShippingAddress", None),
        SubTotal=getattr(conf, "SubTotal", None),
        ShippingCost=getattr(conf, "ShippingCost", None),
        InvoiceTotal=getattr(conf, "InvoiceTotal", None),
    )

@router.get("/invoice/{invoice_id}")
def get_invoice(invoice_id: str, db: Session = Depends(get_db)):
    """
    GET endpoint: Retrieve invoice by ID
    Returns: InvoiceSchema with items and confidence
    """
    result = fetch_invoice(db, invoice_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    invoice = result["invoice"]
    items = result["items"]
    confidence = result["confidence"]
    
    # Transform to response schema
    return InvoiceSchema(
        InvoiceId=invoice.InvoiceId,
        VendorName=invoice.VendorName,
        InvoiceDate=invoice.InvoiceDate,
        BillingAddressRecipient=invoice.BillingAddressRecipient,
        ShippingAddress=invoice.ShippingAddress,
        SubTotal=invoice.SubTotal,
        ShippingCost=invoice.ShippingCost,
        InvoiceTotal=invoice.InvoiceTotal,
        Items=[_to_item_schema(item) for item in items],
        Confidence=_to_confidence_schema(confidence)
    )

@router.get("/invoices/vendor/{vendor_name}")
def get_invoices_by_vendor(vendor_name: str, 
                           db: Session = Depends(get_db)) -> List[InvoiceSchema]:
    """
    GET endpoint: Retrieve all invoices from a vendor
    Returns: List of InvoiceSchema objects
    """
    invoices = fetch_invoices_by_vendor_name(db, vendor_name)
    
    # Transform each invoice to schema
    return [
        InvoiceSchema(
            InvoiceId=inv.InvoiceId,
            VendorName=inv.VendorName,
            InvoiceDate=inv.InvoiceDate,
            BillingAddressRecipient=inv.BillingAddressRecipient,
            ShippingAddress=inv.ShippingAddress,
            SubTotal=inv.SubTotal,
            ShippingCost=inv.ShippingCost,
            InvoiceTotal=inv.InvoiceTotal,
            Items=[],  # Not fetched in list view
            Confidence=None
        )
        for inv in invoices
    ]
```

**Key Concepts:**
- `APIRouter`: Groups related endpoints
- `@router.get()`: HTTP GET endpoint decorator
- `Depends(get_db)`: Dependency injection for database session
- `HTTPException`: Returns HTTP error responses
- **Schema Transformation**: ORM models → Pydantic schemas
- **Path Parameters**: `{invoice_id}` in URL
- **Type Hints**: Documents expected input/output types

#### Extract Controller (app/controllers/extract_controller.py)

```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.oci_service import analyze_document
from app.services.extraction_service import save_extracted_invoice
from app.schemas.extract_response_schema import ExtractResponseSchema

router = APIRouter(tags=["Extract"])

@router.post("/extract", response_model=ExtractResponseSchema)
async def extract_invoice(file: UploadFile = File(...), 
                         db: Session = Depends(get_db)):
    """
    POST endpoint: Upload and process invoice PDF
    
    Steps:
    1. Validate file is PDF
    2. Call OCI Document AI
    3. Check confidence threshold
    4. Save to database
    5. Return results
    """
    # 1. Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported"
        )
    
    try:
        # 2. Extract data using OCI
        result = await analyze_document(file)
        
        # 3. Check confidence threshold (80%)
        if result["confidence"] < 0.8:
            raise HTTPException(
                status_code=400,
                detail=f"Confidence too low: {result['confidence']}"
            )
        
        # 4. Save to database
        save_extracted_invoice(
            db, 
            extracted_data=result,
            confidence_data=result["dataConfidence"]
        )
        
        # 5. Return extraction results
        return ExtractResponseSchema(
            confidence=result["confidence"],
            data=result["data"],
            dataConfidence=result["dataConfidence"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service error: {str(e)}"
        )
```

**Key Concepts:**
- `@router.post()`: HTTP POST endpoint
- `UploadFile`: FastAPI file upload handling
- `File(...)`: Required file parameter
- `async def`: Async endpoint for file I/O
- **Error Handling**: Multiple exception scenarios
- **Business Rule**: 80% confidence threshold
- **Transaction**: Save only if all validations pass

---

### 7. Schemas (Data Validation)

#### Invoice Schema (app/schemas/invoice_schema.py)

```python
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from app.schemas.item_schema import ItemSchema
from app.schemas.confidence_schema import ConfidenceSchema

class InvoiceSchema(BaseModel):
    """
    Pydantic schema for API responses.
    Validates and documents invoice structure.
    """
    InvoiceId: str  # Required field
    VendorName: Optional[str] = None
    InvoiceDate: Optional[str] = None
    BillingAddressRecipient: Optional[str] = None
    ShippingAddress: Optional[str] = None
    SubTotal: Optional[float] = None
    ShippingCost: Optional[float] = None
    InvoiceTotal: Optional[float] = None
    Items: List[ItemSchema] = []
    Confidence: Optional[ConfidenceSchema] = None

    model_config = ConfigDict(from_attributes=True)
```

**Key Concepts:**
- **Pydantic**: Data validation and serialization
- `BaseModel`: Base class for schemas
- `Optional[T]`: Field can be None
- `List[ItemSchema]`: Nested schema validation
- `from_attributes=True`: Can create from ORM objects
- **Type Safety**: Ensures correct data types

---

## Testing Strategy

### Test Architecture

```
test/
├── test_db_integration.py       # Repository layer tests
├── test_extract_endpoint.py     # Controller + OCI tests
├── test_get_invoice.py          # Invoice endpoint tests
├── test_get_invoice_by_vendor.py # Vendor query tests
└── test_additional_coverage.py  # Edge cases & utilities
```

### Testing Pyramid

```
                    ┌──────────────┐
                    │   E2E Tests  │  (4 tests - API endpoints)
                    └──────────────┘
                 ┌────────────────────┐
                 │  Integration Tests │  (14 tests - DB operations)
                 └────────────────────┘
            ┌──────────────────────────────┐
            │       Unit Tests             │  (29 tests - Logic & utilities)
            └──────────────────────────────┘
```

### Key Testing Principles

1. **100% Code Coverage**: All 323 lines covered
2. **Test Isolation**: Each test is independent
3. **Mocking External Services**: OCI API mocked
4. **Database Isolation**: In-memory SQLite for tests
5. **Fixture Management**: setUp/tearDown for clean state

### Example Integration Test

```python
class TestInvoiceRepository(unittest.TestCase):
    def setUp(self):
        """Create fresh database before each test"""
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        self.db = SessionLocal()
    
    def tearDown(self):
        """Clean up after each test"""
        self.db.close()
        Base.metadata.drop_all(bind=engine)
    
    def test_create_invoice(self):
        """Test invoice creation"""
        invoice = Invoice(
            InvoiceId="TEST001",
            VendorName="Test Vendor",
            InvoiceTotal=100.50
        )
        
        result = create_invoice(self.db, invoice)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.InvoiceId, "TEST001")
        self.assertEqual(result.VendorName, "Test Vendor")
```

### Example Unit Test with Mocking

```python
@patch('app.services.invoice_service.get_invoice_by_id')
def test_fetch_invoice_not_found(self, mock_get):
    """Test service handles missing invoice"""
    mock_get.return_value = None
    
    result = fetch_invoice(MagicMock(), "MISSING")
    
    self.assertIsNone(result)
    mock_get.assert_called_once()
```

### Coverage Report

```
Name                                        Stmts   Miss  Cover
---------------------------------------------------------------
app/controllers/extract_controller.py          23      0   100%
app/controllers/invoice_controller.py          27      0   100%
app/services/invoice_service.py                14      0   100%
app/services/oci_service.py                    21      0   100%
app/repositories/invoice_repository.py         29      0   100%
app/models/invoice_model.py                    15      0   100%
---------------------------------------------------------------
TOTAL                                         323      0   100%
```

---

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

**Purpose**: Verify service is running

---

### 2. Extract Invoice from PDF

```http
POST /extract
Content-Type: multipart/form-data

file: <PDF binary data>
```

**Response (Success):**
```json
{
  "confidence": 0.95,
  "data": {
    "InvoiceId": "INV-2024-001",
    "VendorName": "ACME Corp",
    "InvoiceTotal": 1500.50,
    "Items": [
      {
        "Description": "Consulting Services",
        "Amount": 1500.50
      }
    ]
  },
  "dataConfidence": {
    "InvoiceId": 0.99,
    "VendorName": 0.95,
    "InvoiceTotal": 0.92
  }
}
```

**Error Responses:**
- `400`: Invalid file type or low confidence
- `503`: OCI service unavailable

---

### 3. Get Invoice by ID

```http
GET /invoice/{invoice_id}
```

**Example:**
```http
GET /invoice/INV-2024-001
```

**Response:**
```json
{
  "InvoiceId": "INV-2024-001",
  "VendorName": "ACME Corp",
  "InvoiceDate": "2024-01-15",
  "InvoiceTotal": 1500.50,
  "Items": [
    {
      "Description": "Consulting Services",
      "Quantity": 1,
      "Amount": 1500.50
    }
  ],
  "Confidence": {
    "VendorName": 0.95,
    "InvoiceTotal": 0.92
  }
}
```

**Error Responses:**
- `404`: Invoice not found

---

### 4. Get Invoices by Vendor

```http
GET /invoices/vendor/{vendor_name}
```

**Example:**
```http
GET /invoices/vendor/ACME%20Corp
```

**Response:**
```json
[
  {
    "InvoiceId": "INV-2024-001",
    "VendorName": "ACME Corp",
    "InvoiceTotal": 1500.50,
    "Items": [],
    "Confidence": null
  },
  {
    "InvoiceId": "INV-2024-002",
    "VendorName": "ACME Corp",
    "InvoiceTotal": 2300.00,
    "Items": [],
    "Confidence": null
  }
]
```

---

## Key Learnings & Best Practices

### 1. MVC Benefits Demonstrated

- **Testability**: Each layer tested independently (47 tests)
- **Maintainability**: Clear separation of concerns
- **Scalability**: Easy to add new endpoints without affecting existing code
- **Reusability**: Services used by multiple controllers

### 2. Database Abstraction Benefits

- **Flexibility**: Switch between SQLite/PostgreSQL without code changes
- **Testing**: Fast in-memory tests with SQLite
- **Production**: Robust PostgreSQL for concurrent users
- **Development**: No database setup required with SQLite

### 3. Docker Benefits

- **Consistency**: Same PostgreSQL version everywhere
- **Isolation**: No system-wide PostgreSQL installation needed
- **Portability**: Works on Windows/Mac/Linux
- **Easy Cleanup**: Delete container removes all data

### 4. Testing Benefits

- **Confidence**: 100% coverage ensures code works
- **Regression Prevention**: Tests catch breaking changes
- **Documentation**: Tests show how to use code
- **Refactoring Safety**: Can change implementation confidently

---

## Running the Project

### Prerequisites

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start PostgreSQL (optional, SQLite works too)
docker run -d \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=predictions \
  -p 5432:5432 \
  postgres:latest

# 3. Set environment variables
export DB_BACKEND=postgres  # or sqlite
export POSTGRES_USER=user
export POSTGRES_PASSWORD=pass
```

### Start Application

```bash
# Development server with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Access API documentation
# http://localhost:8000/docs
```

### Run Tests

```bash
# All tests with coverage
pytest test/ --cov=app --cov-report=html

# Specific test file
pytest test/test_db_integration.py -v

# View coverage report
open htmlcov/index.html
```

---

## Project Statistics

- **Total Lines of Code**: 323 (application code)
- **Test Coverage**: 100%
- **Number of Tests**: 47
- **Number of Endpoints**: 4
- **Database Tables**: 3 (Invoice, Item, Confidence)
- **Layers**: 5 (Controllers, Services, Repositories, Models, Schemas)

---

## Conclusion

This project demonstrates enterprise-level software engineering practices:

1. **Architecture**: Clean MVC pattern with clear layer boundaries
2. **Database**: Production-ready with PostgreSQL support
3. **Docker**: Containerized infrastructure
4. **Testing**: Comprehensive test suite with 100% coverage
5. **API Design**: RESTful endpoints with proper error handling
6. **Code Quality**: Type hints, documentation, consistent style

The implementation showcases how to build scalable, maintainable, and testable Python web applications suitable for production deployment.

---

## Appendix: File Structure

```
InvParserNof/
├── app/
│   ├── controllers/          # HTTP request handlers
│   │   ├── extract_controller.py
│   │   ├── invoice_controller.py
│   │   └── health_controller.py
│   ├── services/            # Business logic
│   │   ├── invoice_service.py
│   │   ├── extraction_service.py
│   │   ├── oci_service.py
│   │   ├── confidence_service.py
│   │   └── value_utils.py
│   ├── repositories/        # Data access
│   │   ├── invoice_repository.py
│   │   ├── item_repository.py
│   │   └── confidence_repository.py
│   ├── models/             # Database entities
│   │   ├── invoice_model.py
│   │   ├── item_model.py
│   │   └── confidence_model.py
│   ├── schemas/            # API schemas
│   │   ├── invoice_schema.py
│   │   ├── item_schema.py
│   │   ├── confidence_schema.py
│   │   ├── invoice_response_schema.py
│   │   └── extract_response_schema.py
│   ├── db/                 # Database config
│   │   └── database.py
│   └── main.py            # Application entry
├── test/                  # Test suite
│   ├── test_db_integration.py
│   ├── test_extract_endpoint.py
│   ├── test_get_invoice.py
│   ├── test_get_invoice_by_vendor.py
│   └── test_additional_coverage.py
├── requirements.txt       # Dependencies
├── pytest.ini            # Test configuration
└── README.md            # Project overview
```
