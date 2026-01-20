from sqlalchemy import Column, String, Float
from sqlalchemy.orm import relationship
from app.db.database import Base


class Invoice(Base):
    __tablename__ = "invoices"

    # ✅ Match the task/tests keys (CamelCase)
    InvoiceId = Column(String, primary_key=True, index=True)
    VendorName = Column(String, nullable=True)
    InvoiceDate = Column(String, nullable=True)
    BillingAddressRecipient = Column(String, nullable=True)
    ShippingAddress = Column(String, nullable=True)
    SubTotal = Column(Float, nullable=True)
    ShippingCost = Column(Float, nullable=True)
    InvoiceTotal = Column(Float, nullable=True)

    # Relationships
    items = relationship("Item", back_populates="invoice", cascade="all, delete-orphan")
    confidence = relationship("Confidence", back_populates="invoice", uselist=False, cascade="all, delete-orphan")
