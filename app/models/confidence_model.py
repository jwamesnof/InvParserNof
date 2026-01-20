from sqlalchemy import Column, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Confidence(Base):
    __tablename__ = "confidences"

    InvoiceId = Column(String, ForeignKey("invoices.InvoiceId"), primary_key=True)

    VendorName = Column(Float)
    InvoiceDate = Column(Float)
    BillingAddressRecipient = Column(Float)
    ShippingAddress = Column(Float)
    AmountDue = Column(Float)
    SubTotal = Column(Float)
    ShippingCost = Column(Float)
    InvoiceTotal = Column(Float)
    VendorNameLogo = Column(Float)

    invoice = relationship("Invoice", back_populates="confidence")
