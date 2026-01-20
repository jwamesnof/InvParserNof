from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Must match Invoice primary key name
    InvoiceId = Column(String, ForeignKey("invoices.InvoiceId"), index=True)

    Description = Column(String)
    Name = Column(String)
    Quantity = Column(Float)
    UnitPrice = Column(Float)
    Amount = Column(Float)

    invoice = relationship("Invoice", back_populates="items")
