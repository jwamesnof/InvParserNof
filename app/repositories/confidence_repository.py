from sqlalchemy.orm import Session
from typing import Optional
from app.models.confidence_model import Confidence


def create_confidence(db: Session, confidence: Confidence) -> Confidence:
    db.add(confidence)
    db.commit()
    db.refresh(confidence)
    return confidence


def get_confidence_by_invoice(db: Session, InvoiceId: str):
    return db.query(Confidence).filter(Confidence.InvoiceId == str(InvoiceId)).first()

def delete_confidence_for_invoice(db: Session, InvoiceId: str) -> bool:
    conf = db.query(Confidence).filter(Confidence.InvoiceId == str(InvoiceId)).first()
    if not conf:
        return False
    db.delete(conf)
    db.commit()
    return True
