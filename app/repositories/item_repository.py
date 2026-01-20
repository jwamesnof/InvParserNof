from sqlalchemy.orm import Session
from typing import List
from app.models.item_model import Item


def create_items(db: Session, items: List[Item]) -> None:
    db.add_all(items)
    db.commit()


def get_items_by_invoice(db: Session, InvoiceId: str):
    return db.query(Item).filter(Item.InvoiceId == str(InvoiceId)).all()

def delete_items_for_invoice(db: Session, InvoiceId: str) -> int:
    count = db.query(Item).filter(Item.InvoiceId == str(InvoiceId)).delete()
    db.commit()
    return count



