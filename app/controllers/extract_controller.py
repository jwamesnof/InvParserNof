from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import inspect

from app.db.database import get_db
from app.services.oci_service import analyze_document
from app.services.extraction_service import save_extracted_invoice
from app.schemas.extract_response_schema import ExtractResponseSchema

router = APIRouter(tags=["Extraction"])


@router.post("/extract", response_model=ExtractResponseSchema)
async def extract(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # PDF validation
    is_pdf = file.content_type == "application/pdf" or file.filename.lower().endswith(".pdf")
    if not is_pdf:
        raise HTTPException(
            status_code=400,
            detail="Invalid document. Please upload a valid PDF invoice with high confidence."
        )

    # Call OCI (tests may patch this as sync OR async)
    try:
        maybe_result = analyze_document(file)
        oci_result = await maybe_result if inspect.isawaitable(maybe_result) else maybe_result
    except Exception:
        raise HTTPException(status_code=503, detail="The service is currently unavailable. Please try again later.")

    # Validate invoice confidence
    doc_conf = float(oci_result.get("confidence", 0.0))
    if doc_conf < 0.9:
        raise HTTPException(
            status_code=400,
            detail="Invalid document. Please upload a valid PDF invoice with high confidence."
        )

    # Save to DB
    save_extracted_invoice(
        db=db,
        data=oci_result.get("data", {}),
        confidences=oci_result.get("dataConfidence", {}),
        document_confidence=doc_conf,
    )

    # Return response
    return {
        "confidence": doc_conf,
        "data": oci_result.get("data", {}),
        "dataConfidence": oci_result.get("dataConfidence", {}),
    }
