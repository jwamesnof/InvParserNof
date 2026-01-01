import sqlite3
from fastapi import FastAPI, UploadFile, File
import oci
import base64
import json
import db_util
from fastapi import HTTPException
from db_util import init_db, save_inv_extraction, getInvoiceById, get_invoices_by_vendor  
import time
from datetime import datetime, timezone


app = FastAPI()

# Load OCI config from ~/.oci/config
config = oci.config.from_file()
doc_client = oci.ai_document.AIServiceDocumentClient(config)


@app.post("/extract")
async def extract(file: UploadFile = File(...)):

    # ---------- PDF VALIDATION ----------
    pdf_type = file.content_type == "application/pdf"
    pdf_filename = file.filename.lower().endswith(".pdf")

    if not (pdf_type or pdf_filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid document. Please upload a valid PDF invoice with high confidence."
        )

    # ---------- ENCODE DOCUMENT ----------

    #Processes an uploaded PDF by encoding it to Base64 and submitting it to
    #OCI AI Document for key-value extraction and document classification.
    pdf_bytes = await file.read() 
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")    # Base64 encode PDF

    document = oci.ai_document.models.InlineDocumentDetails(data=encoded_pdf)

    request = oci.ai_document.models.AnalyzeDocumentDetails(
        document=document,
        features=[
            oci.ai_document.models.DocumentFeature(
                feature_type="KEY_VALUE_EXTRACTION"
            ),
            oci.ai_document.models.DocumentClassificationFeature(
                max_results=5
            )
        ]
    )

    # ---------- CALL OCI SAFELY ----------
    try:
        response = doc_client.analyze_document(request)   
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The service is currently unavailable. Please try again later."
        )

    # ---------- DATA STRUCTURES ----------
    data = {}
    data_confidence = {}
    single_item = {}
    extracted_items = []

    # ---------- PARSE PAGES ----------
    for page in response.data.pages:
        if page.document_fields:
            for field in page.document_fields:
                
                field_name = field.field_label.name if field.field_label and field.field_label.name else None
                field_value = get_value(field.field_value)
                
                
                # ---------- DATE FORMAT ----------
                if field_name == "InvoiceDate":
                    field_value = format_date(field_value)

                # ---------- NUMERIC / MONEY FIELDS ----------
                if field_name in (
                    "InvoiceTotal",
                    "SubTotal",
                    "ShippingCost",
                    "Amount",
                    "UnitPrice",
                    "AmountDue"
                ):
                    field_value = amount_format(field_value)

                
                # ---------- CONFIDENCE ----------
                field_confidence = field.field_label.confidence if field.field_label and field.field_label.confidence is not None else 0.0


                # ---------- HANDLE ITEMS ----------
                if field_name == "Items":
                    
                    extracted_items = []    # Reset the list for this invoice/document (avoid accumulating items across pages)

                    for sub_field in field.field_value.items:

                        single_item = {}

                        for sub in sub_field.field_value.items:

                            sub_key = sub.field_label.name if sub.field_label else None
                            sub_value = get_value(sub.field_value)

                            # Clean numeric fields inside items
                            if sub_key in ("Quantity", "UnitPrice", "Amount"):
                                sub_value = amount_format(sub_value)

                            single_item[sub_key] = sub_value

                        extracted_items.append(single_item)

                    field_value = extracted_items
                
                data[field_name] = field_value
                
                if field_name != "Items":
                    data_confidence[field_name] = field_confidence
        
     

        # ---------- DOCUMENT VALIDATION ----------
    if response.data.detected_document_types:
        for doc_type in response.data.detected_document_types:
            confidence_file = doc_type.confidence
            if confidence_file < 0.9:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid document. Please upload a valid PDF invoice with high confidence."
                )

    # ---------- FINAL RESPONSE ----------
    result = {
        "confidence": confidence_file,
        "data": data,
        "dataConfidence": data_confidence,
    }

    save_inv_extraction(result)

    return result


# ---------- HELPERS ----------
def format_date(date_text):
    """
    Converts date like:
    'Mar 06 2012' → '2012-03-06T00:00:00+00:00'
    """
    if not date_text:
        return ""
    try:
        dt = datetime.strptime(date_text.strip(), "%b %d %Y")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return date_text


def amount_format(value):
    """
    Removes $ , and spaces → returns float
    '$58.11' → 58.11
    '4,293.55' → 4293.55
    """
    if not value:
        return ""
    try:
        return float(value.replace("$", "").replace(",", "").strip())
    except Exception:
        return value
    
# OCI may return field values as `.text` in real responses,
# but in unit tests (mock objects) the attribute is `.value`.
# This helper safely supports both to avoid AttributeError. 
# 'field_value': type('obj', (object,), {'value': 'SuperStore'})()
#--AttributeError: 'obj' object has no attribute 'text'--

def get_value(text_value):   
    if not text_value:
        return None
    return getattr(text_value, "text", getattr(text_value, "value", None))


@app.get("/invoice/{invoice_id}")
def getInvoice(invoice_id):
    invoice = db_util.getInvoiceById(invoice_id)    # Retrieve the invoice record from the database using the invoice ID
    if not invoice:       # If no invoice was found, return a 404 Not Found error
        raise HTTPException(
            status_code=404,
            detail="Invoice not found"
        )
    return invoice  # Return the invoice data as the API response

@app.get("/invoices/vendor/{vendor_name}")
def getInvoiceByVendorName(vendor_name):

    invoices = db_util.get_invoices_by_vendor(vendor_name)  ## Retrieve all invoices for the given vendor name from the database
    return {"VendorName": vendor_name if invoices else "Unknown Vendor",
            "TotalInvoices": len(invoices),
            "invoices":invoices}   ## Return the response with vendor details and invoice information


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8080)