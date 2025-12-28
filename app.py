from fastapi import FastAPI, UploadFile, File
import oci
import base64
from db_util import init_db, save_inv_extraction


app = FastAPI()

# Load OCI config from ~/.oci/config
config = oci.config.from_file()

doc_client = oci.ai_document.AIServiceDocumentClient(config)



@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    # Base64 encode PDF
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    document = oci.ai_document.models.InlineDocumentDetails(
        data=encoded_pdf
    )
    
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

    response = doc_client.analyze_document(request)

    data = {}
    data_confidence = {}
    
    data["Items"] = []

    
    for page in response.data.pages:
        if page.document_fields:
            for field in page.document_fields:
                field_name = field.field_label.name 
                field_confidence = field.field_label.confidence if field.field_label.confidence else None
                field_value = field.field_value.text

                # We only care about fields that contain line item blocks
                if not hasattr(field.field_value, "items") and not hasattr(field.field_value, "_items"):
                    continue

                # Some SDK versions expose .items and some expose ._items
                items_list = getattr(field.field_value, "items", None)
                if not items_list:
                    items_list = getattr(field.field_value, "items", [])


                current_item = {
                    "Description": None,
                    "Name": None,
                    "Quantity": None,
                    "UnitPrice": None,
                    "Amount": None
                }

                for item in items_list:
                    for item_field in item.field_value.items:
                        
                        label = item_field.field_label.name if item_field.field_label else None
                        value = item_field.field_value.text if item_field.field_value else None

                        if not label:
                            continue

                        label_lower = label.lower()

                        if "description" in label_lower:
                            current_item["Description"] = value
                        elif "name" in label_lower:
                            current_item["Name"] = value
                        elif "quantity" in label_lower:
                            current_item["Quantity"] = value
                        elif "unitprice" in label_lower:
                            current_item["UnitPrice"] = value
                        elif "amount" in label_lower:
                            current_item["Amount"] = value

                    # Add only if something was filled
                    if any(v is not None for v in current_item.values()):
                        data["Items"].append(current_item)

                
                data[field_name] = field_value
                data_confidence[field_name] = field_confidence
        

    result = {
        "confidence": field_confidence,
        "data": data,
        "dataConfidence": data_confidence
    }

    # TODO: call to save_inv_extraction(result)    ( no need to change this function)
    
    return result

@app.get('/health')
def health():
    return {'status': 'ok'}

if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8080)