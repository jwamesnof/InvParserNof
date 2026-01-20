import base64
import oci
from fastapi import UploadFile
from typing import Dict


async def analyze_document(file: UploadFile) -> Dict:
    """
    Calls OCI Document AI and returns a normalized response:
    {
        "data": {...},
        "dataConfidence": {...},
        "confidence": float
    }
    """

    content = await file.read()
    encoded_content = base64.b64encode(content).decode("utf-8")

    # --- OCI Client Configuration ---
    config = oci.config.from_file()
    ai_client = oci.ai_document.AIServiceDocumentClient(config)

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

    response = ai_client.analyze_document(analyze_request)

    # --- Parse response ---
    extracted_data = {}
    confidences = {}

    for field in response.data.document.fields:
        key = field.field_name
        value = field.field_value
        confidence = field.confidence

        extracted_data[key] = value
        confidences[key] = confidence

    document_confidence = (
        sum(confidences.values()) / len(confidences)
        if confidences else 0
    )

    return {
        "data": extracted_data,
        "dataConfidence": confidences,
        "confidence": document_confidence,
    }
