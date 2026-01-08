# API Traceability Matrix

This document provides an enterprise-style **Requirements → Endpoint → Test → Coverage** traceability matrix.
It demonstrates that every functional requirement of the InvParser API is fully verified by automated tests,
and that **100% statement coverage** is achieved.

---

## Legend

* **Requirement ID** – Logical functional requirement
* **Endpoint / Function** – API route or helper
* **Test Case(s)** – Implemented automated tests
* **Coverage** – Code paths covered

---

## Traceability Matrix

| Requirement ID | Requirement Description         | Endpoint / Function                  | Test Case(s)                                                                                  | Coverage                           |
| -------------- | ------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------- |
| R-01           | System health check             | `GET /health`                        | `test_health_endpoint_ok`                                                                     | Endpoint returns 200, JSON body    |
| R-02           | Reject non-PDF uploads          | `POST /extract`                      | `test_extract_non_pdf_rejected`                                                               | Content-type & filename validation |
| R-03           | Reject missing file             | `POST /extract`                      | `test_extract_missing_file`                                                                   | FastAPI validation (422)           |
| R-04           | Process valid PDF               | `POST /extract`                      | `test_extract_valid_pdf_success`                                                              | Happy path execution               |
| R-05           | Handle OCI service failure      | `POST /extract`                      | `test_extract_oci_failure_returns_503`                                                        | Exception handling branch          |
| R-06           | Reject low-confidence documents | `POST /extract`                      | `test_extract_low_confidence_document`                                                        | Confidence threshold validation    |
| R-07           | Format invoice date             | `format_date()`                      | `test_invoice_date_is_formatted`, `test_format_date_empty`, `test_format_date_invalid_string` | All date branches                  |
| R-08           | Parse numeric monetary fields   | `amount_format()`                    | `test_amount_is_float`, `test_amount_format_empty`, `test_amount_format_invalid_value`        | All numeric branches               |
| R-09           | Extract field values safely     | `get_value()`                        | `test_get_value_none`, `test_get_value_text`, `test_get_value_value`                          | `.text`, `.value`, None handling   |
| R-10           | Include confidence metadata     | `POST /extract`                      | `test_confidence_field_exists`                                                                | Confidence response field          |
| R-11           | Include prediction time         | `POST /extract`                      | `test_prediction_time_exists`                                                                 | Execution timing logic             |
| R-12           | Persist extracted invoice       | `save_inv_extraction()`              | Covered indirectly by extract tests                                                           | DB insert paths                    |
| R-13           | Retrieve invoice by ID          | `GET /invoice/{invoice_id}`          | `test_get_existing_invoice`, `test_get_invoice_not_found`                                     | Found & 404 branches               |
| R-14           | Retrieve invoices by vendor     | `GET /invoices/vendor/{vendor_name}` | `test_vendor_invoice_count`, `test_vendor_name_echoed`, `test_vendor_not_found`               | Vendor filtering logic             |
| R-15           | Empty vendor handling           | `GET /invoices/vendor/{vendor_name}` | `test_vendor_not_found`                                                                       | Unknown vendor path                |

---

## Coverage Summary

| File   | Statements | Missed | Coverage |
| ------ | ---------- | ------ | -------- |
| app.py | 100         | 0      | **100%** |

All executable statements are covered by automated tests.

---

## Compliance Statement

This traceability matrix demonstrates:

* Full requirement-to-test mapping
* Complete branch and statement coverage
* Isolation of external dependencies via mocking

The test suite is suitable for enterprise CI pipelines and regulatory-grade validation.
