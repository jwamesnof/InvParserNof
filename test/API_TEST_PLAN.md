# InvParser App – API Test Plan

## 1. What to test

The purpose of this test plan is to validate the correctness, stability, and reliability of the InvParser API.
The following API endpoints and functionalities will be tested:

### Core API Endpoints

* **POST /extract**

  * Successful invoice extraction from a valid PDF file
  * Validation error when uploading a non-PDF file
  * Handling missing or invalid request payloads

* **GET /invoice/{invoice_id}**

  * Retrieve an existing invoice by ID
  * Error handling when invoice ID does not exist

* **GET /invoice/vendor/{vendor_name}**

  * Retrieve all invoices for a specific vendor
  * Correct response structure including vendor name and extracted fields
  * Handling vendor names with no matching invoices

### Error & Edge Cases

* Invalid path parameters
* Empty database responses
* Database interaction failures (where applicable)

All API responses will be validated for:

* Correct HTTP status codes
* Correct JSON response structure
* Correct business logic behavior

---

## 2. Test design strategy

The selected strategy is **integration testing**.

### Key characteristics

* Tests will be written using **Python unittest** framework
* API endpoints will be tested using **FastAPI TestClient** (no real server like Uvicorn will be started)
* A **real SQLite database** will be used for each test
* Database setup and cleanup will be handled in `setUp()` and `tearDown()` methods
* External dependencies (OCI Document AI service) will be **mocked** using `unittest.mock`

### Why integration testing

* Validates integration between API layer and database layer
* Faster and more deterministic than end-to-end tests
* No dependency on external services

---

## 3. Test environment

Tests will be executed in the following environments:

* **Local development environment**

  * Python 3.x
  * SQLite database
  * Executed via command line: `python -m unittest`

* **CI environment (optional / future)**

  * GitHub Actions
  * Automated execution on each pull request or commit

---

## 4. Success criteria

The test execution will be considered successful if:

* **100% API endpoint coverage** is achieved
* All defined API endpoints have at least one positive and one negative test case
* All tests pass without failures or errors
* API responses conform exactly to the expected status codes and JSON schemas
* Code coverage is as close as possible to **100% of the API layer**, including:

  * Endpoint logic
  * Database interactions
  * Error handling paths

---

## 5. Reporting

Test results will be reported using:

* **Console output** from `unittest` showing:

  * Passed / failed tests
  * Error traces if failures occur

* **Coverage reports (optional enhancement)**

  * Generated using `coverage.py`
  * Displayed as terminal output or HTML report

These reports will allow clear visibility into test status and code coverage quality.
