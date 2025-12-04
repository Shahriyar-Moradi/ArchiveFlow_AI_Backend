# Property Files – Auto-Creation from SPA Documents

## 1. Goal

Add a feature called **“Property Files”** that automatically groups and manages all required documents for a client’s property transaction (buy or rent).

Key points:

- A **Property File is NOT created manually by the user**.
- It is **automatically created from SPA (Sales & Purchase Agreement) documents** using AI extraction.
- Once a Property File exists, **other documents (Invoices, ID, Proof of Payment)** will be automatically matched and attached to it.
- The **Property File details view** is shown when a property is listed and/or when viewing that client’s transactions.

Each **Property File** represents one transaction for one client (for a specific property) and contains up to **4 document categories**:

1. SPA (Sales & Purchase Agreement)  
2. Invoices  
3. ID  
4. Proof of Payment  

These documents may be **uploaded or scanned separately**, but the system should attach them together under a single Property File based on AI-extracted data (client name, property reference, etc.).

---

## 2. Core Concepts & Data Model

Main entities involved:

### 2.1 `Client`

- `id`
- `full_name`
- `email` (optional)
- `phone` (optional)
- `created_at`

### 2.2 `Property` (or `Listing`)

Represents a property that is listed for sale or rent.

- `id`
- `reference` (e.g. unit number, listing code, internal ID)
- `title` / `name`
- `address` (optional)
- `created_at`
- other existing fields as per current system

### 2.3 `Document`

Generic document stored in the system.

- `id`
- `file_url` (or storage reference)
- `original_filename`
- `document_type` (enum: `SPA`, `INVOICE`, `ID`, `PROOF_OF_PAYMENT`, etc.)
- `client_full_name_extracted` (from OCR / AI)
- `property_reference_extracted` (from OCR / AI, optional)
- `client_id` (nullable, link to `Client` if matched)
- `property_id` (nullable, link to `Property` if matched)
- `uploaded_at`
- `status` (e.g. `unlinked`, `linked`, `needs_review`)

### 2.4 `PropertyFile`

Represents a full transaction “file” for one client on one property.

- `id`
- `client_id`
- `client_full_name`
- `property_id`
- `property_reference`
- `transaction_type` (`BUY` or `RENT`)
- `status` (e.g. `INCOMPLETE`, `COMPLETE`, `NEEDS_REVIEW`)
- `created_at`
- `updated_at`

### 2.5 `PropertyFileDocument` (join table)

Links documents to a Property File.

- `id`
- `property_file_id`
- `document_id`
- `document_type` (enum: `SPA`, `INVOICE`, `ID`, `PROOF_OF_PAYMENT`)

**Rules:**

- Each Property File should contain **at most one active document per type** (SPA, Invoice, ID, Proof of Payment).
- If multiple documents of the same type are uploaded, the system can:
  - mark the latest one as the current active doc, and/or
  - keep older ones in history.

---

## 3. High-Level Behavior

### 3.1 SPA Upload → Auto Create Property File

1. A SPA document is uploaded or scanned.
2. The AI pipeline:
   - Classifies the document as `SPA`.
   - Extracts:
     - `client_full_name_extracted`
     - `property_reference_extracted`
     - `transaction_type` (`BUY` or `RENT`, if available)
   - Attempts to match or create:
     - **Client**:
       - If a client with matching `full_name` exists, reuse it.
       - Otherwise, create a new `Client` with `full_name`.
     - **Property**:
       - If a property with matching `reference` exists, reuse it.
       - Otherwise, optionally create a new `Property` (depending on the existing system rules).
3. The system then:
   - Checks if there is already an existing `PropertyFile` for that `(client_id, property_id, transaction_type)` combination.
   - If not, **creates a new Property File**:
     - Sets `client_id`, `client_full_name`, `property_id`, `property_reference`, `transaction_type`.
     - Status = `INCOMPLETE` (since only SPA is present).
4. The SPA document is:
   - Linked to the Property File via `PropertyFileDocument`.
   - `Document.status` updated to `linked`.

From this point, the Property File exists and can be viewed in the UI.

---

### 3.2 Other Documents (Invoices, ID, Proof of Payment) → Auto Attach

When **Invoices**, **ID**, or **Proof of Payment** documents are uploaded:

1. System classifies each document type (`INVOICE`, `ID`, `PROOF_OF_PAYMENT`).
2. AI extracts:
   - `client_full_name_extracted`
   - `property_reference_extracted` (if present)
3. Matching logic:
   - Find `Client` where `full_name` matches `client_full_name_extracted`.
   - Find `Property` where `reference` matches `property_reference_extracted` (if available).
   - Find a `PropertyFile` where:
     - `client_id` matches Client.
     - `property_id` matches Property (if known).
     - and/or `client_full_name` matches and `property_reference` matches.
   - If such a Property File exists:
     - Attach the document to that Property File in the correct `document_type` slot.
     - Update `PropertyFile.status`:
       - If all 4 types present → `COMPLETE`
       - If some missing → `INCOMPLETE`
   - If multiple candidates or low confidence:
     - Mark the document as `needs_review` and present suggestions in the UI.
4. If **no matching Property File** is found:
   - Leave document as `unlinked` but searchable.
   - It can be manually attached from the Property File detail page.

---

### 3.3 Property File Visibility

- When a **property is listed** or when viewing a Property’s detail page:
  - The system should show a **“Property Files” section** for that property.
  - Each Property File belongs to a `(client, property)` pair.
- When viewing a **Client**:
  - Show that client’s Property Files as well.

Users **do not manually create** Property Files; they only review and correct attachments if needed.

---

## 4. Sections in the App

### 4.1 Client Section (Existing)

- Shows client info and a list of related Property Files.
- Example: on the client detail page:
  - “Property Files”
  - Each entry shows property reference & transaction type.

### 4.2 Property Section / Listing Page (Existing)

- On each property’s detail page:
  - Show a “Property Files” subsection.
  - For each Property File:
    - Show client name, transaction type, and document completion status.

### 4.3 Property Files Section (New UI Area)

- A dedicated **Property Files page** listing all auto-created Property Files.
- Useful for admins or back-office review.

---

## 5. Backend Requirements

Implement backend services to support **auto-creation** and **auto-attachment**.

### 5.1 Internal Service: Create Property File from SPA

This can be exposed as an internal function or an API endpoint.

**Endpoint (optional):**

`POST /property-files/from-spa`

**Request body:**

```json
{
  "document_id": "UUID-or-numeric"
}
