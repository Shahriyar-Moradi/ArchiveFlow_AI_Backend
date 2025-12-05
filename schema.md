# Firestore Data Model – Real Estate Deals & Documents

This document describes the Firestore schema for managing:

- Agents  
- Clients  
- Properties  
- Deals (relationship hub)  
- Documents  
- Optional: Property Files (4-doc checklist per deal)

The goal is to support:

- Many-to-many relationships between agents, clients, and properties  
- All documents related to a specific deal  
- An AI-driven "Property File" feature that groups SPA, Invoice, ID, and Proof of Payment for each deal

---

## 1. Collections Overview

Top-level collections:

- `agents/{agentId}`
- `clients/{clientId}`
- `properties/{propertyId}`
- `deals/{dealId}` ✅ **Relationship Hub**
- `documents/{documentId}`
- `propertyFiles/{propertyFileId}` (optional but recommended)

**Core idea:**  
`deal` is the central relationship object connecting `agent`, `client`, and `property`.  
All documents for that transaction are linked to the `deal`.

---

## 2. Collection Schemas

### 2.1 `agents` Collection

**Path:** `agents/{agentId}`

Represents a real estate agent in the system.

**Example document:**

```jsonc
// agents/agent_abc
{
  "id": "agent_abc",
  "fullName": "Sara Al Mansoori",
  "email": "sara@oasisproperties.ae",
  "phone": "+971 50 123 4567",
  "created_at": "2025-12-05T10:00:00Z",
  "updated_at": "2025-12-05T10:00:00Z",
  "status": "ACTIVE"  // ACTIVE, INACTIVE, etc.
}
```

**Fields:**
- `id` (string): Unique agent identifier
- `fullName` (string): Agent's full name
- `email` (string, optional): Email address
- `phone` (string, optional): Phone number
- `status` (string): ACTIVE, INACTIVE
- `created_at` (timestamp): Creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Relationships:**
- No direct relationship arrays stored
- Relationships derived via `deals` collection

---

### 2.2 `clients` Collection

**Path:** `clients/{clientId}`

Represents a client/customer in the system.

**Example document:**

```jsonc
// clients/client_xyz
{
  "id": "client_xyz",
  "full_name": "John Michael Doe",
  "email": "john.doe@example.com",
  "phone": "+971 52 987 6543",
  "created_at": "2025-12-05T10:05:00Z",
  "updated_at": "2025-12-05T10:05:00Z",
  "property_file_count": 2,
  "created_from": "document_extraction"  // manual, document_extraction
}
```

**Fields:**
- `id` (string): Unique client identifier
- `full_name` (string): Client's full name
- `email` (string, optional): Email address
- `phone` (string, optional): Phone number
- `property_file_count` (int): Number of property files for this client
- `created_from` (string): Source of creation (manual, document_extraction)
- `created_at` (timestamp): Creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Relationships:**
- No direct relationship arrays stored
- Relationships derived via `deals` collection

---

### 2.3 `properties` Collection

**Path:** `properties/{propertyId}`

Represents a property/listing in the system.

**Example document:**

```jsonc
// properties/property_789
{
  "id": "property_789",
  "reference": "MVTA-2305-DXB",
  "title": "2BR – Marina View Tower A",
  "location": "Dubai Marina, Dubai, UAE",
  "type": "APARTMENT",  // APARTMENT, VILLA, etc.
  "status": "LISTED",  // LISTED, RENTED, SOLD
  "agentId": "agent_abc",  // Optional – primary/owner agent
  "created_at": "2025-12-04T09:00:00Z",
  "updated_at": "2025-12-04T09:00:00Z"
}
```

**Fields:**
- `id` (string): Unique property identifier
- `reference` (string): Property reference code (e.g., unit number, listing code)
- `title` (string, optional): Property title/name
- `location` (string, optional): Property location/address
- `type` (string, optional): Property type (APARTMENT, VILLA, etc.)
- `status` (string, optional): Property status (LISTED, RENTED, SOLD)
- `agentId` (string, optional): Primary agent ID (for convenience)
- `created_at` (timestamp): Creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Relationships:**
- `agentId` stored for convenience (primary agent)
- All agent relationships derived via `deals` collection

---

### 2.4 `deals` Collection (Relationship Hub) ✅

**Path:** `deals/{dealId}`

**This is the central relationship hub** connecting agent + client + property.

Each deal represents one transaction/workflow between an agent, client, and property.

**Example document:**

```jsonc
// deals/deal_123
{
  "id": "deal_123",
  "agentId": "agent_abc",
  "clientId": "client_xyz",
  "propertyId": "property_789",
  "dealType": "RENT",  // RENT, BUY
  "stage": "OFFER",  // LEAD, VIEWING, OFFER, CLOSED, LOST
  "status": "ACTIVE",  // ACTIVE, INACTIVE
  "createdAt": "2025-12-05T11:00:00Z",
  "updatedAt": "2025-12-05T11:30:00Z",
  "source": "Portal",  // Optional: WhatsApp, Portal, Walk-in, etc.
  "notes": "Client prefers sea view, budget flexible."  // Optional
}
```

**Fields:**
- `id` (string): Unique deal identifier
- `agentId` (string): Agent working on this deal
- `clientId` (string): Client involved in this deal
- `propertyId` (string): Property involved in this deal
- `dealType` (string): RENT or BUY
- `stage` (string): LEAD, VIEWING, OFFER, CLOSED, LOST
- `status` (string): ACTIVE, INACTIVE
- `source` (string, optional): Lead source
- `notes` (string, optional): Additional notes
- `createdAt` (timestamp): Creation timestamp
- `updatedAt` (timestamp): Last update timestamp

**Relationships:**
- All documents for this deal: `documents` where `dealId == deal_123`
- Property file for this deal: `propertyFiles` where `dealId == deal_123`

**Query Patterns:**
- Get all properties for an agent: `deals` where `agentId == X` → extract `propertyId`
- Get all clients for an agent: `deals` where `agentId == X` → extract `clientId`
- Get all deals for a property: `deals` where `propertyId == Y`
- Get all deals for a client: `deals` where `clientId == Z`

---

### 2.5 `documents` Collection

**Path:** `documents/{documentId}`

Represents a document (SPA, Invoice, ID, Proof of Payment, etc.) in the system.

**Example document:**

```jsonc
// documents/doc_spa_1
{
  "id": "doc_spa_1",
  "dealId": "deal_123",  // ✅ Primary relationship (NEW)
  "agentId": "agent_abc",  // Convenience field
  "propertyId": "property_789",  // Convenience field
  "clientId": "client_xyz",  // Convenience field
  "documentType": "SPA",  // SPA, INVOICE, ID, PROOF_OF_PAYMENT, etc.
  "filename": "spa_123.pdf",
  "gcs_path": "gs://bucket/deals/deal_123/doc_spa_1.pdf",
  "gcs_temp_path": "temp/flow_123/doc_spa_1.pdf",
  "file_size": 1024000,
  "content_type": "application/pdf",
  "processing_status": "completed",  // pending, completed, failed
  "status": "linked",  // linked, unlinked, needs_review
  "property_file_id": "pf_deal_123",
  "flow_id": "flow-20251128_121507",
  "metadata": {
    "classification": "SPA",
    "ui_category": "SPA",
    "client_full_name_extracted": "John Michael Doe",
    "property_reference_extracted": "MVTA-2305-DXB",
    "document_date": "2025-12-05",
    // ... more extracted fields
  },
  "created_at": "2025-12-05T11:05:00Z",
  "updated_at": "2025-12-05T11:05:00Z"
}
```

**Fields:**
- `id` (string): Unique document identifier
- `dealId` (string, optional): ✅ **Primary relationship** - links to deal
- `agentId` (string, optional): Agent who uploaded (convenience)
- `propertyId` (string, optional): Related property (convenience)
- `clientId` (string, optional): Related client (convenience)
- `documentType` (string, optional): Document type (SPA, INVOICE, ID, PROOF_OF_PAYMENT)
- `filename` (string): Original filename
- `gcs_path` (string): Final GCS storage path
- `gcs_temp_path` (string): Temporary upload path
- `file_size` (int): File size in bytes
- `content_type` (string): MIME type
- `processing_status` (string): pending, completed, failed
- `status` (string): linked, unlinked, needs_review
- `property_file_id` (string, optional): Linked property file ID
- `flow_id` (string): Upload flow ID
- `metadata` (object): Extracted metadata from AI/OCR
- `created_at` (timestamp): Creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Relationships:**
- **Primary:** `dealId` → links to `deals/{dealId}`
- **Convenience:** `agentId`, `propertyId`, `clientId` for direct queries

---

### 2.6 `propertyFiles` Collection

**Path:** `propertyFiles/{propertyFileId}`

Represents a Property File - a 4-document checklist (SPA, Invoice, ID, Proof of Payment) for a deal.

**Example document:**

```jsonc
// propertyFiles/pf_deal_123
{
  "id": "pf_deal_123",
  "dealId": "deal_123",  // ✅ Primary relationship (NEW)
  "client_id": "client_xyz",  // Convenience field
  "client_full_name": "John Michael Doe",
  "property_id": "property_789",  // Convenience field
  "property_reference": "MVTA-2305-DXB",
  "agent_id": "agent_abc",  // Convenience field
  "transaction_type": "BUY",  // BUY, RENT
  "status": "COMPLETE",  // INCOMPLETE, COMPLETE, NEEDS_REVIEW
  "spa_document_id": "doc_spa_1",
  "invoice_document_id": "doc_invoice_1",
  "id_document_id": "doc_id_1",
  "proof_of_payment_document_id": "doc_pop_1",
  "created_from_document_type": "SPA",  // Which document type created this
  "created_at": "2025-12-05T11:10:00Z",
  "updated_at": "2025-12-05T11:20:00Z"
}
```

**Fields:**
- `id` (string): Unique property file identifier
- `dealId` (string, optional): ✅ **Primary relationship** - links to deal
- `client_id` (string): Client ID (convenience)
- `client_full_name` (string): Client full name
- `property_id` (string, optional): Property ID (convenience)
- `property_reference` (string): Property reference code
- `agent_id` (string, optional): Agent ID (convenience)
- `transaction_type` (string): BUY or RENT
- `status` (string): INCOMPLETE, COMPLETE, NEEDS_REVIEW
- `spa_document_id` (string, optional): SPA document ID
- `invoice_document_id` (string, optional): Invoice document ID
- `id_document_id` (string, optional): ID document ID
- `proof_of_payment_document_id` (string, optional): Proof of Payment document ID
- `created_from_document_type` (string): Which document type created this file
- `created_at` (timestamp): Creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Relationships:**
- **Primary:** `dealId` → links to `deals/{dealId}`
- **Convenience:** `client_id`, `property_id`, `agent_id` for direct queries

---

## 3. Relationship Patterns

### 3.1 How Relationships Work

All relationships flow through the `deals` collection:

**Document → Deal:**
- Documents have `dealId` field (primary relationship)
- Also store `agentId`, `propertyId`, `clientId` for convenience

**Property File → Deal:**
- Property files have `dealId` field (primary relationship)
- Also store `client_id`, `property_id`, `agent_id` for convenience

**Deal → Agent/Client/Property:**
- Deal stores `agentId`, `clientId`, `propertyId`
- This is the single source of truth for relationships

### 3.2 Query Examples

**Get all properties for an agent:**
```python
deals = deals_collection.where("agentId", "==", agent_id).stream()
property_ids = [deal.propertyId for deal in deals]
properties = [get_property(pid) for pid in property_ids]
```

**Get all clients for an agent:**
```python
deals = deals_collection.where("agentId", "==", agent_id).stream()
client_ids = [deal.clientId for deal in deals]
clients = [get_client(cid) for cid in client_ids]
```

**Get all documents for a deal:**
```python
documents = documents_collection.where("dealId", "==", deal_id).stream()
```

**Get property file for a deal:**
```python
property_files = property_files_collection.where("dealId", "==", deal_id).stream()
```

---

## 4. Migration Notes

When migrating existing data:

1. **Scan documents** with `agentId`, `propertyId`, `clientId`
2. **Group by unique combinations** (agent + client + property)
3. **Create deals** for each unique combination
4. **Update documents** with `dealId`
5. **Update property files** with `dealId`

See `scripts/migrate_to_deals_schema.py` for the migration script.

---

## 5. Benefits of This Schema

1. **Cleaner relationships:** Single source of truth (deals collection)
2. **Better query performance:** Direct queries on deals instead of scanning documents
3. **Easier maintenance:** Relationships managed in one place
4. **Scalability:** Better suited for many-to-many relationships
5. **Backward compatibility:** Keep existing fields (`agentId`, `propertyId`, `clientId`) for convenience

---

## 6. Indexes

Required Firestore composite indexes:

**Deals:**
- `agentId` + `createdAt` (DESCENDING)
- `clientId` + `createdAt` (DESCENDING)
- `propertyId` + `createdAt` (DESCENDING)
- `status` + `createdAt` (DESCENDING)

**Documents:**
- `dealId` + `created_at` (DESCENDING)

**Property Files:**
- `dealId` + `created_at` (DESCENDING)

**Agents:**
- `created_at` (DESCENDING)

See `firestore.indexes.json` for complete index definitions.
