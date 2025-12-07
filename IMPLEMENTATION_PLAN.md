# Implementation Plan – FastAPI + React + Firestore

This document provides a concrete plan to implement the end-to-end system for managing agents, clients, properties, deals, documents, and property files using FastAPI, React, and Firestore on Google Cloud.

## 1. Architecture Overview

- **Backend:** FastAPI deployed to Cloud Run. Uses the `google-cloud-firestore` SDK and optionally the Firebase Admin SDK for token verification.
- **Frontend:** React SPA deployed to Firebase Hosting. Uses Firebase Auth and calls FastAPI with a Bearer token.
- **Database:** Firestore with top-level collections `agents`, `clients`, `properties`, `deals`, `documents`, and `propertyFiles`.
- **Storage:** Firebase Storage / Google Cloud Storage for document files.

## 2. Firestore Collections & Relationships

Use `deals` as the relationship hub: each deal links one agent, one client, and one property. Documents and property files reference the deal (and optionally the related entities for convenience).

### agents
- Path: `agents/{agentId}`
- Fields: `id`, `fullName`, `email`, `phone`, `status`, timestamps.

### clients
- Path: `clients/{clientId}`
- Fields: `id`, `full_name`, `email`, `phone`, `property_file_count`, `created_from`, timestamps.

### properties
- Path: `properties/{propertyId}`
- Fields: `id`, `reference`, `title`, `location`, `type`, `status`, `agentId`, timestamps.

### deals (relationship hub)
- Path: `deals/{dealId}`
- Fields: `id`, `agentId`, `clientId`, `propertyId`, `dealType`, `stage`, `status`, optional `source` and `notes`, timestamps.
- Query patterns: list properties or clients for an agent, deals for a property, and deals for a client.

### documents
- Path: `documents/{documentId}`
- Fields: `id`, `dealId` (primary link), `agentId`, `propertyId`, `clientId`, `documentType`, `filename`, `gcs_path`, `gcs_temp_path`, `file_size`, `content_type`, `processing_status`, `status`, `property_file_id`, `flow_id`, `metadata`, timestamps.
- Queries: documents by `dealId`, uploader `agentId`, `propertyId`, or `clientId`.

### propertyFiles
- Path: `propertyFiles/{propertyFileId}`
- Fields: `id`, `dealId` (primary link), `client_id`, `client_full_name`, `property_id`, `property_reference`, `agent_id`, `transaction_type`, `status`, IDs for SPA/Invoice/ID/Proof of Payment documents, `created_from_document_type`, timestamps.
- Represents the 4-document checklist for a deal.

## 3. Backend Project Structure

Suggested layout under `backend/app/`:

```
backend/app/
  main.py
  config.py
  dependencies.py
  models/
    agents.py, clients.py, properties.py, deals.py, documents.py, property_files.py
  services/
    firestore_client.py, agents_service.py, clients_service.py,
    properties_service.py, deals_service.py, documents_service.py,
    property_files_service.py, ai_document_processing.py
  routers/
    agents_router.py, clients_router.py, properties_router.py,
    deals_router.py, documents_router.py, property_files_router.py
```

### Firestore client
- `firestore_client.py`: instantiate `firestore.Client()` (uses Cloud Run service account by default). For local dev, honor `GOOGLE_APPLICATION_CREDENTIALS`.
- `dependencies.py`: provide `get_db()` dependency to inject Firestore client into routers/services.

### Models
- Pydantic schemas for create/update/read on each collection (e.g., `AgentCreate`, `AgentUpdate`, `AgentOut`). Avoid import try/except wrappers.

### Services
- Encapsulate Firestore CRUD. Example `create_agent`, `update_agent`, `list_agents`. Similar services for clients, properties, deals, documents, and property files.
- Documents service should support filtering by `dealId`, `agentId`, `propertyId`, and `clientId`.

### Routers
- CRUD routers per entity using FastAPI dependency injection. Example `/agents`, `/clients`, `/properties`, `/deals`, `/documents`, `/property-files` with list/get/create/update routes and appropriate HTTP status handling.

## 4. Core Workflows

### Deals
- POST `/deals`: create a deal from `agentId`, `clientId`, `propertyId`, and `dealType`; default `stage` to `LEAD`.
- GET `/deals`: filter by `agentId`, `clientId`, or `propertyId` to support relationship lookups.

### Documents
- Upload flow: either upload file via FastAPI then store to GCS, or upload via Firebase Storage then call FastAPI with the `filePath`.
- Create `documents/{documentId}` with `filePath`, `agentId`, `processing_status` (e.g., `processing` or `unlinked`).
- Background AI step (see below) updates document linkage fields and status.

### Property Files
- Treat a property file as the 4-document checklist for a deal. Either compute on the fly from documents with `dealId`, or persist in `propertyFiles` with explicit doc IDs.
- GET `/property-files?dealId=...` should return checklist status (SPA, Invoice, ID, Proof of Payment) and overall completion flag.

## 5. AI Processing Hook

- `ai_document_processing.py` takes a document ID, reads `filePath`, runs OCR/LLM classification, extracts client/property references, and then:
  1) Finds or creates `client`, `property`, and `deal` (agent + client + property combination).
  2) Updates the document with `dealId`, `clientId`, `propertyId`, `documentType`, and `status="linked"`.
  3) Creates or updates `propertyFiles/{propertyFileId}` for the deal, filling SPA/Invoice/ID/Proof of Payment slots; mark `status="COMPLETE"` when all four are present.
- Trigger via a Cloud Function on Firestore create, or as a background task kicked off by FastAPI after upload.

## 6. Frontend (React)

Suggested structure under `frontend/src/`:

```
api/ (httpClient.ts, agentsApi.ts, clientsApi.ts, propertiesApi.ts,
      dealsApi.ts, documentsApi.ts, propertyFilesApi.ts)
components/ (layout, forms, tables, PropertyFileChecklist.tsx)
pages/
  Agents/, Clients/, Properties/, Deals/, Documents/, PropertyFiles/
hooks/ (useAuth.ts, useAgentDeals.ts, usePropertyFile.ts)
firebase/ (firebaseConfig.ts, auth.ts)
App.tsx, index.tsx
```

### HTTP client with Firebase Auth
Use Axios interceptor to attach the Firebase ID token:

```ts
const api = axios.create({ baseURL: process.env.REACT_APP_API_BASE_URL });
api.interceptors.request.use(async (config) => {
  const auth = getAuth();
  const user = auth.currentUser;
  if (user) {
    const token = await user.getIdToken();
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

### Key pages
- Agents/Clients/Properties list & detail views.
- Deals list/detail pages that pivot relationships via `deals` queries.
- Documents review page filtered by `dealId` or `agentId`.
- Property File detail: `/property-files/:dealId` showing SPA/Invoice/ID/Proof of Payment slots and status badges.

## 7. Deployment

### Backend
- Dockerize FastAPI; run `uvicorn app.main:app` on Cloud Run.
- Service account must allow Firestore and Storage access.
- Env vars: `GOOGLE_CLOUD_PROJECT`, `FIREBASE_PROJECT_ID`, and any AI/OCR keys.

### Frontend
- Build React app and deploy via `firebase deploy --only hosting`.
- Configure `REACT_APP_API_BASE_URL` to point to the Cloud Run service URL.

### Firestore Rules
- Short term: access Firestore only through FastAPI.
- If direct frontend reads are needed, enforce rules using `request.auth.uid` and collection-level checks.

## 8. Next Steps Checklist

1. Scaffold FastAPI app with routers/services for agents, clients, properties, deals, documents, and property files.
2. Implement Firestore client dependency and Pydantic models.
3. Wire document upload endpoint and background AI processing hook to classify and link documents.
4. Build React pages and API clients per entity; add property file checklist UI.
5. Set up Cloud Run deployment and Firebase Hosting; verify Firestore access and auth token flow (ensure service accounts are bound correctly and Firebase ID tokens are accepted end-to-end, including IAM roles for Firestore/Storage and backend token verification paths).

## 9. Firestore optimizations already implemented (and why)

We tuned the agent/client/property flows to avoid slow list endpoints and excessive reads:

- **Count aggregation instead of full scans.** List endpoints (clients, properties, agents, and agent-scoped views) now call a shared `_get_query_count` helper that uses Firestore's aggregation queries. This returns real totals without loading every document; if the environment does not support aggregation, the helper safely falls back to `-1` so the API stays responsive instead of timing out.
- **Chunked `IN` fetches for related entities.** When an agent listing needs the properties or clients tied to their deals, we gather the related IDs and fetch them in batches of 10 using `document_id IN` queries. This eliminates N network round-trips (one per document) and keeps pagination fast even as relationships grow.

How to apply these patterns elsewhere:

1) When adding a new listing endpoint, build the base query first, call `_get_query_count` to compute totals, then apply cursor/offset and `limit`. This keeps list responses accurate without extra scans.

2) When you need to hydrate many referenced docs (e.g., properties for a list of deals), collect their IDs and reuse `_fetch_documents_by_ids(collection, ids)` to retrieve them in minimal batches. Do **not** loop over `.document(id).get()` calls.

Both helpers live in `services/firestore_service.py` and are safe to reuse across new routers or background jobs.
