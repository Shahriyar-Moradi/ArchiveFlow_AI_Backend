# Testing Checklist - GCP Integration

## ✅ Pre-Testing Setup

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed from requirements.txt
- [ ] `.env` file configured with all required variables
- [ ] GCP service account key file present (`voucher-storage-key.json`)
- [ ] GCS bucket created
- [ ] Firestore database initialized

## ✅ Code Quality Tests

### Syntax and Import Tests
```bash
cd rizanai/backend

# Test 1: Python syntax check
python3 -m py_compile main.py
# Expected: No output (success)

# Test 2: Config loading
python3 -c "from config import settings; print('✅ Config loaded')"
# Expected: ✅ Config loaded

# Test 3: Service imports
python3 -c "from services.document_processor import DocumentProcessor; print('✅ DocumentProcessor')"
python3 -c "from services.firestore_service import FirestoreService; print('✅ FirestoreService')"
python3 -c "from services.task_queue import TaskQueue; print('✅ TaskQueue')"
python3 -c "from gcs_service import GCSVoucherService; print('✅ GCSVoucherService')"
# Expected: All services import successfully

# Test 4: Main application import
python3 -c "from main import app; print('✅ Main app loaded')"
# Expected: ✅ Main app loaded
```

## ✅ Service Initialization Tests

### Start the Server
```bash
# Terminal 1: Start server
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Expected output should show:
# 🚀 Starting Document Processing Backend with GCP...
# ✅ Document Processor (Anthropic OCR) initialized
# ✅ Firestore service initialized
# ✅ GCS service initialized
# ✅ Task queue initialized
# ✅ Batch processor started
```

## ✅ API Endpoint Tests

### Terminal 2: Run these tests

#### Test 1: Health Check
```bash
curl http://localhost:8080/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-25T...",
  "document_processor": "ready",
  "firestore_service": "ready",
  "gcs_service": "ready",
  "task_queue": "ready",
  "active_jobs": 0,
  "queue_size": 0,
  "processed_documents": 0
}
```

#### Test 2: Root Endpoint
```bash
curl http://localhost:8080/
```
**Expected Response:**
```json
{
  "service": "Axiom Spark Document Processing API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {...}
}
```

#### Test 3: GCP Connection Test
```bash
curl -X POST http://localhost:8080/api/aws/test
```
**Expected Response:**
```json
{
  "success": true,
  "message": "GCP connection test completed",
  "services": {
    "gcs": {
      "status": "connected",
      "bucket": "voucher-bucket-1",
      "project": "rocasoft"
    },
    "firestore": {
      "status": "connected",
      "project": "rocasoft"
    },
    "anthropic": {
      "status": "configured",
      "model": "claude-sonnet-4-5-20250929"
    }
  }
}
```

#### Test 4: GCP Integration Test
```bash
curl -X POST http://localhost:8080/api/aws/test-integration
```
**Expected Response:**
```json
{
  "success": true,
  "message": "GCP integration test completed",
  "gcs_temp_files": {...},
  "gcs_organized_files": {...},
  "firestore_documents": {...},
  "config": {...}
}
```

#### Test 5: Batch Management
```bash
# List batches
curl http://localhost:8080/api/batches

# Expected:
# {"success": true, "batches": [], "count": 0, ...}
```

#### Test 6: Processing Summary
```bash
curl http://localhost:8080/api/aws/processing/summary
```
**Expected Response:**
```json
{
  "success": true,
  "summary": {
    "total_batches": 0,
    "total_documents": 0,
    "completed_documents": 0,
    "processing_documents": 0,
    "failed_documents": 0,
    "success_rate": 0
  }
}
```

## ✅ File Upload Tests

### Test 7: Single File Upload
```bash
# Create a test image file
echo "Test file content" > test_document.txt

# Upload using multipart form
curl -X POST http://localhost:8080/api/aws/upload \
  -F "file=@test_document.txt" \
  -F "batch_id=test-batch-001"
```
**Expected Response:**
```json
{
  "success": true,
  "document_id": "...",
  "batch_id": "test-batch-001",
  "filename": "test_document.txt",
  "message": "File uploaded and queued for processing"
}
```

### Test 8: Batch File Upload
```bash
curl -X POST http://localhost:8080/api/aws/batch-upload \
  -F "files=@test_document.txt" \
  -F "files=@test_document2.txt" \
  -F "batch_id=test-batch-002"
```
**Expected Response:**
```json
{
  "success": true,
  "batch_id": "test-batch-002",
  "job_id": "...",
  "uploaded": [...],
  "failed": [],
  "message": "Uploaded 2 files, 0 failed"
}
```

## ✅ GCS Storage Tests

### Test 9: List GCS Files
```bash
# List temp files
curl http://localhost:8080/api/aws/s3/list

# List organized files
curl http://localhost:8080/api/aws/s3/organized

# List organized folders
curl http://localhost:8080/api/aws/s3/organized/folders
```

## ✅ Firestore Tests

### Test 10: Create Batch
```bash
curl -X POST http://localhost:8080/api/batches/create \
  -H "Content-Type: application/json" \
  -d '{"batch_name": "Test Batch", "branch_id": "01", "source": "web"}'
```
**Expected Response:**
```json
{
  "success": true,
  "batch_id": "batch-...",
  "batch_name": "Test Batch",
  "created_at": "...",
  "status": "active"
}
```

### Test 11: List Documents
```bash
curl http://localhost:8080/api/batches/{batch_id}/vouchers
```

## ✅ Integration Tests

### Test 12: API Documentation
```bash
# Open in browser
open http://localhost:8080/docs

# Or curl the OpenAPI schema
curl http://localhost:8080/openapi.json
```

### Test 13: WebSocket Connection
```javascript
// In browser console or Node.js
const ws = new WebSocket('ws://localhost:8080/ws');
ws.onopen = () => console.log('✅ WebSocket connected');
ws.onmessage = (event) => console.log('Message:', event.data);
ws.send('ping');
// Expected: 'pong' response
```

## ✅ Error Handling Tests

### Test 14: Invalid Endpoints
```bash
curl http://localhost:8080/api/invalid-endpoint
# Expected: 404 error

curl -X POST http://localhost:8080/api/batches/create
# Expected: 422 validation error (missing required fields)
```

### Test 15: Missing Credentials
```bash
# Temporarily rename service account key file
mv voucher-storage-key.json voucher-storage-key.json.bak

# Restart server - should show warnings but still start
# Check /health endpoint
curl http://localhost:8080/health

# Restore key file
mv voucher-storage-key.json.bak voucher-storage-key.json
```

## ✅ Performance Tests

### Test 16: Concurrent Requests
```bash
# Install apache bench if not available
# brew install httpd (macOS)
# sudo apt-get install apache2-utils (Ubuntu)

# Test with 100 concurrent requests
ab -n 100 -c 10 http://localhost:8080/health

# Expected: No errors, reasonable response times
```

## ✅ Frontend Integration Tests

### Test 17: CORS Headers
```bash
curl -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS \
  http://localhost:8080/api/batches \
  -v
```
**Expected**: CORS headers in response

### Test 18: Frontend API Calls
- [ ] Login/Authentication works
- [ ] Document upload works
- [ ] Batch listing displays correctly
- [ ] Document viewing works
- [ ] Real-time updates via WebSocket work

## ✅ Cleanup

After testing:
```bash
# Remove test files
rm -f test_document.txt test_document2.txt

# Stop server (Ctrl+C)

# Deactivate virtual environment
deactivate
```

## 📊 Test Results Summary

Fill this out after running all tests:

- [ ] All syntax and import tests passed
- [ ] Server starts without errors
- [ ] All health checks pass
- [ ] GCP services connect successfully
- [ ] File uploads work
- [ ] Firestore operations work
- [ ] GCS operations work
- [ ] API documentation accessible
- [ ] Error handling works correctly
- [ ] Frontend integration works

## 🐛 Known Issues

Document any issues found:
1. [Issue description]
2. [Issue description]
...

## ✅ Sign-off

- Tested by: ___________
- Date: ___________
- Environment: [Development/Staging/Production]
- Status: [Pass/Fail/Partial]
- Notes: ___________


