# GCS CORS Configuration Fix

## Problem
CORS error when uploading files directly to GCS using signed URLs:
```
Access to XMLHttpRequest at 'https://storage.googleapis.com/voucher-bucket-1/...' 
from origin 'https://docflowai-c88e6.web.app' has been blocked by CORS policy: 
Response to preflight request doesn't pass access control check: 
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Root Cause
The GCS bucket `voucher-bucket-1` didn't have CORS rules configured to allow requests from the production frontend domain.

## Solution Applied

### 1. Updated CORS Configuration Files

**Files Updated:**
- `cors.json` - CORS configuration file
- `scripts/configure_gcs_cors.py` - Python script for configuring CORS

**Added Production Domains:**
- `https://docflowai-c88e6.web.app`
- `https://docflowai-c88e6.firebaseapp.com`

### 2. Applied CORS Configuration

```bash
gsutil cors set cors.json gs://voucher-bucket-1
```

### Current CORS Configuration

```json
[
  {
    "origin": [
      "http://localhost:3000",
      "http://localhost:8080",
      "http://localhost:4200",
      "http://localhost",
      "https://localhost",
      "capacitor://localhost",
      "ionic://localhost",
      "https://docflowai-c88e6.web.app",
      "https://docflowai-c88e6.firebaseapp.com"
    ],
    "method": ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
    "responseHeader": [
      "Content-Type",
      "Content-Length",
      "Content-Range",
      "Accept",
      "Authorization",
      "X-Goog-Upload-Protocol",
      "X-Goog-Upload-Command",
      "X-Goog-Upload-Offset",
      "X-Goog-Upload-Status",
      "X-Goog-Upload-Chunk-Granularity",
      "X-Goog-Upload-Header-Content-Length",
      "X-Goog-Upload-Header-Content-Type"
    ],
    "maxAgeSeconds": 3600
  }
]
```

## Verification

Check current CORS configuration:
```bash
gsutil cors get gs://voucher-bucket-1
```

## What This Fixes

✅ Allows direct uploads from frontend to GCS using signed URLs  
✅ Handles preflight OPTIONS requests correctly  
✅ Supports all necessary HTTP methods (GET, POST, PUT, DELETE, HEAD, OPTIONS)  
✅ Includes all required response headers for uploads  
✅ Works for both development (localhost) and production domains  

## Testing

After this fix, file uploads from the frontend should work without CORS errors. The browser will:
1. Send a preflight OPTIONS request (handled by CORS config)
2. Receive proper CORS headers in response
3. Proceed with the actual PUT request to upload the file

## Additional Notes

- CORS configuration is applied at the bucket level
- Changes take effect immediately
- `maxAgeSeconds: 3600` means browsers cache CORS preflight responses for 1 hour
- All necessary headers for GCS uploads are included in `responseHeader`

## Future Updates

If you add new frontend domains, update `cors.json` and run:
```bash
gsutil cors set cors.json gs://voucher-bucket-1
```

Or use the Python script:
```bash
python3 scripts/configure_gcs_cors.py
```

