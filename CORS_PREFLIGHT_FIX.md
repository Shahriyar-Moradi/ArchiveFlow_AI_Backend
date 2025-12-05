# CORS Preflight Request Fix

## Current Error

```
Access to fetch at 'https://docflow-demo-backend-672967533609.europe-west1.run.app/api/auth/profile' 
from origin 'https://docflowai-c88e6.web.app' has been blocked by CORS policy: 
Response to preflight request doesn't pass access control check: 
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

## Root Cause

The error indicates that the **OPTIONS preflight request** is failing. This happens when:
1. The browser sends an OPTIONS request before the actual GET/POST request
2. The OPTIONS request doesn't receive proper CORS headers
3. The browser blocks the actual request

## Why This Is Happening

**The updated code with CORS fixes has not been deployed to Cloud Run yet.** The service is still running the old code that doesn't have:
- The `CORSResponseMiddleware` 
- Enhanced exception handlers
- Explicit OPTIONS handler

## Solution Implemented

### 1. Enhanced OPTIONS Handler
The explicit OPTIONS handler now:
- Properly handles preflight requests for all routes
- Returns correct CORS headers including `Access-Control-Allow-Origin`
- Handles the case when `allow_credentials=True` (cannot use `*` for origin)
- Includes all required CORS headers for preflight

### 2. Response Middleware
The `CORSResponseMiddleware` ensures CORS headers are added to ALL responses, including:
- Successful responses
- Error responses
- OPTIONS preflight responses

### 3. Exception Handlers
Enhanced exception handlers ensure CORS headers are present even when errors occur.

## Next Steps - DEPLOY THE CODE

The fixes are implemented in the code but **must be deployed to Cloud Run**:

```bash
cd ArchiveFlow_AI_Backend

# Build Docker image
docker build -t gcr.io/rocasoft/docflow-backend:latest .

# Push to Google Container Registry
docker push gcr.io/rocasoft/docflow-backend:latest

# Deploy to Cloud Run
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --project rocasoft \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=rocasoft,FIRESTORE_PROJECT_ID=rocasoft
```

Or use the deployment script:
```bash
cd ArchiveFlow_AI_Backend
./deploy_to_cloud_run.sh
```

## Verification After Deployment

1. **Test OPTIONS preflight**:
   ```bash
   curl -X OPTIONS \
     -H "Origin: https://docflowai-c88e6.web.app" \
     -H "Access-Control-Request-Method: GET" \
     -H "Access-Control-Request-Headers: authorization" \
     -v https://docflow-demo-backend-672967533609.europe-west1.run.app/api/auth/profile
   ```
   
   Should return 204 with CORS headers:
   ```
   < HTTP/1.1 204 No Content
   < Access-Control-Allow-Origin: https://docflowai-c88e6.web.app
   < Access-Control-Allow-Credentials: true
   < Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH
   < Access-Control-Allow-Headers: *
   < Access-Control-Max-Age: 3600
   ```

2. **Test actual request**:
   ```bash
   curl -X GET \
     -H "Origin: https://docflowai-c88e6.web.app" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -v https://docflow-demo-backend-672967533609.europe-west1.run.app/api/auth/profile
   ```

3. **Check browser DevTools**:
   - Open Network tab
   - Look for OPTIONS request (should succeed with 204)
   - Look for actual GET request (should succeed with 200)
   - Both should have CORS headers

## Important Notes

1. **When `allow_credentials=True`**: You cannot use `Access-Control-Allow-Origin: *`. The middleware now always uses the specific origin from the request.

2. **Preflight Caching**: The `Access-Control-Max-Age: 3600` header tells browsers to cache the preflight response for 1 hour, reducing the number of OPTIONS requests.

3. **Multiple Layers of CORS Protection**:
   - FastAPI CORSMiddleware (handles standard CORS)
   - CORSResponseMiddleware (ensures headers on all responses)
   - Explicit OPTIONS handler (handles preflight explicitly)
   - Exception handlers (ensure headers on errors)

## If Issues Persist After Deployment

1. **Check Cloud Run logs**:
   ```bash
   gcloud run services logs read docflow-demo-backend \
     --region europe-west1 \
     --project rocasoft \
     --limit 100
   ```

2. **Verify service is running**:
   ```bash
   curl https://docflow-demo-backend-672967533609.europe-west1.run.app/health
   ```

3. **Check CORS configuration**:
   - Verify `allowed_origins` includes `https://docflowai-c88e6.web.app`
   - Check that `allow_credentials=True` is set
   - Verify all CORS headers are being added

## Summary

✅ **Code fixes are complete** - All CORS issues have been addressed in the code
⏳ **Deployment required** - The code must be deployed to Cloud Run for fixes to take effect
🔍 **Verify after deployment** - Test OPTIONS and actual requests to confirm CORS is working

