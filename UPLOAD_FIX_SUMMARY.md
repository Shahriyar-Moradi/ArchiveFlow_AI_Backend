# Upload Fix Summary

## Problem
Backend upload requests failing with errors related to signed URL generation.

## Root Cause
The `s3_service` is initialized at module import time (when `main.py` imports it), which may happen before the Secret Manager secret is fully mounted and accessible in the Cloud Run container.

## Solution Implemented

### Code Changes in `s3_service.py`

**Location:** `generate_presigned_url()` method (line ~669)

**What was added:**
1. **Dynamic credential re-checking**: When generating signed URLs, if credentials don't support signing, the code now:
   - Checks if `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
   - Verifies the file exists and is readable
   - Validates the file contains a `private_key` field
   - Reinitializes the GCS client if the key file is now available
   - Provides detailed error messages if still not accessible

2. **Better error handling**: Added comprehensive logging and error messages to help diagnose issues

## Deployment Required

**⚠️ IMPORTANT:** The code changes need to be deployed to Cloud Run for the fix to take effect.

### Option 1: Automatic Deployment (if using Cloud Build)
```bash
# Push changes to trigger Cloud Build
git push origin main
```

### Option 2: Manual Deployment
```bash
# Build and push Docker image
cd ArchiveFlow_AI_Backend
docker build -t gcr.io/rocasoft/docflow-backend:latest .
docker push gcr.io/rocasoft/docflow-backend:latest

# Deploy to Cloud Run
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --region europe-west1 \
  --project rocasoft \
  --service-account=voucher-storage-sa@rocasoft.iam.gserviceaccount.com \
  --set-secrets=/secrets/voucher-storage-key.json=voucher-storage-key:latest \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json
```

## Verification After Deployment

### 1. Check Logs for Reinitialization
```bash
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 50
```

Look for:
- `🔄 Key file now available at /secrets/voucher-storage-key.json, reinitializing GCS client...`
- `✅ Successfully reinitialized GCS client with signing support: True`

### 2. Test Upload
Try uploading a document through the frontend or API. The upload should now work.

## Current Configuration Status

✅ **Service Account**: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`  
✅ **Secret Mount**: `/secrets/voucher-storage-key.json`  
✅ **Environment Variable**: `GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json`  
✅ **Code Fix**: Implemented (needs deployment)

## Next Steps

1. **Deploy the updated code** (see options above)
2. **Monitor logs** for reinitialization messages
3. **Test upload** functionality
4. **Verify** signed URLs are generated correctly

The fix ensures that even if credentials aren't available at startup, they'll be detected and used when needed for signed URL generation.

