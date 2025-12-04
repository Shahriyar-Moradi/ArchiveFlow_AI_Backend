# Troubleshooting Upload Issues

## Problem
Upload requests are failing with errors related to signed URL generation.

## Root Cause
The `s3_service` is initialized at module import time, which may happen before the Secret Manager secret is fully mounted and accessible in the Cloud Run container.

## Solution Applied

### 1. Dynamic Credential Re-checking
Updated `s3_service.py` to re-check for credentials when generating signed URLs, not just at initialization time.

**What changed:**
- When `generate_presigned_url()` is called and credentials don't support signing, the code now:
  1. Checks if `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
  2. Verifies the file exists and is readable
  3. Validates the file contains a `private_key` field
  4. Reinitializes the GCS client if the key file is now available
  5. Provides detailed error messages if the key file is still not accessible

### 2. Better Error Logging
Added comprehensive error logging to help diagnose issues:
- Logs when key file becomes available
- Logs when reinitialization succeeds or fails
- Provides detailed error messages with file paths and status

## Verification Steps

### 1. Check Service Logs
```bash
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 100
```

Look for:
- `🔄 Key file now available at /secrets/voucher-storage-key.json, reinitializing GCS client...`
- `✅ Successfully reinitialized GCS client with signing support: True`
- Any error messages about file access

### 2. Test Upload Endpoint
```bash
# Test the create document endpoint (this generates signed URLs)
curl -X POST https://docflow-demo-backend-672967533609.europe-west1.run.app/api/documents/create \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "filename": "test.pdf",
    "file_size": 1024,
    "content_type": "application/pdf",
    "flow_id": "test-flow"
  }'
```

### 3. Verify Secret Mount
```bash
gcloud run services describe docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --format="yaml(spec.template.spec.containers[0].volumeMounts,spec.template.spec.containers[0].env)" | grep -A 3 "mountPath\|GOOGLE_APPLICATION"
```

Expected:
- `mountPath: /secrets`
- `GOOGLE_APPLICATION_CREDENTIALS: /secrets/voucher-storage-key.json`

## Common Issues and Fixes

### Issue 1: Secret Not Mounted
**Symptoms:** Error "Key file found: No"

**Fix:**
```bash
gcloud run services update docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --update-secrets=/secrets/voucher-storage-key.json=voucher-storage-key:latest
```

### Issue 2: Environment Variable Not Set
**Symptoms:** Error "GOOGLE_APPLICATION_CREDENTIALS: Not set"

**Fix:**
```bash
gcloud run services update docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --update-env-vars GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json
```

### Issue 3: File Exists But No Private Key
**Symptoms:** Error "Key file found: Yes" but "Credentials support signing: No"

**Fix:**
1. Verify secret contains private_key:
   ```bash
   gcloud secrets versions access latest \
     --secret=voucher-storage-key \
     --project=rocasoft | python3 -c "import sys, json; d=json.load(sys.stdin); print('Has private_key:', 'private_key' in d and bool(d.get('private_key')))"
   ```

2. If missing, update the secret:
   ```bash
   gcloud secrets versions add voucher-storage-key \
     --data-file=voucher-storage-key.json \
     --project=rocasoft
   ```

### Issue 4: Permission Denied
**Symptoms:** Error reading the key file

**Fix:**
The secret should be automatically readable. If not, check service account permissions:
```bash
gcloud secrets get-iam-policy voucher-storage-key \
  --project=rocasoft
```

Ensure `voucher-storage-sa@rocasoft.iam.gserviceaccount.com` has `roles/secretmanager.secretAccessor`.

## Next Steps After Fix

1. **Deploy the updated code** to Cloud Run
2. **Monitor logs** for the reinitialization messages
3. **Test upload** functionality
4. **Verify signed URLs** are generated correctly

## Expected Behavior After Fix

When an upload request is made:
1. Code checks if credentials support signing
2. If not, checks if key file is now available at `/secrets/voucher-storage-key.json`
3. If available, reinitializes GCS client with signing support
4. Generates signed URL successfully
5. Returns signed URL to client for direct GCS upload

## Monitoring

Watch for these log messages:
- ✅ `✅ Successfully reinitialized GCS client with signing support: True` - Fix working!
- ⚠️ `⚠️ Key file now available but reinitialization failed` - Check error details
- ⚠️ `⚠️ GOOGLE_APPLICATION_CREDENTIALS points to X but file does not exist` - Secret mount issue

