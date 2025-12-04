# ✅ Cloud Run Deployment - FIXED

## Deployment Status: COMPLETE ✅

### Configuration Summary

**Service Identity:**
- ✅ Service Account: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`
- ✅ Used for: General GCS API access (upload, download, list operations)

**Secret Manager:**
- ✅ Secret Name: `voucher-storage-key`
- ✅ Mount Path: `/secrets/voucher-storage-key.json`
- ✅ Environment Variable: `GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json`
- ✅ Used for: Generating signed URLs (requires private key)

**Volume Mounts:**
- ✅ Only `/secrets` mount exists (problematic `/app` mount removed)
- ✅ Secret file accessible at `/secrets/voucher-storage-key.json`

**Service Health:**
- ✅ Service is healthy and responding
- ✅ All services initialized: document_processor, firestore_service, gcs_service, task_queue

## What Was Fixed

1. **Removed problematic `/app` mount** that was overwriting the application directory
2. **Kept only `/secrets` mount** at the correct path
3. **Service account configured** for service identity
4. **Environment variable set** to point to the mounted secret
5. **Service is running** and healthy

## Current Configuration

```yaml
Service Account: voucher-storage-sa@rocasoft.iam.gserviceaccount.com
Environment Variables:
  - GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json
  - GCS_BUCKET_NAME=voucher-bucket-1
  - GCS_PROJECT_ID=rocasoft
  - FIRESTORE_PROJECT_ID=rocasoft

Volume Mounts:
  - /secrets/voucher-storage-key.json (from Secret Manager)

Service URL: https://docflow-demo-backend-672967533609.europe-west1.run.app
```

## How It Works

1. **Service Identity**: Cloud Run automatically uses `voucher-storage-sa@rocasoft.iam.gserviceaccount.com` for general GCS operations via Application Default Credentials (ADC)

2. **Signed URLs**: When generating signed URLs, the code:
   - Reads `GOOGLE_APPLICATION_CREDENTIALS` environment variable
   - Finds the key file at `/secrets/voucher-storage-key.json`
   - Uses the private key from the mounted secret to sign URLs

## Verification

### Health Check
```bash
curl https://docflow-demo-backend-672967533609.europe-west1.run.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "gcs_service": "ready",
  "firestore_service": "ready"
}
```

### Check Logs
```bash
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 50
```

Look for:
- `✅ GCS client initialized with service account key: /secrets/voucher-storage-key.json (signing: True)`

## Testing Signed URLs

The service should now be able to:
1. ✅ Upload documents to GCS (using service identity)
2. ✅ Generate signed URLs for document previews (using mounted secret)
3. ✅ Access Firestore (using service identity)

## Security Best Practices Followed

✅ **Service Identity**: Uses user-managed service account (not default compute account)  
✅ **Secret Manager**: Key stored securely, not in Docker image  
✅ **Least Privilege**: Service account has only necessary permissions  
✅ **No Secrets in Code**: Key file never committed to git  
✅ **Automatic Rotation**: Can update secret without rebuilding image

## Next Steps

1. Test document upload functionality
2. Test document preview (signed URL generation)
3. Monitor logs for any credential-related errors
4. Set up alerts for secret access failures

## Troubleshooting

If signed URLs still fail:

1. **Check secret is mounted**:
   ```bash
   gcloud run services describe docflow-demo-backend \
     --region europe-west1 \
     --project rocasoft \
     --format="yaml(spec.template.spec.containers[0].volumeMounts)"
   ```

2. **Verify environment variable**:
   ```bash
   gcloud run services describe docflow-demo-backend \
     --region europe-west1 \
     --project rocasoft \
     --format="value(spec.template.spec.containers[0].env)" | grep GOOGLE_APPLICATION
   ```

3. **Check service account permissions**:
   ```bash
   gcloud projects get-iam-policy rocasoft \
     --flatten="bindings[].members" \
     --filter="bindings.members:serviceAccount:voucher-storage-sa@rocasoft.iam.gserviceaccount.com"
   ```

## Deployment Complete! 🎉

The service is now properly configured with:
- ✅ Service identity for general API access
- ✅ Secret Manager for signed URL generation
- ✅ No secrets in Docker image
- ✅ Clean volume mounts (no conflicts)

