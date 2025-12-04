# Cloud Run Deployment Status

## ✅ Completed Setup

### 1. Secret Manager
- ✅ Secret created: `voucher-storage-key`
- ✅ Service account has access: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`

### 2. Service Identity
- ✅ Service account assigned: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`
- ✅ Service identity configured for general GCS API access

### 3. Secret Mount
- ✅ Secret mounted at: `/secrets/voucher-storage-key.json`
- ✅ Environment variable set: `GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json`

## ⚠️ Known Issue

There's a legacy secret mount at `/app` that Cloud Run isn't removing. However, this doesn't seem to be causing issues as:
- The service is healthy and responding
- The correct secret is mounted at `/secrets/voucher-storage-key.json`
- The environment variable points to the correct location

## Current Configuration

```yaml
Service Account: voucher-storage-sa@rocasoft.iam.gserviceaccount.com
Environment Variables:
  - GOOGLE_APPLICATION_CREDENTIALS=/secrets/voucher-storage-key.json
Volume Mounts:
  - /secrets/voucher-storage-key.json (from Secret Manager)
```

## Testing

To verify signed URLs are working:

```bash
# Test the service
curl https://docflow-demo-backend-672967533609.europe-west1.run.app/health

# Check logs for credential loading
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 50
```

Look for:
- `✅ GCS client initialized with service account key: /secrets/voucher-storage-key.json (signing: True)`

## Next Steps

1. Test document upload/preview to verify signed URLs work
2. If the `/app` mount causes issues, we may need to redeploy from scratch
3. Monitor logs for any credential-related errors

## Service URL

https://docflow-demo-backend-672967533609.europe-west1.run.app

