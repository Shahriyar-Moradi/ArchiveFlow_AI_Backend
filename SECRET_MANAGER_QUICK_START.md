# Secret Manager Quick Start Guide

## Overview

This guide shows you how to use Google Cloud Secret Manager to securely store and access your service account key in Cloud Run, without embedding it in your Docker image.

## Benefits

✅ **Security**: Key never stored in Docker image  
✅ **Rotation**: Update keys without rebuilding images  
✅ **Access Control**: Fine-grained IAM permissions  
✅ **Audit**: Full logging of secret access  
✅ **Best Practice**: Recommended by Google Cloud

---

## Step-by-Step Setup

### 1. Create Secret in Secret Manager

```bash
cd ArchiveFlow_AI_Backend

# Create the secret
gcloud secrets create voucher-storage-key \
  --project=rocasoft \
  --data-file=voucher-storage-key.json \
  --replication-policy="automatic"
```

### 2. Grant Cloud Run Access

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe rocasoft --format="value(projectNumber)")

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding voucher-storage-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=rocasoft
```

### 3. Build Docker Image (without key file)

The key file is **not** included in the image:

```bash
docker build -t gcr.io/rocasoft/docflow-backend:latest .
docker push gcr.io/rocasoft/docflow-backend:latest
```

### 4. Deploy Cloud Run with Secret Mount

```bash
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --project rocasoft \
  --allow-unauthenticated \
  --set-secrets=/app/voucher-storage-key.json=voucher-storage-key:latest \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json,GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=rocasoft,FIRESTORE_PROJECT_ID=rocasoft
```

**Key parameters:**
- `--set-secrets=/app/voucher-storage-key.json=voucher-storage-key:latest`
  - Mounts the secret as a file at `/app/voucher-storage-key.json`
  - Uses the `latest` version of the secret
  
- `GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json`
  - Points to the mounted secret file
  - Your application code will automatically use this path

---

## Automated Setup

Use the provided script for automated setup:

```bash
./setup_secret_manager.sh
```

This script will:
1. ✅ Check if key file exists
2. ✅ Create or update the secret
3. ✅ Grant Cloud Run access
4. ✅ Optionally deploy/update the service

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Google Cloud Secret Manager                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ voucher-storage-key (secret)                      │  │
│  │ └─ Contains: voucher-storage-key.json content    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Mounted at runtime
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Cloud Run Container                                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ /app/voucher-storage-key.json (mounted file)     │  │
│  │ └─ Automatically available at runtime            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  Environment Variable:                                   │
│  GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage... │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Your code reads this path
                        ▼
┌─────────────────────────────────────────────────────────┐
│  s3_service.py                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │ key_path = os.getenv("GOOGLE_APPLICATION_...")     │  │
│  │ # Returns: /app/voucher-storage-key.json          │  │
│  │ # File exists and is readable ✅                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Updating the Secret

To update the service account key:

```bash
# Add new version of the secret
gcloud secrets versions add voucher-storage-key \
  --data-file=voucher-storage-key.json \
  --project=rocasoft

# Cloud Run will use the latest version automatically
# No need to redeploy (unless you want to force a restart)
```

---

## Verification

### Check Secret Exists

```bash
gcloud secrets describe voucher-storage-key --project=rocasoft
```

### Verify Cloud Run Configuration

```bash
gcloud run services describe docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --format="yaml(spec.template.spec.containers[0].env,spec.template.spec.containers[0].volumeMounts)"
```

### Check Logs

```bash
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 50
```

Look for:
- `✅ GCS client initialized with service account key: /app/voucher-storage-key.json (signing: True)`

---

## Troubleshooting

### Error: Secret not found

```bash
# Verify secret exists
gcloud secrets list --project=rocasoft | grep voucher-storage-key

# If missing, create it
gcloud secrets create voucher-storage-key \
  --data-file=voucher-storage-key.json \
  --project=rocasoft
```

### Error: Permission denied

```bash
# Re-grant access
PROJECT_NUMBER=$(gcloud projects describe rocasoft --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding voucher-storage-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=rocasoft
```

### Error: File not found at /app/voucher-storage-key.json

- Verify `--set-secrets` parameter is correct
- Check the mount path matches `GOOGLE_APPLICATION_CREDENTIALS`
- Ensure secret name is correct: `voucher-storage-key`

---

## Security Best Practices

1. **Rotate keys regularly** - Update secrets every 90 days
2. **Use least privilege** - Only grant `secretAccessor` role
3. **Monitor access** - Enable Cloud Audit Logs
4. **Never commit keys** - Already in `.gitignore`
5. **Use separate secrets** - One per environment (dev/staging/prod)

---

## Cost

Secret Manager pricing:
- **Storage**: $0.06 per secret per month
- **Access**: $0.03 per 10,000 operations
- **Estimated cost**: ~$0.10/month for this use case

Very affordable for the security benefits!

