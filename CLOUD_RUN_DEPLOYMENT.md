# Cloud Run Deployment Guide - Service Account Key Configuration

## Problem
The Cloud Run service is failing to generate signed URLs because it cannot find the service account key file at `/app/voucher-storage-key.json`.

## Solution Options

### Option 1: Include Key File in Docker Image (Quick Fix)

**Steps:**

1. **Ensure the key file is in the backend directory:**
   ```bash
   cd ArchiveFlow_AI_Backend
   ls -la voucher-storage-key.json
   # Should show the file exists
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t gcr.io/rocasoft/docflow-backend:latest .
   ```

3. **Push to Google Container Registry:**
   ```bash
   docker push gcr.io/rocasoft/docflow-backend:latest
   ```

4. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy docflow-demo-backend \
     --image gcr.io/rocasoft/docflow-backend:latest \
     --platform managed \
     --region europe-west1 \
     --allow-unauthenticated \
     --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1 \
     --set-env-vars GCS_PROJECT_ID=rocasoft \
     --set-env-vars FIRESTORE_PROJECT_ID=rocasoft
   ```

**Note:** This method includes the key in the Docker image. While it works, it's less secure than using Secret Manager.

---

### Option 2: Use Google Cloud Secret Manager (Recommended - More Secure)

This is the recommended approach for production deployments.

#### Step 1: Create Secret in Secret Manager

```bash
# Create the secret
gcloud secrets create voucher-storage-key \
  --project=rocasoft \
  --data-file=voucher-storage-key.json

# Grant Cloud Run service account access to the secret
gcloud secrets add-iam-policy-binding voucher-storage-key \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=rocasoft
```

#### Step 2: Update Dockerfile to Remove Key File Copy

The Dockerfile should NOT copy the key file when using Secret Manager.

#### Step 3: Deploy with Secret Mount

```bash
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-secrets=/app/voucher-storage-key.json=voucher-storage-key:latest \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json \
  --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1 \
  --set-env-vars GCS_PROJECT_ID=rocasoft \
  --set-env-vars FIRESTORE_PROJECT_ID=rocasoft
```

---

### Option 3: Use Environment Variable with Base64 (Alternative)

If you prefer not to use Secret Manager, you can encode the key as base64 and set it as an environment variable:

```bash
# Encode the key file
cat voucher-storage-key.json | base64 > key-base64.txt

# Deploy with the encoded key
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars GCS_SERVICE_ACCOUNT_KEY_JSON="$(cat key-base64.txt | base64 -d)" \
  --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1 \
  --set-env-vars GCS_PROJECT_ID=rocasoft
```

**Note:** This requires updating the code to support `GCS_SERVICE_ACCOUNT_KEY_JSON` environment variable (which was removed in recent changes).

---

## Verification

After deployment, verify the service is working:

```bash
# Check service status
gcloud run services describe docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft

# Test the health endpoint
curl https://docflow-demo-backend-672967533609.europe-west1.run.app/health

# Check logs for credential loading
gcloud run services logs read docflow-demo-backend \
  --region europe-west1 \
  --project rocasoft \
  --limit 50
```

Look for log messages like:
- `✅ GCS client initialized with service account key: /app/voucher-storage-key.json (signing: True)`

---

## Current Status

- ✅ Key file exists: `voucher-storage-key.json` in `ArchiveFlow_AI_Backend/`
- ✅ Key file is valid: Contains private_key and correct service account email
- ✅ Dockerfile updated: Will copy key file if present in build context
- ⚠️  Need to rebuild and redeploy Docker image to Cloud Run

---

## Quick Deploy Command (Option 1 - Quick Fix)

```bash
cd ArchiveFlow_AI_Backend

# Build and push
docker build -t gcr.io/rocasoft/docflow-backend:latest .
docker push gcr.io/rocasoft/docflow-backend:latest

# Deploy
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --project rocasoft \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=rocasoft,FIRESTORE_PROJECT_ID=rocasoft
```

---

## Security Notes

1. **Never commit key files to Git** - They're already in `.gitignore`
2. **Rotate keys regularly** - Create new service account keys annually
3. **Use Secret Manager for production** - More secure than embedding in Docker images
4. **Limit key permissions** - Only grant necessary GCS permissions to the service account

