# Cloud Run Deployment Guide - Service Account Key Configuration

## Problem
The Cloud Run service is failing to generate signed URLs because it cannot find the service account key file at `/app/voucher-storage-key.json`.

## Quick Start (Recommended: Secret Manager)

**For the most secure deployment, use Google Cloud Secret Manager:**

```bash
cd ArchiveFlow_AI_Backend

# 1. Setup Secret Manager (one-time)
./setup_secret_manager.sh

# 2. Build and push Docker image (key file NOT needed in image)
docker build -t gcr.io/rocasoft/docflow-backend:latest .
docker push gcr.io/rocasoft/docflow-backend:latest

# 3. Deploy with secret mount
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --project rocasoft \
  --allow-unauthenticated \
  --set-secrets=/app/voucher-storage-key.json=voucher-storage-key:latest \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json,GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=rocasoft,FIRESTORE_PROJECT_ID=rocasoft
```

**How it works:**
- Secret Manager stores the key securely
- Cloud Run mounts it as a file at `/app/voucher-storage-key.json`
- `GOOGLE_APPLICATION_CREDENTIALS` environment variable points to the mounted file
- The key is **never** in your Docker image

---

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

### Option 2: Use Google Cloud Secret Manager + Service Identity (Recommended - More Secure) ⭐

This is the **recommended approach** for production deployments. It uses Cloud Run's service identity for most operations and Secret Manager for the private key needed for signed URLs.

#### Important: Service Identity vs. GOOGLE_APPLICATION_CREDENTIALS

**For most Google Cloud APIs**: Use Cloud Run's service identity (service account assigned to the service).  
**For signed URLs**: We still need the private key file because signing requires the private key.

According to Google Cloud best practices:
- ✅ **Do**: Assign a user-managed service account as the service identity
- ⚠️ **Avoid**: Setting `GOOGLE_APPLICATION_CREDENTIALS` for general API access (use service identity instead)
- ✅ **Exception**: For signed URLs, we mount the key file from Secret Manager and use `GOOGLE_APPLICATION_CREDENTIALS` specifically for signing operations

#### Quick Setup (Automated)

Use the provided setup script:

```bash
cd ArchiveFlow_AI_Backend
./setup_secret_manager.sh
```

This script will:
1. Create the secret in Secret Manager
2. Grant Cloud Run service account access
3. Optionally deploy/update your Cloud Run service

#### Manual Setup

**Step 1: Create or Use a Service Account for Service Identity**

```bash
# Option A: Use existing service account (recommended)
SERVICE_ACCOUNT="voucher-storage-sa@rocasoft.iam.gserviceaccount.com"

# Option B: Create a new service account
gcloud iam service-accounts create cloud-run-sa \
  --display-name="Cloud Run Service Account" \
  --project=rocasoft

SERVICE_ACCOUNT="cloud-run-sa@rocasoft.iam.gserviceaccount.com"

# Grant necessary permissions
gcloud projects add-iam-policy-binding rocasoft \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin" \
  --condition=None
```

**Step 2: Create Secret in Secret Manager**

```bash
# Create the secret
gcloud secrets create voucher-storage-key \
  --project=rocasoft \
  --data-file=voucher-storage-key.json \
  --replication-policy="automatic"

# Grant the service account access to the secret
gcloud secrets add-iam-policy-binding voucher-storage-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor" \
  --project=rocasoft
```

**Step 3: Deploy with Service Identity and Secret Mount**

When deploying, assign the service account as the service identity AND mount the secret for signed URLs:

```bash
gcloud run deploy docflow-demo-backend \
  --image gcr.io/rocasoft/docflow-backend:latest \
  --platform managed \
  --region europe-west1 \
  --project rocasoft \
  --allow-unauthenticated \
  --service-account=${SERVICE_ACCOUNT} \
  --set-secrets=/app/voucher-storage-key.json=voucher-storage-key:latest \
  --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json,GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=rocasoft,FIRESTORE_PROJECT_ID=rocasoft
```

**Key Points:**
- `--service-account` assigns the service account as the service identity (for general API access)
- `--set-secrets` mounts the secret as a file at `/app/voucher-storage-key.json` (for signed URLs)
- `GOOGLE_APPLICATION_CREDENTIALS` points to the mounted file path (used only for signing)
- The service identity handles most GCS operations automatically
- The mounted key file is used specifically for generating signed URLs

**Step 4: Update Secret (if needed)**

To update the secret with a new key:

```bash
gcloud secrets versions add voucher-storage-key \
  --data-file=voucher-storage-key.json \
  --project=rocasoft
```

The Cloud Run service will automatically use the latest version on next deployment or restart.

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
- ✅ Dockerfile updated: Works with or without key file (supports both methods)
- ✅ Secret Manager setup script: `setup_secret_manager.sh` available
- ⚠️  Need to rebuild and redeploy Docker image to Cloud Run

## Recommended Approach

**Use Secret Manager (Option 2)** for production deployments:
- ✅ More secure (key not in Docker image)
- ✅ Easier to rotate keys
- ✅ Better access control
- ✅ Audit logging
- ✅ No need to rebuild image when updating keys

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

