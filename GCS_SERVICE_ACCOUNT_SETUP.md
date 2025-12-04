# GCS Service Account Credentials Setup Guide

## Overview

This document explains how to set up Google Cloud Storage (GCS) service account credentials for generating signed URLs in the ArchiveFlow AI application. Signed URLs are required for secure access to private GCS objects (images, PDFs) without making them publicly accessible.

## Problem

Without proper service account credentials, the application will fail with errors like:
```
Failed to generate signed URL: Service account credentials with private key are required
Error: Service account credentials with private key required for signed URLs
```

## Solution

Download and configure a service account JSON key file that contains the private key needed for signing URLs.

---

## Step-by-Step Setup

### Step 1: Access Google Cloud Console

1. Go to the Google Cloud Console:
   - Direct link: https://console.cloud.google.com/iam-admin/serviceaccounts?project=rocasoft
   - Or navigate: **IAM & Admin** → **Service Accounts**

2. Make sure you're in the correct project: **rocasoft**

### Step 2: Locate the Service Account

1. Find the service account:
   - **Email**: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`
   - **Name**: "Voucher Storage Service Account"
   - **Description**: "Service account for uploading voucher documents to GCS"

2. Click on the service account email/name to open its details

### Step 3: Create a New Key

1. Click on the **"Keys"** tab at the top of the service account page

2. Click **"Add Key"** → **"Create new key"**

3. Select **"JSON"** as the key type

4. Click **"Create"**

5. The JSON key file will automatically download to your computer

   **Note**: The file will be named something like `rocasoft-7e5fad67f504.json` (the name includes the project ID and key ID)

### Step 4: Place the Key File

1. **Rename the downloaded file** to: `voucher-storage-key.json`

2. **Move it to the backend directory**:
   ```
   /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend/voucher-storage-key.json
   ```

3. **Set secure file permissions** (important for security):
   ```bash
   cd /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend
   chmod 600 voucher-storage-key.json
   ```

### Step 5: Verify the Key File

Verify the file is valid JSON and contains the required fields:

```bash
cd /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend
python3 -c "import json; f=open('voucher-storage-key.json'); data=json.load(f); print('✅ Key file is valid'); print(f'   Service Account: {data.get(\"client_email\")}'); print(f'   Project: {data.get(\"project_id\")}'); print(f'   Has private_key: {\"private_key\" in data and bool(data.get(\"private_key\"))}')"
```

Expected output:
```
✅ Key file is valid
   Service Account: voucher-storage-sa@rocasoft.iam.gserviceaccount.com
   Project: rocasoft
   Has private_key: True
```

### Step 6: Restart the Backend Server

**IMPORTANT**: The backend must be restarted to load the new credentials.

1. Stop the current backend process:
   - Press `Ctrl+C` in the terminal where the backend is running
   - Or kill the process if running in the background

2. Start the backend again:
   ```bash
   cd /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend
   python3 main.py
   ```

### Step 7: Verify It's Working

After restarting, check the backend logs. You should see:

```
✅ GCS client initialized with service account key: /path/to/voucher-storage-key.json (signing: True)
```

If you see this message, the credentials are loaded correctly and signed URL generation will work.

---

## Alternative: Using Environment Variable

Instead of placing the file in the backend directory, you can set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/voucher-storage-key.json"
```

Then start the backend. The system will use the file specified in the environment variable.

---

## Key File Structure

The service account key file (`voucher-storage-key.json`) should contain:

```json
{
  "type": "service_account",
  "project_id": "rocasoft",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "voucher-storage-sa@rocasoft.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "...",
  "universe_domain": "googleapis.com"
}
```

**Critical fields**:
- `private_key`: Required for signing URLs (must be present and non-empty)
- `client_email`: Service account email address
- `project_id`: GCP project ID

---

## Troubleshooting

### Issue: "Key file found: Yes" but "Credentials support signing: No"

**Solution**:
1. Verify the file is valid JSON:
   ```bash
   python3 -c "import json; json.load(open('voucher-storage-key.json'))"
   ```

2. Check that the file contains a `private_key` field:
   ```bash
   python3 -c "import json; data=json.load(open('voucher-storage-key.json')); print('Has private_key:', 'private_key' in data and bool(data.get('private_key')))"
   ```

3. **Restart the backend server** - this is the most common cause

### Issue: "Key file not found"

**Solution**:
1. Verify the file exists:
   ```bash
   ls -la /Users/shahriar/Desktop/Desktop/Work/Document_Archive/MyDev/ArchiveFlow_AI/backend/voucher-storage-key.json
   ```

2. Check the file name is exactly `voucher-storage-key.json` (not `rocasoft-*.json`)

3. Verify you're in the correct directory

### Issue: "Service account credentials with private key are required"

**Solution**:
1. Ensure the key file was downloaded from GCP Console (not created manually)
2. Verify the `private_key` field exists and is not empty
3. Check file permissions: `chmod 600 voucher-storage-key.json`
4. Restart the backend server

### Issue: File permissions error

**Solution**:
```bash
chmod 600 voucher-storage-key.json
ls -la voucher-storage-key.json
# Should show: -rw------- (owner read/write only)
```

---

## Security Best Practices

1. **File Permissions**: Always set the key file to `600` (owner read/write only):
   ```bash
   chmod 600 voucher-storage-key.json
   ```

2. **Never Commit to Git**: The key file should be in `.gitignore`:
   ```
   backend/voucher-storage-key.json
   backend/*.json
   ```

3. **Rotate Keys Regularly**: Create new keys periodically and delete old ones from GCP Console

4. **Use Environment Variables in Production**: For production deployments, use `GOOGLE_APPLICATION_CREDENTIALS` environment variable instead of placing the file in the codebase

---

## Code Implementation Details

### How It Works

The application uses the `s3_service.py` module which:

1. **Checks for key file** in this order:
   - `GOOGLE_APPLICATION_CREDENTIALS` environment variable
   - `backend/voucher-storage-key.json` (local file)
   - Application-default credentials (fallback, doesn't support signing)

2. **Loads credentials** using:
   ```python
   storage.Client.from_service_account_json(key_path, project=project_id)
   ```

3. **Validates signing support** by checking for `private_key` in the credentials

4. **Generates signed URLs** using:
   ```python
   blob.generate_signed_url(version="v4", expiration=timedelta(seconds=expiration), method="GET")
   ```

### Key Files Modified

- `backend/s3_service.py`: Contains credential loading and validation logic
- `backend/main.py`: Uses `s3_service` for generating presigned URLs

---

## Quick Reference

### Service Account Details
- **Email**: `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`
- **Project**: `rocasoft`
- **Key File Location**: `backend/voucher-storage-key.json`
- **Required Permissions**: Storage Object Admin (for reading/writing GCS objects)

### Commands

**Verify key file**:
```bash
python3 -c "import json; data=json.load(open('voucher-storage-key.json')); print('Valid:', 'private_key' in data)"
```

**Set permissions**:
```bash
chmod 600 voucher-storage-key.json
```

**Check if file exists**:
```bash
ls -la backend/voucher-storage-key.json
```

**Set environment variable** (alternative):
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/full/path/to/voucher-storage-key.json"
```

---

## Related Documentation

- GCP Service Accounts: https://cloud.google.com/iam/docs/service-accounts
- GCS Signed URLs: https://cloud.google.com/storage/docs/access-control/signing-urls-with-helpers
- Application Credentials: https://cloud.google.com/docs/authentication/application-default-credentials

---

## Support

If you continue to experience issues after following this guide:

1. Check backend logs for detailed error messages
2. Verify the key file is valid JSON with all required fields
3. Ensure the backend server has been restarted after placing the key file
4. Verify file permissions are set correctly (`600`)
5. Check that the service account has the necessary GCS permissions in GCP Console

---

**Last Updated**: November 30, 2025  
**Service Account**: voucher-storage-sa@rocasoft.iam.gserviceaccount.com  
**Project**: rocasoft

