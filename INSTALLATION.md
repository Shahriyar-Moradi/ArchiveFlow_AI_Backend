# Installation Guide - GCP Backend

## Prerequisites

- Python 3.8 or higher
- GCP Service Account with permissions for:
  - Cloud Storage
  - Firestore
- Anthropic API key

## Step 1: Install Python Dependencies

### Option A: Using Virtual Environment (Recommended)

```bash
cd rizanai/backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using System Python

```bash
cd rizanai/backend
pip3 install -r requirements.txt
```

## Step 2: Configure Environment Variables

Create a `.env` file in `rizanai/backend/`:

```bash
cp env.template .env
```

Edit `.env` and set the following:

```env
# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Google Cloud Storage
GCS_BUCKET_NAME=voucher-bucket-1
GCS_PROJECT_ID=rocasoft
GCS_SERVICE_ACCOUNT_KEY=voucher-storage-key.json

# Firestore
FIRESTORE_PROJECT_ID=rocasoft

# Server
HOST=0.0.0.0
PORT=8080

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://localhost:4200,capacitor://localhost,ionic://localhost
```

## Step 3: GCP Service Account Setup

### Create Service Account
1. Go to [GCP Console](https://console.cloud.google.com/)
2. Navigate to IAM & Admin > Service Accounts
3. Click "Create Service Account"
4. Name it (e.g., "document-processing-sa")
5. Grant roles:
   - Storage Object Admin
   - Cloud Datastore User

### Download Service Account Key
1. Click on the created service account
2. Go to "Keys" tab
3. Click "Add Key" > "Create new key"
4. Select JSON format
5. Download and save as `voucher-storage-key.json`
6. Place in `rizanai/backend/` directory

## Step 4: Create GCS Bucket

```bash
# Using gcloud CLI
gsutil mb -p rocasoft gs://voucher-bucket-1

# Set bucket permissions (if needed)
gsutil iam ch serviceAccount:YOUR_SERVICE_ACCOUNT@rocasoft.iam.gserviceaccount.com:objectAdmin gs://voucher-bucket-1
```

## Step 5: Initialize Firestore

1. Go to [Firestore Console](https://console.firebase.google.com/)
2. Select your project ("rocasoft")
3. Click "Create database"
4. Choose "Production mode"
5. Select a location
6. Wait for initialization

Collections will be created automatically when first used:
- `documents`
- `processing_jobs`
- `flows`

## Step 6: Verify Installation

```bash
cd rizanai/backend

# Activate venv if using
source venv/bin/activate

# Test configuration
python3 -c "from config import settings; print('✅ Config OK')"

# Test service imports
python3 -c "from services.document_processor import DocumentProcessor; print('✅ Services OK')"

# Test main application
python3 -c "from main import app; print('✅ Main OK')"
```

## Step 7: Start the Server

```bash
# Development mode (with auto-reload)
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Production mode
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

## Step 8: Test the API

```bash
# Health check
curl http://localhost:8080/health

# GCP services test
curl -X POST http://localhost:8080/api/aws/test

# API documentation
open http://localhost:8080/docs
```

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure virtual environment is activated and all packages are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: GCP authentication errors

**Solution**: Verify service account key file exists and path is correct:
```bash
ls -la voucher-storage-key.json
# Should show the file

# Set GOOGLE_APPLICATION_CREDENTIALS (optional)
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/voucher-storage-key.json"
```

### Issue: Anthropic API errors

**Solution**: Verify API key is set:
```bash
grep ANTHROPIC_API_KEY .env
# Should show your key (partially masked)
```

### Issue: Port already in use

**Solution**: Change port in `.env` or kill existing process:
```bash
# Find process using port 8080
lsof -i :8080

# Kill process
kill -9 PID

# Or use different port
python3 -m uvicorn main:app --port 8081
```

### Issue: Permission denied errors

**Solution**: Check service account permissions in GCP Console:
1. IAM & Admin > Service Accounts
2. Verify roles are assigned
3. Check bucket/Firestore permissions

## Production Deployment

### Using systemd (Linux)

Create `/etc/systemd/system/document-backend.service`:

```ini
[Unit]
Description=Document Processing Backend
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/rizanai/backend
Environment="PATH=/path/to/rizanai/backend/venv/bin"
ExecStart=/path/to/rizanai/backend/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable document-backend
sudo systemctl start document-backend
sudo systemctl status document-backend
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:
```bash
docker build -t document-backend .
docker run -p 8080:8080 --env-file .env document-backend
```

## Next Steps

1. Configure frontend to use backend URL
2. Set up monitoring and logging
3. Configure backup strategies for Firestore
4. Set up CI/CD pipeline
5. Review security settings

## Support

For issues or questions, check:
- [GCP Integration Status](./GCP_INTEGRATION_STATUS.md)
- API documentation at `/docs`
- Health endpoint at `/health`

