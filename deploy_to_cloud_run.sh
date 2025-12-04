#!/bin/bash
# Quick deployment script for Cloud Run with service account key

set -e

PROJECT_ID="rocasoft"
SERVICE_NAME="docflow-demo-backend"
REGION="europe-west1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying ${SERVICE_NAME} to Cloud Run..."
echo ""

# Check if key file exists
if [ ! -f "voucher-storage-key.json" ]; then
    echo "❌ Error: voucher-storage-key.json not found in current directory"
    echo "   Please ensure the service account key file is present"
    exit 1
fi

echo "✅ Key file found: voucher-storage-key.json"
echo ""

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"
echo ""

# Push to GCR
echo "📤 Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    exit 1
fi

echo "✅ Image pushed successfully"
echo ""

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=${PROJECT_ID},FIRESTORE_PROJECT_ID=${PROJECT_ID}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "📋 Service URL:"
    gcloud run services describe ${SERVICE_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --format 'value(status.url)'
    echo ""
    echo "🔍 To check logs:"
    echo "   gcloud run services logs read ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID}"
else
    echo "❌ Deployment failed"
    exit 1
fi

