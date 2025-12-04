#!/bin/bash
# Setup Google Cloud Secret Manager for service account key
# This is the recommended secure method for Cloud Run deployment

set -e

PROJECT_ID="rocasoft"
SECRET_NAME="voucher-storage-key"
KEY_FILE="voucher-storage-key.json"
SERVICE_NAME="docflow-demo-backend"
REGION="europe-west1"
# Service account to use as service identity (for general API access)
# Use the same service account that the key file belongs to
SERVICE_ACCOUNT="voucher-storage-sa@rocasoft.iam.gserviceaccount.com"

echo "🔐 Setting up Google Cloud Secret Manager for service account key..."
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "❌ Error: $KEY_FILE not found in current directory"
    echo "   Please ensure the service account key file is present"
    exit 1
fi

echo "✅ Key file found: $KEY_FILE"
echo ""

# Check if secret already exists
if gcloud secrets describe $SECRET_NAME --project=$PROJECT_ID &>/dev/null; then
    echo "⚠️  Secret '$SECRET_NAME' already exists"
    read -p "Do you want to update it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📝 Updating existing secret..."
        gcloud secrets versions add $SECRET_NAME \
            --data-file=$KEY_FILE \
            --project=$PROJECT_ID
        echo "✅ Secret updated successfully"
    else
        echo "ℹ️  Using existing secret"
    fi
else
    # Create new secret
    echo "📝 Creating new secret: $SECRET_NAME"
    gcloud secrets create $SECRET_NAME \
        --data-file=$KEY_FILE \
        --project=$PROJECT_ID \
        --replication-policy="automatic"
    
    echo "✅ Secret created successfully"
fi

echo ""

# Extract service account from key file
echo "🔍 Extracting service account from key file..."
KEY_SERVICE_ACCOUNT=$(python3 -c "import json; print(json.load(open('$KEY_FILE'))['client_email'])" 2>/dev/null || echo "")

if [ -n "$KEY_SERVICE_ACCOUNT" ]; then
    SERVICE_ACCOUNT="$KEY_SERVICE_ACCOUNT"
    echo "   Using service account from key file: $SERVICE_ACCOUNT"
else
    echo "   ⚠️  Could not extract service account from key file"
    echo "   Using configured service account: $SERVICE_ACCOUNT"
fi

# Verify service account exists
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT --project=$PROJECT_ID &>/dev/null; then
    echo "❌ Error: Service account $SERVICE_ACCOUNT not found"
    echo "   Please create the service account first or update SERVICE_ACCOUNT in the script"
    exit 1
fi

# Grant Secret Manager access to the service account
echo ""
echo "🔑 Granting Secret Manager access to service account: $SERVICE_ACCOUNT"
gcloud secrets add-iam-policy-binding $SECRET_NAME \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID

echo "✅ Access granted"
echo ""

# Deploy or update Cloud Run service with secret
echo "🚀 Deploying Cloud Run service with Secret Manager..."
echo ""

read -p "Do you want to deploy/update the Cloud Run service now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if service exists
    if gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
        echo "📝 Updating existing Cloud Run service..."
        gcloud run services update $SERVICE_NAME \
            --region=$REGION \
            --project=$PROJECT_ID \
            --service-account=$SERVICE_ACCOUNT \
            --update-secrets=/app/voucher-storage-key.json=$SECRET_NAME:latest \
            --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json
    else
        echo "📝 Creating new Cloud Run service..."
        echo "   Note: You'll need to build and push the Docker image first"
        echo "   Run: docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME:latest ."
        echo "   Then: docker push gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"
        echo ""
        read -p "Press Enter after pushing the image, or Ctrl+C to cancel..."
        
        gcloud run deploy $SERVICE_NAME \
            --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
            --platform managed \
            --region=$REGION \
            --project=$PROJECT_ID \
            --allow-unauthenticated \
            --service-account=$SERVICE_ACCOUNT \
            --set-secrets=/app/voucher-storage-key.json=$SECRET_NAME:latest \
            --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json,GCS_BUCKET_NAME=voucher-bucket-1,GCS_PROJECT_ID=$PROJECT_ID,FIRESTORE_PROJECT_ID=$PROJECT_ID
    fi
    
    echo ""
    echo "✅ Cloud Run service updated with Secret Manager"
else
    echo ""
    echo "ℹ️  To deploy later, use:"
    echo ""
    echo "   gcloud run deploy $SERVICE_NAME \\"
    echo "     --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \\"
    echo "     --platform managed \\"
    echo "     --region=$REGION \\"
    echo "     --project=$PROJECT_ID \\"
    echo "     --allow-unauthenticated \\"
    echo "     --service-account=$SERVICE_ACCOUNT \\"
    echo "     --set-secrets=/app/voucher-storage-key.json=$SECRET_NAME:latest \\"
    echo "     --set-env-vars GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Summary:"
echo "   - Service account (service identity): $SERVICE_ACCOUNT"
echo "   - Secret created: $SECRET_NAME"
echo "   - Mount path: /app/voucher-storage-key.json"
echo "   - Environment variable: GOOGLE_APPLICATION_CREDENTIALS=/app/voucher-storage-key.json"
echo ""
echo "📝 How it works:"
echo "   - Service identity ($SERVICE_ACCOUNT) handles general GCS API access"
echo "   - Mounted key file is used specifically for generating signed URLs"
echo ""
echo "🔍 To verify:"
echo "   gcloud secrets versions access latest --secret=$SECRET_NAME --project=$PROJECT_ID"
echo "   gcloud run services describe $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --format='value(spec.template.spec.serviceAccountName)'"

