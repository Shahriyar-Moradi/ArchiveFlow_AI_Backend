#!/usr/bin/env python3
"""
Configure CORS on GCS bucket to allow direct uploads from frontend
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import storage
from google.oauth2 import service_account
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_cors():
    """Configure CORS on GCS bucket"""
    try:
        bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("GCS_BUCKET") or "voucher-bucket-1"
        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCS_PROJECT_ID") or "rocasoft"
        
        # Initialize GCS client
        credentials = None
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if key_path and os.path.exists(key_path):
            credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            logger.info(f"Using credentials from: {key_path}")
        else:
            fallback_key = Path(__file__).parent.parent / "voucher-storage-key.json"
            if fallback_key.exists():
                credentials = service_account.Credentials.from_service_account_file(
                    str(fallback_key),
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info(f"Using credentials from: {fallback_key}")
        
        if credentials:
            client = storage.Client(credentials=credentials, project=project_id)
        else:
            client = storage.Client(project=project_id)
        
        bucket = client.bucket(bucket_name)
        
        # CORS configuration
        cors_config = [
            {
                "origin": [
                    "http://localhost:3000",
                    "http://localhost:8080",
                    "http://localhost:4200",
                    "http://localhost",
                    "https://localhost",
                    "capacitor://localhost",
                    "ionic://localhost",
                    # Add your production domains here
                    # "https://yourdomain.com",
                ],
                "method": ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
                "responseHeader": [
                    "Content-Type",
                    "Content-Length",
                    "Content-Range",
                    "Accept",
                    "Authorization",
                    "X-Goog-Upload-Protocol",
                    "X-Goog-Upload-Command",
                    "X-Goog-Upload-Offset",
                    "X-Goog-Upload-Status",
                    "X-Goog-Upload-Chunk-Granularity",
                    "X-Goog-Upload-Header-Content-Length",
                    "X-Goog-Upload-Header-Content-Type",
                ],
                "maxAgeSeconds": 3600
            }
        ]
        
        # Set CORS configuration
        bucket.cors = cors_config
        bucket.patch()
        
        logger.info(f"✅ Successfully configured CORS on bucket: {bucket_name}")
        logger.info(f"   Allowed origins: {cors_config[0]['origin']}")
        logger.info(f"   Allowed methods: {cors_config[0]['method']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to configure CORS: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = configure_cors()
    sys.exit(0 if success else 1)

