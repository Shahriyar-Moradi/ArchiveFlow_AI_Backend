#!/usr/bin/env python3
"""
GCS to Firestore Migration Script
This script syncs existing GCS flow metadata and file metadata to Firestore.
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to sys.path to import config and services
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from services.firestore_service import FirestoreService
from s3_service import S3Service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize services
firestore_service = None
s3_service = None

def init_services():
    """Initialize Firestore and S3 services"""
    global firestore_service, s3_service
    
    try:
        firestore_service = FirestoreService()
        logger.info("✅ Firestore service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FirestoreService: {e}")
        sys.exit(1)

    try:
        s3_service = S3Service()
        if not s3_service.client:
            raise Exception("GCS client not initialized in S3Service")
        logger.info("✅ GCS service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize S3Service (GCS): {e}")
        sys.exit(1)

async def migrate_flows(dry_run: bool = False):
    """Migrate flow metadata from GCS to Firestore"""
    logger.info("=" * 80)
    logger.info("Starting GCS to Firestore Flow Migration...")
    logger.info("=" * 80)
    
    # List all flows from GCS
    gcs_flows_result = s3_service.list_flows_from_s3(max_results=10000)
    if not gcs_flows_result.get("success"):
        logger.error(f"❌ Failed to list flows from GCS: {gcs_flows_result.get('error')}")
        return 0, 0, 0
    
    gcs_flows = gcs_flows_result.get("flows", [])
    logger.info(f"📊 Found {len(gcs_flows)} flows in GCS")
    
    created_count = 0
    updated_count = 0
    skipped_count = 0
    
    for gcs_flow in gcs_flows:
        flow_id = gcs_flow.get("flow_id")
        flow_name = gcs_flow.get("flow_name", f"Migrated Flow {flow_id}")
        created_at = gcs_flow.get("created_at", datetime.now().isoformat())
        total_files = gcs_flow.get("total_files", 0)
        status = gcs_flow.get("status", "created")
        
        if not flow_id:
            logger.warning(f"⚠️  Skipping GCS flow with missing flow_id: {gcs_flow}")
            skipped_count += 1
            continue
        
        # Check if flow already exists in Firestore
        firestore_flow = firestore_service.get_flow(flow_id)
        
        if dry_run:
            if firestore_flow:
                logger.info(f"[DRY RUN] Would update flow: {flow_id} ({flow_name})")
                updated_count += 1
            else:
                logger.info(f"[DRY RUN] Would create flow: {flow_id} ({flow_name})")
                created_count += 1
        else:
            if firestore_flow:
                logger.info(f"📝 Updating flow {flow_id} in Firestore")
                update_data = {
                    'flow_name': flow_name,
                    'document_count': total_files,
                    'status': status
                }
                firestore_service.update_flow(flow_id, update_data)
                updated_count += 1
            else:
                logger.info(f"➕ Creating flow {flow_id} in Firestore")
                flow_data = {
                    'flow_name': flow_name,
                    'source': 'gcs_migration',
                    'document_count': total_files,
                    'status': status,
                    'created_at': created_at
                }
                firestore_service.create_flow(flow_id, flow_data)
                created_count += 1
            
            # Migrate documents for this flow
            await migrate_documents_for_flow(flow_id, dry_run)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ Flow Migration Complete!")
    logger.info(f"   Created: {created_count} flows")
    logger.info(f"   Updated: {updated_count} flows")
    logger.info(f"   Skipped: {skipped_count} flows")
    logger.info("=" * 80)
    
    return created_count, updated_count, skipped_count

async def migrate_documents_for_flow(flow_id: str, dry_run: bool = False):
    """Migrate all documents for a specific flow from GCS to Firestore"""
    logger.info(f"\n  📄 Migrating documents for flow: {flow_id}")
    
    # List all files for this flow from GCS
    gcs_files_result = s3_service.get_flow_files_from_s3(flow_id, max_results_per_type=10000)
    if not gcs_files_result.get("success"):
        logger.error(f"  ❌ Failed to list GCS files for flow {flow_id}: {gcs_files_result.get('error')}")
        return 0, 0, 0
    
    gcs_temp_files = gcs_files_result.get("temp_files", [])
    gcs_organized_files = gcs_files_result.get("organized_files", [])
    gcs_failed_files = gcs_files_result.get("failed_files", [])
    all_gcs_files = gcs_temp_files + gcs_organized_files + gcs_failed_files
    
    if not all_gcs_files:
        logger.info(f"  ℹ️  No files found in GCS for flow {flow_id}")
        return 0, 0, 0
    
    logger.info(f"  📊 Found {len(all_gcs_files)} files in GCS for flow {flow_id}")
    
    created_count = 0
    updated_count = 0
    skipped_count = 0
    
    for gcs_file in all_gcs_files:
        s3_key = gcs_file.get("key") or gcs_file.get("s3_key")
        filename = s3_key.split('/')[-1] if s3_key else gcs_file.get("filename", "unknown")
        uploaded_at = gcs_file.get("uploaded_at") or gcs_file.get("last_modified", datetime.now().isoformat())
        file_size = gcs_file.get("size", 0)
        
        if not s3_key:
            logger.warning(f"  ⚠️  Skipping GCS file with missing s3_key: {gcs_file}")
            skipped_count += 1
            continue
        
        # Construct full GCS path
        gcs_path = f"gs://{s3_service.bucket_name}/{s3_key}"
        
        # Generate a document_id based on flow_id and filename
        document_id = f"{flow_id}_{filename}".replace('/', '_').replace(' ', '_')
        
        # Check if document already exists in Firestore
        try:
            firestore_doc = firestore_service.get_document(document_id)
        except:
            firestore_doc = None
        
        # Determine processing status
        if s3_key.startswith('temp/'):
            processing_status = 'uploaded'
            organized_path = ''
        elif '/organized_vouchers/' in s3_key or s3_key.startswith('organized_vouchers/'):
            processing_status = 'completed'
            organized_path = s3_key.replace(f"{s3_service.bucket_name}/", "")
        elif 'failed/' in s3_key:
            processing_status = 'failed'
            organized_path = ''
        else:
            processing_status = gcs_file.get("status", "uploaded")
            organized_path = gcs_file.get("organized_path", "")
        
        # Try to extract metadata from GCS blob
        metadata = {}
        try:
            blob = s3_service.bucket.blob(s3_key)
            if blob.exists() and blob.metadata:
                metadata = {k.replace('-', '_'): v for k, v in blob.metadata.items()}
        except Exception as e:
            pass  # Metadata fetch is optional
        
        doc_data = {
            'filename': filename,
            'flow_id': flow_id,
            'gcs_path': gcs_path,
            'organized_path': organized_path,
            'processing_status': processing_status,
            'file_size': file_size,
            'created_at': uploaded_at,
            'metadata': metadata
        }
        
        if 'temp/' in s3_key:
            doc_data['gcs_temp_path'] = s3_key
        
        if dry_run:
            if firestore_doc:
                logger.info(f"  [DRY RUN] Would update document: {document_id}")
                updated_count += 1
            else:
                logger.info(f"  [DRY RUN] Would create document: {document_id}")
                created_count += 1
        else:
            if firestore_doc:
                # Update existing document
                firestore_service.update_document(document_id, doc_data)
                updated_count += 1
            else:
                # Create new document
                firestore_service.create_document(document_id, doc_data)
                created_count += 1
    
    logger.info(f"  ✅ Document migration for flow {flow_id}: {created_count} created, {updated_count} updated, {skipped_count} skipped")
    return created_count, updated_count, skipped_count

async def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate GCS metadata to Firestore')
    parser.add_argument('--dry-run', action='store_true', help='Preview migration without making changes')
    args = parser.parse_args()
    
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("  GCS → FIRESTORE MIGRATION SCRIPT")
    logger.info("=" * 80)
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No changes will be made")
        logger.info("")
    
    init_services()
    
    start_time = datetime.now()
    
    created, updated, skipped = await migrate_flows(dry_run=args.dry_run)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("  MIGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  Flows Created: {created}")
    logger.info(f"  Flows Updated: {updated}")
    logger.info(f"  Flows Skipped: {skipped}")
    logger.info("=" * 80)
    logger.info("")
    
    if args.dry_run:
        logger.info("✅ Dry run complete. Run without --dry-run to apply changes.")
    else:
        logger.info("✅ Migration complete! Check migration.log for details.")

if __name__ == "__main__":
    asyncio.run(main())
