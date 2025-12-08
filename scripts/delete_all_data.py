#!/usr/bin/env python3
"""
Script to delete all records from GCS bucket and all Firestore collections.

WARNING: This script will permanently delete ALL data from:
- All blobs in the GCS bucket
- All documents in all Firestore collections

Use with extreme caution. This action cannot be undone.

Usage:
    python scripts/delete_all_data.py [--confirm] [--gcs-only] [--firestore-only]
    
    Without --confirm flag, the script will run in dry-run mode and show what would be deleted.
    With --confirm flag, the script will actually delete all records.
    
    Use --gcs-only to delete only GCS data
    Use --firestore-only to delete only Firestore data
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import firestore
from google.cloud import storage
from google.oauth2 import service_account
from config import settings
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All Firestore collections in the database
FIRESTORE_COLLECTIONS = [
    'documents',
    'processing_jobs',
    'flows',
    'clients',
    'properties',
    'property_files',
    'agents',
    'deals'
]


def get_gcs_client():
    """Initialize GCS client with flexible credential loading"""
    bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("GCS_BUCKET") or settings.GCS_BUCKET_NAME
    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCS_PROJECT_ID") or settings.GCS_PROJECT_ID
    
    try:
        credentials = None
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Try environment variable first
        if key_path and os.path.exists(key_path):
            credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            logger.info(f"Using GCS credentials from GOOGLE_APPLICATION_CREDENTIALS: {key_path}")
        else:
            # Fallback to local key file
            fallback_key = os.path.join(Path(__file__).parent.parent, "voucher-storage-key.json")
            if os.path.exists(fallback_key):
                credentials = service_account.Credentials.from_service_account_file(
                    fallback_key,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info(f"Using GCS credentials from local key file: {fallback_key}")
            else:
                logger.info("Using application-default credentials for GCS")
        
        # Initialize client with credentials (or None for default)
        if credentials:
            client = storage.Client(credentials=credentials, project=project_id)
        else:
            client = storage.Client(project=project_id)
        
        return client, bucket_name
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        raise


def count_gcs_blobs(client: storage.Client, bucket_name: str) -> int:
    """Count all blobs in the GCS bucket"""
    try:
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs())
        return len(blobs)
    except Exception as e:
        logger.error(f"Failed to count GCS blobs: {e}")
        return -1


def delete_all_gcs_blobs(client: storage.Client, bucket_name: str, dry_run: bool = True) -> int:
    """Delete all blobs from the GCS bucket"""
    try:
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.warning(f"Bucket '{bucket_name}' does not exist")
            return 0
        
        blobs = list(bucket.list_blobs())
        total_blobs = len(blobs)
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would delete {total_blobs} blobs from GCS bucket '{bucket_name}'")
            # Show some example blob names
            if total_blobs > 0:
                logger.info(f"  Example blobs (showing first 5):")
                for i, blob in enumerate(blobs[:5]):
                    logger.info(f"    - {blob.name}")
                if total_blobs > 5:
                    logger.info(f"    ... and {total_blobs - 5} more")
        else:
            deleted_count = 0
            batch_size = 100  # Process in batches for better progress tracking
            
            for i in range(0, total_blobs, batch_size):
                batch = blobs[i:i + batch_size]
                for blob in batch:
                    try:
                        blob.delete()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"  Failed to delete blob '{blob.name}': {e}")
                
                logger.info(f"  Deleted batch: {deleted_count}/{total_blobs} blobs")
            
            logger.info(f"  ✅ Deleted {deleted_count} blobs from GCS bucket '{bucket_name}'")
            return deleted_count
        
        return total_blobs
        
    except Exception as e:
        logger.error(f"  ❌ Error deleting GCS blobs: {e}")
        raise


def get_collection_count(db: firestore.Client, collection_name: str) -> int:
    """Get the count of documents in a collection"""
    try:
        collection_ref = db.collection(collection_name)
        # Use count aggregation if available
        try:
            count_query = collection_ref.count()
            results = count_query.get()
            if results and results[0] and hasattr(results[0][0], "value"):
                return int(results[0][0].value)
        except:
            pass
        # Fallback: count by streaming (slower but works)
        return sum(1 for _ in collection_ref.stream())
    except Exception as e:
        logger.warning(f"Could not get count for {collection_name}: {e}")
        return -1


def delete_collection(db: firestore.Client, collection_name: str, dry_run: bool = True) -> int:
    """Delete all documents from a collection"""
    collection_ref = db.collection(collection_name)
    deleted_count = 0
    
    try:
        # Get all documents
        docs = list(collection_ref.stream())
        
        if dry_run:
            # Count documents without deleting
            deleted_count = len(docs)
            logger.info(f"  [DRY RUN] Would delete {deleted_count} documents from '{collection_name}'")
        else:
            # Delete documents in batches
            batch = db.batch()
            batch_count = 0
            max_batch_size = 500  # Firestore batch limit
            
            for doc in docs:
                batch.delete(doc.reference)
                batch_count += 1
                deleted_count += 1
                
                # Commit batch when it reaches the limit
                if batch_count >= max_batch_size:
                    batch.commit()
                    logger.info(f"  Deleted batch of {batch_count} documents from '{collection_name}' (total: {deleted_count})")
                    batch = db.batch()
                    batch_count = 0
            
            # Commit remaining documents
            if batch_count > 0:
                batch.commit()
                logger.info(f"  Deleted final batch of {batch_count} documents from '{collection_name}'")
            
            logger.info(f"  ✅ Deleted {deleted_count} documents from '{collection_name}'")
        
    except Exception as e:
        logger.error(f"  ❌ Error deleting from '{collection_name}': {e}")
        raise
    
    return deleted_count


def delete_firestore_data(project_id: str, dry_run: bool = True) -> Dict[str, int]:
    """Delete all Firestore data"""
    try:
        # Initialize Firestore client
        logger.info(f"🔌 Connecting to Firestore project: {project_id}")
        db = firestore.Client(project=project_id)
        logger.info("✅ Connected to Firestore")
        
        # Get counts for all collections
        logger.info("\n📊 Counting documents in each collection...")
        collection_counts: Dict[str, int] = {}
        total_documents = 0
        
        for collection_name in FIRESTORE_COLLECTIONS:
            count = get_collection_count(db, collection_name)
            collection_counts[collection_name] = count
            if count >= 0:
                total_documents += count
                logger.info(f"  {collection_name}: {count} documents")
            else:
                logger.warning(f"  {collection_name}: Could not determine count")
        
        logger.info(f"\n📈 Total documents across all collections: {total_documents}")
        
        if total_documents == 0:
            logger.info("✅ No documents found in Firestore. Nothing to delete.")
            return {}
        
        if dry_run:
            logger.info("\n[DRY RUN] Would delete all Firestore data")
            return collection_counts
        
        # Delete from all collections
        logger.info("\n🗑️  Starting Firestore deletion process...")
        deleted_counts = {}
        total_deleted = 0
        
        for collection_name in FIRESTORE_COLLECTIONS:
            logger.info(f"\n📂 Processing collection: '{collection_name}'")
            try:
                deleted = delete_collection(db, collection_name, dry_run=False)
                deleted_counts[collection_name] = deleted
                total_deleted += deleted
            except Exception as e:
                logger.error(f"Failed to delete from '{collection_name}': {e}")
                deleted_counts[collection_name] = 0
                continue
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FIRESTORE DELETION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total documents deleted: {total_deleted}")
        
        return deleted_counts
        
    except Exception as e:
        logger.error(f"❌ Fatal error deleting Firestore data: {e}")
        raise


def delete_gcs_data(dry_run: bool = True) -> int:
    """Delete all GCS data"""
    try:
        client, bucket_name = get_gcs_client()
        logger.info(f"🔌 Connected to GCS bucket: {bucket_name}")
        
        # Count blobs
        logger.info("\n📊 Counting blobs in GCS bucket...")
        blob_count = count_gcs_blobs(client, bucket_name)
        logger.info(f"  Total blobs: {blob_count}")
        
        if blob_count == 0:
            logger.info("✅ No blobs found in GCS bucket. Nothing to delete.")
            return 0
        
        if blob_count < 0:
            logger.warning("⚠️  Could not determine blob count, but will attempt deletion")
        
        if dry_run:
            logger.info("\n[DRY RUN] Would delete all GCS blobs")
            delete_all_gcs_blobs(client, bucket_name, dry_run=True)
            return blob_count
        
        # Delete all blobs
        logger.info("\n🗑️  Starting GCS deletion process...")
        deleted_count = delete_all_gcs_blobs(client, bucket_name, dry_run=False)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ GCS DELETION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total blobs deleted: {deleted_count}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Fatal error deleting GCS data: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Delete all records from GCS bucket and all Firestore collections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This script will permanently delete ALL data from:
- All blobs in the GCS bucket
- All documents in all Firestore collections

This action cannot be undone. Use with extreme caution.

Examples:
  # Dry run (safe, shows what would be deleted):
  python scripts/delete_all_data.py
  
  # Actually delete all data (requires confirmation):
  python scripts/delete_all_data.py --confirm
  
  # Delete only GCS data:
  python scripts/delete_all_data.py --confirm --gcs-only
  
  # Delete only Firestore data:
  python scripts/delete_all_data.py --confirm --firestore-only
        """
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Actually delete the data (without this flag, runs in dry-run mode)'
    )
    parser.add_argument(
        '--gcs-only',
        action='store_true',
        help='Delete only GCS data (skip Firestore)'
    )
    parser.add_argument(
        '--firestore-only',
        action='store_true',
        help='Delete only Firestore data (skip GCS)'
    )
    parser.add_argument(
        '--project-id',
        type=str,
        default=None,
        help=f'Firestore project ID (defaults to {settings.FIRESTORE_PROJECT_ID})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip interactive confirmation (use with caution)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gcs_only and args.firestore_only:
        logger.error("❌ Cannot specify both --gcs-only and --firestore-only")
        sys.exit(1)
    
    project_id = args.project_id or settings.FIRESTORE_PROJECT_ID
    dry_run = not args.confirm
    
    delete_gcs = not args.firestore_only
    delete_firestore = not args.gcs_only
    
    if dry_run:
        logger.warning("=" * 70)
        logger.warning("⚠️  DRY RUN MODE - No data will be deleted")
        logger.warning("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("⚠️  WARNING: This will DELETE ALL DATA!")
        if delete_gcs:
            logger.error("  - All blobs in GCS bucket")
        if delete_firestore:
            logger.error("  - All documents in all Firestore collections")
        logger.error("=" * 70)
        if not args.force:
            response = input("Type 'DELETE ALL' to confirm: ")
            if response != 'DELETE ALL':
                logger.info("❌ Deletion cancelled. No data was deleted.")
                return
        else:
            logger.warning("⚠️  --force flag used: Skipping interactive confirmation")
    
    try:
        results = {}
        
        # Delete GCS data
        if delete_gcs:
            logger.info("\n" + "=" * 70)
            logger.info("🗑️  GCS BUCKET DELETION")
            logger.info("=" * 70)
            try:
                deleted_blobs = delete_gcs_data(dry_run=dry_run)
                results['gcs_blobs_deleted'] = deleted_blobs
            except Exception as e:
                logger.error(f"❌ Failed to delete GCS data: {e}")
                results['gcs_error'] = str(e)
        
        # Delete Firestore data
        if delete_firestore:
            logger.info("\n" + "=" * 70)
            logger.info("🗑️  FIRESTORE DELETION")
            logger.info("=" * 70)
            try:
                deleted_counts = delete_firestore_data(project_id, dry_run=dry_run)
                results['firestore_deleted'] = deleted_counts
                total_firestore = sum(deleted_counts.values()) if deleted_counts else 0
                results['firestore_total'] = total_firestore
            except Exception as e:
                logger.error(f"❌ Failed to delete Firestore data: {e}")
                results['firestore_error'] = str(e)
        
        # Summary
        logger.info("\n" + "=" * 70)
        if dry_run:
            logger.info("📊 DRY RUN SUMMARY - No data was deleted")
        else:
            logger.info("✅ DELETION COMPLETE")
        logger.info("=" * 70)
        
        if delete_gcs and 'gcs_blobs_deleted' in results:
            logger.info(f"GCS blobs: {results['gcs_blobs_deleted']}")
        if delete_firestore and 'firestore_total' in results:
            logger.info(f"Firestore documents: {results['firestore_total']}")
        
        if dry_run:
            logger.info("\nTo actually delete the data, run:")
            logger.info("  python scripts/delete_all_data.py --confirm")
        else:
            logger.info("\n⚠️  All data has been permanently deleted.")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
