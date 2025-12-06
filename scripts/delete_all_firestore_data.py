#!/usr/bin/env python3
"""
Script to delete all records from all Firestore collections.

WARNING: This script will permanently delete ALL data from ALL collections in Firestore.
Use with extreme caution. This action cannot be undone.

Usage:
    python scripts/delete_all_firestore_data.py [--confirm]
    
    Without --confirm flag, the script will run in dry-run mode and show what would be deleted.
    With --confirm flag, the script will actually delete all records.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import firestore
from config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All collections in the database
COLLECTIONS = [
    'documents',
    'processing_jobs',
    'flows',
    'clients',
    'properties',
    'property_files',
    'agents',
    'deals'
]


def get_collection_count(db: firestore.Client, collection_name: str) -> int:
    """Get the count of documents in a collection"""
    try:
        collection_ref = db.collection(collection_name)
        # Use count aggregation if available
        count_query = collection_ref.count()
        results = count_query.get()
        if results and results[0] and hasattr(results[0][0], "value"):
            return int(results[0][0].value)
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
        docs = collection_ref.stream()
        
        if dry_run:
            # Count documents without deleting
            for doc in docs:
                deleted_count += 1
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


def main():
    parser = argparse.ArgumentParser(
        description='Delete all records from all Firestore collections',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: This script will permanently delete ALL data from ALL collections.
This action cannot be undone. Use with extreme caution.

Examples:
  # Dry run (safe, shows what would be deleted):
  python scripts/delete_all_firestore_data.py
  
  # Actually delete all data (requires confirmation):
  python scripts/delete_all_firestore_data.py --confirm
        """
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Actually delete the data (without this flag, runs in dry-run mode)'
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
    
    project_id = args.project_id or settings.FIRESTORE_PROJECT_ID
    dry_run = not args.confirm
    
    if dry_run:
        logger.warning("=" * 70)
        logger.warning("⚠️  DRY RUN MODE - No data will be deleted")
        logger.warning("=" * 70)
    else:
        logger.error("=" * 70)
        logger.error("⚠️  WARNING: This will DELETE ALL DATA from ALL collections!")
        logger.error("=" * 70)
        if not args.force:
            response = input("Type 'DELETE ALL' to confirm: ")
            if response != 'DELETE ALL':
                logger.info("❌ Deletion cancelled. No data was deleted.")
                return
        else:
            logger.warning("⚠️  --force flag used: Skipping interactive confirmation")
    
    try:
        # Initialize Firestore client
        logger.info(f"🔌 Connecting to Firestore project: {project_id}")
        db = firestore.Client(project=project_id)
        logger.info("✅ Connected to Firestore")
        
        # Get counts for all collections
        logger.info("\n📊 Counting documents in each collection...")
        collection_counts: Dict[str, int] = {}
        total_documents = 0
        
        for collection_name in COLLECTIONS:
            count = get_collection_count(db, collection_name)
            collection_counts[collection_name] = count
            if count >= 0:
                total_documents += count
                logger.info(f"  {collection_name}: {count} documents")
            else:
                logger.warning(f"  {collection_name}: Could not determine count")
        
        logger.info(f"\n📈 Total documents across all collections: {total_documents}")
        
        if total_documents == 0:
            logger.info("✅ No documents found. Nothing to delete.")
            return
        
        if dry_run:
            logger.info("\n" + "=" * 70)
            logger.info("DRY RUN SUMMARY - No data was deleted")
            logger.info("=" * 70)
            logger.info("To actually delete the data, run:")
            logger.info("  python scripts/delete_all_firestore_data.py --confirm")
            return
        
        # Delete from all collections
        logger.info("\n🗑️  Starting deletion process...")
        total_deleted = 0
        
        for collection_name in COLLECTIONS:
            logger.info(f"\n📂 Processing collection: '{collection_name}'")
            try:
                deleted = delete_collection(db, collection_name, dry_run=False)
                total_deleted += deleted
            except Exception as e:
                logger.error(f"Failed to delete from '{collection_name}': {e}")
                continue
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ DELETION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total documents deleted: {total_deleted}")
        logger.info("\n⚠️  All data has been permanently deleted from Firestore.")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

