#!/usr/bin/env python3
"""
Script to verify that all Firestore collections are empty.

This script will:
1. List all collections in Firestore
2. Count documents in each collection
3. Verify that all known collections are empty
"""

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import firestore
from config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All known collections
KNOWN_COLLECTIONS = [
    'documents',
    'processing_jobs',
    'flows',
    'clients',
    'properties',
    'property_files',
    'agents',
    'deals'
]


def list_all_collections(db: firestore.Client):
    """List all collections in Firestore"""
    try:
        collections = db.collections()
        collection_names = [col.id for col in collections]
        return collection_names
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        return []


def count_collection_documents(db: firestore.Client, collection_name: str) -> int:
    """Count documents in a collection"""
    try:
        collection_ref = db.collection(collection_name)
        # Try count aggregation first
        try:
            count_query = collection_ref.count()
            results = count_query.get()
            if results and results[0] and hasattr(results[0][0], "value"):
                return int(results[0][0].value)
        except:
            pass
        # Fallback: count by streaming
        return sum(1 for _ in collection_ref.stream())
    except Exception as e:
        logger.warning(f"Could not count documents in '{collection_name}': {e}")
        return -1


def main():
    try:
        project_id = settings.FIRESTORE_PROJECT_ID
        logger.info(f"🔌 Connecting to Firestore project: {project_id}")
        db = firestore.Client(project=project_id)
        logger.info("✅ Connected to Firestore")
        
        # List all collections
        logger.info("\n📋 Listing all collections in Firestore...")
        all_collections = list_all_collections(db)
        
        if not all_collections:
            logger.info("✅ No collections found in Firestore (database is completely empty)")
            return
        
        logger.info(f"Found {len(all_collections)} collection(s):")
        for col_name in sorted(all_collections):
            logger.info(f"  - {col_name}")
        
        # Count documents in each collection
        logger.info("\n📊 Counting documents in each collection...")
        total_documents = 0
        collections_with_data = []
        
        for collection_name in sorted(all_collections):
            count = count_collection_documents(db, collection_name)
            if count >= 0:
                logger.info(f"  {collection_name}: {count} documents")
                if count > 0:
                    collections_with_data.append(collection_name)
                    total_documents += count
            else:
                logger.warning(f"  {collection_name}: Could not determine count")
        
        logger.info(f"\n📈 Total documents across all collections: {total_documents}")
        
        # Check known collections
        logger.info("\n🔍 Verifying known collections...")
        unknown_collections = [col for col in all_collections if col not in KNOWN_COLLECTIONS]
        missing_known_collections = [col for col in KNOWN_COLLECTIONS if col not in all_collections]
        
        if unknown_collections:
            logger.warning(f"⚠️  Found {len(unknown_collections)} unknown collection(s):")
            for col in unknown_collections:
                count = count_collection_documents(db, col)
                logger.warning(f"  - {col}: {count} documents")
        
        if missing_known_collections:
            logger.info(f"ℹ️  {len(missing_known_collections)} known collection(s) not found (this is normal if never used):")
            for col in missing_known_collections:
                logger.info(f"  - {col}")
        
        # Final verification
        logger.info("\n" + "=" * 70)
        if total_documents == 0:
            logger.info("✅ VERIFICATION PASSED: All Firestore collections are empty")
            logger.info("=" * 70)
            logger.info(f"Total collections checked: {len(all_collections)}")
            logger.info(f"Total documents: 0")
            logger.info("✅ All data has been successfully deleted from Firestore")
        else:
            logger.error("❌ VERIFICATION FAILED: Some collections still contain data")
            logger.info("=" * 70)
            logger.error(f"Collections with data: {', '.join(collections_with_data)}")
            logger.error(f"Total documents remaining: {total_documents}")
            logger.error("\n⚠️  Run the delete script again to remove remaining data:")
            logger.error("  python scripts/delete_all_data.py --confirm --firestore-only")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
