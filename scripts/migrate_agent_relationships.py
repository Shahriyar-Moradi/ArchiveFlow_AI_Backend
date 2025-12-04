"""
Migration script to assign agentId to existing documents and properties.

This script:
1. Finds all documents without agentId
2. Finds all properties without agentId
3. Assigns them to a default agent (first user or admin)
4. Updates Firestore records in batch
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.firestore_service import FirestoreService
from auth_service import auth_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_default_agent_id():
    """Get the first available user as default agent"""
    try:
        # Try to get users from auth_service
        if hasattr(auth_service, 'users') and auth_service.users:
            first_user_id = list(auth_service.users.keys())[0]
            logger.info(f"Using first user as default agent: {first_user_id}")
            return first_user_id
        
        # Fallback: try simple_auth
        from simple_auth import simple_auth
        if hasattr(simple_auth, 'users') and simple_auth.users:
            first_user_id = list(simple_auth.users.keys())[0]
            logger.info(f"Using first user from simple_auth as default agent: {first_user_id}")
            return first_user_id
        
        logger.warning("No users found in auth systems. Cannot assign default agent.")
        return None
    except Exception as e:
        logger.error(f"Error getting default agent: {e}")
        return None

def migrate_documents(firestore_service: FirestoreService, default_agent_id: str):
    """Migrate documents without agentId"""
    try:
        logger.info("Starting document migration...")
        
        # Query all documents
        query = firestore_service.documents_collection
        docs = list(query.stream())
        
        migrated_count = 0
        skipped_count = 0
        
        for doc in docs:
            data = doc.to_dict()
            
            # Skip if already has agentId
            if data.get('agentId'):
                skipped_count += 1
                continue
            
            # Update document with default agentId
            try:
                firestore_service.update_document(doc.id, {
                    'agentId': default_agent_id
                })
                migrated_count += 1
                if migrated_count % 100 == 0:
                    logger.info(f"Migrated {migrated_count} documents...")
            except Exception as e:
                logger.error(f"Failed to migrate document {doc.id}: {e}")
        
        logger.info(f"✅ Document migration complete: {migrated_count} migrated, {skipped_count} skipped")
        return migrated_count, skipped_count
        
    except Exception as e:
        logger.error(f"Error during document migration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0, 0

def migrate_properties(firestore_service: FirestoreService, default_agent_id: str):
    """Migrate properties without agentId"""
    try:
        logger.info("Starting property migration...")
        
        # Query all properties
        query = firestore_service.properties_collection
        docs = list(query.stream())
        
        migrated_count = 0
        skipped_count = 0
        
        for doc in docs:
            data = doc.to_dict()
            
            # Skip if already has agentId
            if data.get('agentId'):
                skipped_count += 1
                continue
            
            # Update property with default agentId
            try:
                doc_ref = firestore_service.properties_collection.document(doc.id)
                doc_ref.update({
                    'agentId': default_agent_id
                })
                migrated_count += 1
                if migrated_count % 50 == 0:
                    logger.info(f"Migrated {migrated_count} properties...")
            except Exception as e:
                logger.error(f"Failed to migrate property {doc.id}: {e}")
        
        logger.info(f"✅ Property migration complete: {migrated_count} migrated, {skipped_count} skipped")
        return migrated_count, skipped_count
        
    except Exception as e:
        logger.error(f"Error during property migration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0, 0

def main():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("Agent Relationship Migration Script")
    logger.info("=" * 60)
    
    # Get default agent
    default_agent_id = get_default_agent_id()
    if not default_agent_id:
        logger.error("❌ Cannot proceed without a default agent. Please ensure at least one user exists.")
        return
    
    logger.info(f"Default agent ID: {default_agent_id}")
    
    # Initialize Firestore service
    try:
        firestore_service = FirestoreService()
        logger.info("✅ Firestore service initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firestore service: {e}")
        return
    
    # Migrate documents
    doc_migrated, doc_skipped = migrate_documents(firestore_service, default_agent_id)
    
    # Migrate properties
    prop_migrated, prop_skipped = migrate_properties(firestore_service, default_agent_id)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Migration Summary")
    logger.info("=" * 60)
    logger.info(f"Documents: {doc_migrated} migrated, {doc_skipped} skipped")
    logger.info(f"Properties: {prop_migrated} migrated, {prop_skipped} skipped")
    logger.info(f"Total migrated: {doc_migrated + prop_migrated}")
    logger.info("=" * 60)
    logger.info("✅ Migration complete!")

if __name__ == "__main__":
    main()

