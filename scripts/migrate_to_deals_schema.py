"""
Migration script to convert existing data to use deals schema.

This script:
1. Scans all documents with agentId, propertyId, clientId
2. Groups by unique combinations (agent + client + property)
3. Creates deals for each unique combination
4. Updates documents with dealId
5. Updates property files with dealId
"""

import sys
import os
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import uuid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.firestore_service import FirestoreService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_unique_deal_combinations(firestore_service: FirestoreService) -> Dict[Tuple[str, str, str], list]:
    """
    Get all unique (agentId, clientId, propertyId) combinations from documents.
    Returns a dict mapping (agentId, clientId, propertyId) -> list of document_ids
    """
    logger.info("Scanning documents for unique agent+client+property combinations...")
    
    # Get all documents
    all_docs = list(firestore_service.documents_collection.stream())
    logger.info(f"Found {len(all_docs)} total documents")
    
    # Group by unique combinations
    combinations: Dict[Tuple[Optional[str], Optional[str], Optional[str]], list] = {}
    
    for doc in all_docs:
        data = doc.to_dict()
        agent_id = data.get('agentId') or data.get('agent_id')
        client_id = data.get('clientId') or data.get('client_id')
        property_id = data.get('propertyId') or data.get('property_id')
        
        # Only process documents that have all three relationships
        if agent_id and client_id and property_id:
            key = (agent_id, client_id, property_id)
            if key not in combinations:
                combinations[key] = []
            combinations[key].append(doc.id)
    
    logger.info(f"Found {len(combinations)} unique agent+client+property combinations")
    return combinations

def create_deals_for_combinations(
    firestore_service: FirestoreService,
    combinations: Dict[Tuple[str, str, str], list]
) -> Dict[Tuple[str, str, str], str]:
    """
    Create deals for each unique combination.
    Returns a dict mapping (agentId, clientId, propertyId) -> dealId
    """
    logger.info("Creating deals for unique combinations...")
    
    deal_map: Dict[Tuple[str, str, str], str] = {}
    created_count = 0
    existing_count = 0
    
    for (agent_id, client_id, property_id), document_ids in combinations.items():
        # Check if deal already exists
        existing_deals = firestore_service.list_deals(
            agent_id=agent_id,
            client_id=client_id,
            property_id=property_id,
            page_size=1
        )[0]
        
        if existing_deals:
            deal_id = existing_deals[0]['id']
            deal_map[(agent_id, client_id, property_id)] = deal_id
            existing_count += 1
            logger.debug(f"Using existing deal {deal_id} for agent {agent_id}, client {client_id}, property {property_id}")
        else:
            # Create new deal
            deal_id = f"deal_{uuid.uuid4().hex[:12]}"
            deal_data = {
                'agentId': agent_id,
                'clientId': client_id,
                'propertyId': property_id,
                'dealType': 'RENT',  # Default, can be updated later
                'stage': 'LEAD',
                'status': 'ACTIVE'
            }
            
            try:
                firestore_service.create_deal(deal_id, deal_data)
                deal_map[(agent_id, client_id, property_id)] = deal_id
                created_count += 1
                logger.info(f"Created deal {deal_id} for agent {agent_id}, client {client_id}, property {property_id}")
            except Exception as e:
                logger.error(f"Failed to create deal for agent {agent_id}, client {client_id}, property {property_id}: {e}")
    
    logger.info(f"Created {created_count} new deals, found {existing_count} existing deals")
    return deal_map

def update_documents_with_deal_ids(
    firestore_service: FirestoreService,
    deal_map: Dict[Tuple[str, str, str], str],
    combinations: Dict[Tuple[str, str, str], list]
):
    """Update all documents with their corresponding dealId"""
    logger.info("Updating documents with dealId...")
    
    updated_count = 0
    skipped_count = 0
    
    for (agent_id, client_id, property_id), document_ids in combinations.items():
        deal_id = deal_map.get((agent_id, client_id, property_id))
        if not deal_id:
            logger.warning(f"No deal found for combination agent {agent_id}, client {client_id}, property {property_id}")
            skipped_count += len(document_ids)
            continue
        
        for document_id in document_ids:
            try:
                # Check if document already has dealId
                doc = firestore_service.get_document(document_id)
                if doc and doc.get('dealId'):
                    logger.debug(f"Document {document_id} already has dealId {doc.get('dealId')}, skipping")
                    skipped_count += 1
                    continue
                
                # Update document with dealId
                firestore_service.update_document(document_id, {'dealId': deal_id})
                updated_count += 1
                
                if updated_count % 100 == 0:
                    logger.info(f"Updated {updated_count} documents so far...")
            except Exception as e:
                logger.error(f"Failed to update document {document_id}: {e}")
    
    logger.info(f"Updated {updated_count} documents with dealId, skipped {skipped_count} (already had dealId)")

def update_property_files_with_deal_ids(firestore_service: FirestoreService):
    """Update property files with dealId based on their agent_id, client_id, property_id"""
    logger.info("Updating property files with dealId...")
    
    # Get all property files
    all_property_files = list(firestore_service.property_files_collection.stream())
    logger.info(f"Found {len(all_property_files)} property files")
    
    updated_count = 0
    skipped_count = 0
    not_found_count = 0
    
    for pf_doc in all_property_files:
        pf_data = pf_doc.to_dict()
        pf_id = pf_doc.id
        
        # Skip if already has dealId
        if pf_data.get('dealId'):
            skipped_count += 1
            continue
        
        agent_id = pf_data.get('agent_id') or pf_data.get('agentId')
        client_id = pf_data.get('client_id') or pf_data.get('clientId')
        property_id = pf_data.get('property_id') or pf_data.get('propertyId')
        
        if not (agent_id and client_id and property_id):
            logger.debug(f"Property file {pf_id} missing required fields (agent_id, client_id, property_id), skipping")
            not_found_count += 1
            continue
        
        # Find or create deal
        deal = firestore_service.find_or_create_deal(
            agent_id=agent_id,
            client_id=client_id,
            property_id=property_id,
            deal_type=pf_data.get('transaction_type', 'RENT'),
            stage='LEAD'
        )
        
        if deal:
            try:
                firestore_service.update_property_file(pf_id, {'dealId': deal.get('id')})
                updated_count += 1
                logger.debug(f"Updated property file {pf_id} with dealId {deal.get('id')}")
            except Exception as e:
                logger.error(f"Failed to update property file {pf_id}: {e}")
        else:
            logger.warning(f"Could not find or create deal for property file {pf_id}")
            not_found_count += 1
    
    logger.info(f"Updated {updated_count} property files with dealId, skipped {skipped_count} (already had dealId), {not_found_count} not found/missing fields")

def main():
    """Main migration function"""
    logger.info("=" * 60)
    logger.info("Starting migration to deals schema")
    logger.info("=" * 60)
    
    try:
        firestore_service = FirestoreService()
        logger.info("Firestore service initialized")
        
        # Step 1: Get unique combinations from documents
        combinations = get_unique_deal_combinations(firestore_service)
        
        if not combinations:
            logger.info("No documents with complete relationships found. Migration complete.")
            return
        
        # Step 2: Create deals for each combination
        deal_map = create_deals_for_combinations(firestore_service, combinations)
        
        # Step 3: Update documents with dealId
        update_documents_with_deal_ids(firestore_service, deal_map, combinations)
        
        # Step 4: Update property files with dealId
        update_property_files_with_deal_ids(firestore_service)
        
        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
