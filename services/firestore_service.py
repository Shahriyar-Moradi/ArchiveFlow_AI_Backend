"""
Firestore service for storing document metadata and job status
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from google.cloud import firestore
from google.cloud.firestore import Query, FieldFilter, FieldPath

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings

logger = logging.getLogger(__name__)


def _get_query_count(query: Query) -> int:
    """Return the total number of documents for a query using aggregation.

    Why: list endpoints for agents, clients, and properties need accurate totals
    for pagination, but fetching every document to compute `len()` is slow and
    expensive. Firestore's count aggregation runs server-side and only returns
    the count metadata, keeping responses fast. If aggregation is unsupported
    (older emulator/SDK), we fall back to -1 so callers can skip showing totals
    instead of triggering costly full scans.
    """

    try:
        count_query = query.count()
        results = count_query.get()
        if results and results[0] and hasattr(results[0][0], "value"):
            return int(results[0][0].value)
    except Exception as e:
        logger.warning(f"Count aggregation failed: {e}")
    return -1

class FirestoreService:
    """Service for interacting with Firestore database"""

    def __init__(self):
        """Initialize Firestore client"""
        try:
            self.db = firestore.Client(project=settings.FIRESTORE_PROJECT_ID)
            self.documents_collection = self.db.collection(settings.FIRESTORE_COLLECTION_DOCUMENTS)
            self.jobs_collection = self.db.collection(settings.FIRESTORE_COLLECTION_JOBS)
            self.flows_collection = self.db.collection(settings.FIRESTORE_COLLECTION_FLOWS)
            self.clients_collection = self.db.collection('clients')
            self.properties_collection = self.db.collection('properties')
            self.property_files_collection = self.db.collection('property_files')
            self.agents_collection = self.db.collection('agents')
            self.deals_collection = self.db.collection('deals')
            logger.info(f"Firestore client initialized for project: {settings.FIRESTORE_PROJECT_ID}")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            raise

    @staticmethod
    def _fetch_documents_by_ids(collection, ids: set[str]) -> List[Dict[str, Any]]:
        """Fetch documents by IDs in efficient chunks using document_id IN queries.

        Rationale:
        - Calling `.document(id).get()` per record makes agent/client/property list
          endpoints slow because each fetch is a separate network round trip.
        - Firestore limits `IN` filters to 10 elements, so we batch IDs into chunks
          to stay within limits while keeping requests to the absolute minimum.

        Use this helper whenever you need to hydrate related entities (e.g.,
        properties from a batch of deals) instead of looping over individual gets.
        """

        documents: List[Dict[str, Any]] = []
        if not ids:
            return documents

        id_list = list(ids)
        chunk_size = 10
        for i in range(0, len(id_list), chunk_size):
            chunk_ids = id_list[i : i + chunk_size]
            doc_refs = [collection.document(doc_id) for doc_id in chunk_ids]
            query = collection.where(filter=FieldFilter(FieldPath.document_id(), "in", doc_refs))
            for doc in query.stream():
                data = doc.to_dict()
                data["id"] = doc.id
                documents.append(data)

        return documents
    
    # Document Operations
    
    def create_document(self, document_id: str, data: Dict[str, Any]) -> str:
        """Create a new document record"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            logger.info(f"Created document record: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['document_id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None
    
    def update_document(self, document_id: str, data: Dict[str, Any]) -> bool:
        """Update a document record"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated document record: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update document record: {e}")
            return False
    
    def update_compliance_check_results(self, document_id: str, compliance_data: Dict[str, Any]) -> bool:
        """Update compliance check results in document record"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            
            # Store compliance data in a dedicated field
            update_data = {
                'compliance_check': compliance_data,
                'updated_at': firestore.SERVER_TIMESTAMP
            }
            
            doc_ref.update(update_data)
            logger.info(f"Updated compliance check results for document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update compliance check results: {e}")
            return False
    
    def get_compliance_check_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get compliance check results for a document"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                return data.get('compliance_check')
            return None
        except Exception as e:
            logger.error(f"Failed to get compliance check results: {e}")
            return None
    
    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List documents with efficient cursor-based pagination and optional filters"""
        try:
            from services.category_mapper import map_backend_to_ui_category
            
            query = self.documents_collection
            
            # Apply filters (except ui_category for now - we'll filter in memory)
            ui_category_filter = None
            if filters:
                if filters.get('classification'):
                    query = query.where(filter=FieldFilter('metadata.classification', '==', filters['classification']))
                if filters.get('ui_category'):
                    # Store ui_category filter to apply after fetching (to handle missing ui_category)
                    ui_category_filter = filters['ui_category']
                if filters.get('branch_id'):
                    query = query.where(filter=FieldFilter('metadata.branch_id', '==', filters['branch_id']))
                if filters.get('date_from'):
                    query = query.where(filter=FieldFilter('metadata.document_date', '>=', filters['date_from']))
                if filters.get('date_to'):
                    query = query.where(filter=FieldFilter('metadata.document_date', '<=', filters['date_to']))
                if filters.get('flow_id'):
                    query = query.where(filter=FieldFilter('flow_id', '==', filters['flow_id']))
                if filters.get('agentId'):
                    query = query.where(filter=FieldFilter('agentId', '==', filters['agentId']))
                if filters.get('propertyId'):
                    query = query.where(filter=FieldFilter('propertyId', '==', filters['propertyId']))
                if filters.get('clientId'):
                    query = query.where(filter=FieldFilter('clientId', '==', filters['clientId']))
                if filters.get('dealId'):
                    query = query.where(filter=FieldFilter('dealId', '==', filters['dealId']))
            
            # Order by created_at descending
            query = query.order_by('created_at', direction=Query.DESCENDING)
            
            # Use cursor-based pagination if cursor provided (more efficient than offset)
            if cursor_doc_id:
                cursor_doc = self.documents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                # Fallback to offset for first-time pagination without cursor
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            # If ui_category filter is needed, fetch more and filter in memory
            fetch_size = page_size * 3 if ui_category_filter else page_size
            docs = list(query.limit(fetch_size).stream())
            
            documents = []
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                metadata = data.get('metadata', {})
                
                # Get or compute ui_category
                ui_category = metadata.get('ui_category')
                if not ui_category:
                    classification = metadata.get('classification') or data.get('document_type') or data.get('classification')
                    ui_category = map_backend_to_ui_category(classification)
                    metadata['ui_category'] = ui_category
                    data['metadata'] = metadata
                
                # Apply ui_category filter if specified
                if ui_category_filter and ui_category != ui_category_filter:
                    continue
                
                documents.append(data)
                
                # Stop when we have enough documents
                if len(documents) >= page_size:
                    break
            
            # For total count, return -1 (expensive to compute)
            total = -1
            
            return documents, total
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def search_documents(self, search_params: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """Search documents by various criteria"""
        try:
            query = self.documents_collection
            
            # Apply search filters
            if search_params.get('document_no'):
                query = query.where(filter=FieldFilter('metadata.document_no', '==', search_params['document_no']))
            if search_params.get('classification'):
                query = query.where(filter=FieldFilter('metadata.classification', '==', search_params['classification']))
            if search_params.get('branch_id'):
                query = query.where(filter=FieldFilter('metadata.branch_id', '==', search_params['branch_id']))
            if search_params.get('date_from'):
                query = query.where(filter=FieldFilter('metadata.document_date', '>=', search_params['date_from']))
            if search_params.get('date_to'):
                query = query.where(filter=FieldFilter('metadata.document_date', '<=', search_params['date_to']))
            if search_params.get('min_amount_usd'):
                query = query.where(filter=FieldFilter('metadata.invoice_amount_usd', '>=', str(search_params['min_amount_usd'])))
            if search_params.get('max_amount_usd'):
                query = query.where(filter=FieldFilter('metadata.invoice_amount_usd', '<=', str(search_params['max_amount_usd'])))
            if search_params.get('agentId'):
                query = query.where(filter=FieldFilter('agentId', '==', search_params['agentId']))
            if search_params.get('propertyId'):
                query = query.where(filter=FieldFilter('propertyId', '==', search_params['propertyId']))
            if search_params.get('clientId'):
                query = query.where(filter=FieldFilter('clientId', '==', search_params['clientId']))
            if search_params.get('dealId'):
                query = query.where(filter=FieldFilter('dealId', '==', search_params['dealId']))
            
            # Order by created_at descending
            query = query.order_by('created_at', direction=Query.DESCENDING)
            
            # Apply pagination directly (no need to fetch all for count)
            page = search_params.get('page', 1)
            page_size = search_params.get('page_size', 20)
            offset = (page - 1) * page_size
            docs = query.offset(offset).limit(page_size).stream()
            
            documents = []
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                documents.append(data)
            
            # For total count, return -1 to indicate it's not efficiently available
            # Callers can estimate based on pagination or maintain a separate counter
            total = -1  # Indicate count unavailable (more efficient than fetching all)
            
            return documents, total
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return [], 0
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document record"""
        try:
            doc_ref = self.documents_collection.document(document_id)
            doc_ref.delete()
            logger.info(f"Deleted document record: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document record: {e}")
            return False
    
    # Job Operations
    
    def create_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """Create a new processing job"""
        try:
            doc_ref = self.jobs_collection.document(job_id)
            data['status'] = 'pending'
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            data['processed_documents'] = 0
            data['failed_documents'] = 0
            doc_ref.set(data)
            logger.info(f"Created job record: {job_id}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to create job record: {e}")
            raise
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        try:
            doc_ref = self.jobs_collection.document(job_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['job_id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            return None
    
    def update_job(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Update a job record"""
        try:
            doc_ref = self.jobs_collection.document(job_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated job record: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update job record: {e}")
            return False
    
    def update_job_progress(self, job_id: str, processed: int = 0, failed: int = 0, status: Optional[str] = None) -> bool:
        """Update job progress"""
        try:
            doc_ref = self.jobs_collection.document(job_id)
            update_data = {'updated_at': firestore.SERVER_TIMESTAMP}

            doc_snapshot = None
            if processed > 0 or failed > 0:
                doc_snapshot = doc_ref.get()

            if processed > 0 and doc_snapshot and doc_snapshot.exists:
                current_data = doc_snapshot.to_dict()
                current_processed = current_data.get('processed_documents', 0)
                update_data['processed_documents'] = current_processed + processed

            if failed > 0 and doc_snapshot and doc_snapshot.exists:
                current_data = doc_snapshot.to_dict()
                current_failed = current_data.get('failed_documents', 0)
                update_data['failed_documents'] = current_failed + failed
            
            if status:
                update_data['status'] = status
                if status == 'completed' or status == 'failed':
                    update_data['completed_at'] = firestore.SERVER_TIMESTAMP
            
            doc_ref.update(update_data)
            return True
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")
            return False
    
    # Flow Operations
    
    def create_flow(self, flow_id: str, data: Dict[str, Any]) -> str:
        """Create a new flow"""
        try:
            doc_ref = self.flows_collection.document(flow_id)
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            data['document_count'] = 0
            doc_ref.set(data)
            logger.info(f"Created flow record: {flow_id}")
            return flow_id
        except Exception as e:
            logger.error(f"Failed to create flow record: {e}")
            raise
    
    def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get a flow by ID"""
        try:
            doc_ref = self.flows_collection.document(flow_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['flow_id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get flow: {e}")
            return None
    
    def list_flows(
        self,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List flows with efficient cursor-based pagination"""
        try:
            logger.info(f"🔍 Listing flows from Firestore: page={page}, page_size={page_size}, cursor={cursor_doc_id}")
            
            # Build query with ordering
            query = self.flows_collection.order_by('created_at', direction=Query.DESCENDING)
            
            # Use cursor-based pagination if cursor provided (more efficient than offset)
            if cursor_doc_id:
                cursor_doc = self.flows_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
                    logger.info(f"   Using cursor: {cursor_doc_id}")
            elif page > 1:
                # Fallback to offset for first-time pagination without cursor
                # This is less efficient but needed for page > 1 without cursor
                offset = (page - 1) * page_size
                query = query.offset(offset)
                logger.info(f"   Using offset: {offset}")
            
            # Fetch flows with limit
            docs = list(query.limit(page_size).stream())
            logger.info(f"   Found {len(docs)} flow documents in Firestore")
            
            flows = []
            for doc in docs:
                data = doc.to_dict()
                data['flow_id'] = doc.id
                logger.debug(f"   Flow: id={doc.id}, name={data.get('flow_name')}")
                flows.append(data)
            
            logger.info(f"✅ Returning {len(flows)} flows from Firestore")
            
            # For total count, return -1 (expensive to compute)
            total = -1
            
            return flows, total
        except Exception as e:
            logger.error(f"Failed to list flows: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def update_flow(self, flow_id: str, data: Dict[str, Any]) -> bool:
        """Update a flow record"""
        try:
            doc_ref = self.flows_collection.document(flow_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated flow record: {flow_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update flow record: {e}")
            return False
    
    def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow from Firestore"""
        try:
            flow_ref = self.flows_collection.document(flow_id)
            flow_ref.delete()
            logger.info(f"✅ Deleted flow {flow_id} from Firestore")
            return True
        except Exception as e:
            logger.error(f"Failed to delete flow {flow_id}: {e}")
            return False
    
    def delete_documents_by_flow_id(self, flow_id: str) -> int:
        """Delete all documents for a flow from Firestore"""
        try:
            # Query all documents for this flow
            query = self.documents_collection.where(filter=FieldFilter('flow_id', '==', flow_id))
            docs = list(query.stream())
            
            # Delete each document
            deleted_count = 0
            for doc in docs:
                doc.reference.delete()
                deleted_count += 1
            
            logger.info(f"✅ Deleted {deleted_count} documents for flow {flow_id} from Firestore")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to delete documents for flow {flow_id}: {e}")
            return 0
    
    def increment_flow_document_count(self, flow_id: str, increment: int = 1) -> bool:
        """Increment the document count for a flow"""
        try:
            doc_ref = self.flows_collection.document(flow_id)
            doc = doc_ref.get()
            if doc.exists:
                current_data = doc.to_dict()
                current_count = current_data.get('document_count', 0)
                doc_ref.update({
                    'document_count': current_count + increment,
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"Incremented document count for flow {flow_id} by {increment}")
                return True
            else:
                logger.warning(f"Flow {flow_id} not found for document count increment")
                return False
        except Exception as e:
            logger.error(f"Failed to increment flow document count: {e}")
            return False
    
    def update_flow_document_count(self, flow_id: str, count: int) -> bool:
        """Update the document count for a flow to a specific value"""
        try:
            doc_ref = self.flows_collection.document(flow_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_ref.update({
                    'document_count': count,
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
                logger.info(f"Updated document count for flow {flow_id} to {count}")
                return True
            else:
                logger.warning(f"Flow {flow_id} not found for document count update")
                return False
        except Exception as e:
            logger.error(f"Failed to update flow document count: {e}")
            return False
    
    def get_documents_by_flow_id(
        self,
        flow_id: str,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get documents by flow_id with efficient cursor-based pagination"""
        try:
            # Build query with filter and ordering
            query = self.documents_collection.where(filter=FieldFilter('flow_id', '==', flow_id))
            query = query.order_by('created_at', direction=Query.DESCENDING)
            
            # Use cursor-based pagination if cursor provided (more efficient than offset)
            if cursor_doc_id:
                cursor_doc = self.documents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                # Fallback to offset for first-time pagination without cursor
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            # Fetch documents with limit
            docs = list(query.limit(page_size).stream())
            
            documents = []
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                documents.append(data)
            
            # For total count, return -1 (expensive to compute)
            total = -1
            
            return documents, total
                    
        except Exception as e:
            logger.error(f"Failed to get documents by flow_id: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def get_documents_by_deal(
        self,
        deal_id: str,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get documents by deal_id with efficient cursor-based pagination"""
        try:
            # Build query with filter and ordering
            query = self.documents_collection.where(filter=FieldFilter('dealId', '==', deal_id))
            query = query.order_by('created_at', direction=Query.DESCENDING)
            
            # Use cursor-based pagination if cursor provided (more efficient than offset)
            if cursor_doc_id:
                cursor_doc = self.documents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                # Fallback to offset for first-time pagination without cursor
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            # Fetch documents with limit
            docs = list(query.limit(page_size).stream())
            
            documents = []
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                documents.append(data)
            
            # For total count, return -1 (expensive to compute)
            total = -1
            
            return documents, total
                    
        except Exception as e:
            logger.error(f"Failed to get documents by deal_id: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def get_all_organized_documents(
        self,
        page: int = 1,
        page_size: int = 50,
        category: Optional[str] = None,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int, Optional[str]]:
        """
        Get all organized (completed) documents with efficient cursor-based pagination.
        Optimized for BrowseFiles page - eliminates N+1 query problem.
        
        Returns:
            Tuple of (documents, total_count, next_cursor_doc_id)
        """
        try:
            from services.category_mapper import map_backend_to_ui_category
            from google.api_core import exceptions as gcp_exceptions
            
            logger.info(f"📊 get_all_organized_documents: page={page}, page_size={page_size}, category={category}")
            
            # Build query for completed documents with organized_path
            query = self.documents_collection.where(
                filter=FieldFilter('processing_status', '==', 'completed')
            )
            
            # Order by created_at descending for consistent pagination
            query = query.order_by('created_at', direction=Query.DESCENDING)
            
            # Use cursor-based pagination if cursor provided
            if cursor_doc_id:
                cursor_doc = self.documents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                # Fallback to offset for first-time pagination without cursor
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            # Fetch documents with limit
            try:
                docs = list(query.limit(page_size + 1).stream())  # Fetch one extra to check for more
            except gcp_exceptions.FailedPrecondition as index_error:
                # Index not ready yet - fallback to simpler query without processing_status filter
                if "index" in str(index_error).lower():
                    logger.warning(f"⚠️  Firestore index not ready for processing_status query. Using fallback query.")
                    logger.warning(f"   Index creation URL: {str(index_error).split('create_composite=')[-1] if 'create_composite=' in str(index_error) else 'N/A'}")
                    logger.warning(f"   Please create the index or wait for it to be built. Using fallback query in the meantime.")
                    
                    # Fallback: query all documents and filter in memory (less efficient but works)
                    query = self.documents_collection.order_by('created_at', direction=Query.DESCENDING)
                    if page > 1:
                        offset = (page - 1) * page_size
                        query = query.offset(offset)
                    
                    docs = list(query.limit(page_size * 2).stream())  # Fetch more to account for filtering
                else:
                    raise
            
            # Filter by processing_status if we used fallback query (index not ready)
            # When index is available, this filter is already applied in the query
            filtered_docs = []
            for doc in docs:
                data = doc.to_dict()
                # Only include completed documents (skip if using fallback query)
                if data.get('processing_status') != 'completed':
                    continue
                filtered_docs.append(doc)
            
            # Check if we have enough after filtering
            has_more = len(filtered_docs) > page_size
            if has_more:
                filtered_docs = filtered_docs[:page_size]  # Trim to requested size
            
            documents = []
            next_cursor = None
            
            for doc in filtered_docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                
                # Get or compute ui_category
                metadata = data.get('metadata', {})
                ui_category = metadata.get('ui_category')
                if not ui_category:
                    classification = metadata.get('classification') or data.get('document_type') or data.get('classification')
                    ui_category = map_backend_to_ui_category(classification) if classification else 'SPA'
                    metadata['ui_category'] = ui_category
                    data['metadata'] = metadata
                
                # Apply category filter if specified
                if category and ui_category.lower() != category.lower():
                    continue
                
                documents.append(data)
            
            # Set next cursor for pagination
            if has_more and filtered_docs:
                next_cursor = filtered_docs[-1].id
            
            # For total count, return -1 (expensive to compute)
            # Frontend should use infinite scroll or "load more" pattern
            total = -1
            
            logger.info(f"✅ Retrieved {len(documents)} organized documents")
            return documents, total, next_cursor
            
        except Exception as e:
            logger.error(f"Failed to get all organized documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0, None
    
    def get_category_statistics(self) -> Dict[str, int]:
        """Get document count by UI category using paginated queries"""
        try:
            from services.category_mapper import map_backend_to_ui_category
            
            category_counts: Dict[str, int] = {}
            total = 0
            page_size = 100  # Process in batches
            last_doc = None
            
            # Use pagination to process documents in batches
            while True:
                query = self.documents_collection.order_by('created_at', direction=Query.DESCENDING)
                
                if last_doc:
                    # Use cursor-based pagination for better performance
                    query = query.start_after(last_doc)
                
                docs = list(query.limit(page_size).stream())
                
                if not docs:
                    break
                
                for doc in docs:
                    total += 1
                    data = doc.to_dict()
                    metadata = data.get('metadata', {})
                    ui_category = metadata.get('ui_category')
                    
                    if not ui_category:
                        # Compute from classification if not set
                        classification = metadata.get('classification') or data.get('document_type') or data.get('classification')
                        ui_category = map_backend_to_ui_category(classification) if classification else 'Unknown'
                    
                    category_counts[ui_category] = category_counts.get(ui_category, 0) + 1
                
                # Update cursor for next batch
                last_doc = docs[-1]
                
                # Safety limit: stop after processing 10,000 documents to avoid timeout
                if total >= 10000:
                    logger.warning("Category statistics limited to first 10,000 documents")
                    break
            
            # Ensure all categories are present (even with 0 count)
            category_counts['total'] = total
            return category_counts
        except Exception as e:
            logger.error(f"Failed to get category statistics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'total': 0}
    
    # Client Operations
    
    def create_client(self, client_id: str, data: Dict[str, Any]) -> str:
        """Create a new client record"""
        try:
            doc_ref = self.clients_collection.document(client_id)
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            data.setdefault('property_file_count', 0)
            data.setdefault('created_from', 'manual')
            doc_ref.set(data)
            logger.info(f"Created client record: {client_id}")
            return client_id
        except Exception as e:
            logger.error(f"Failed to create client record: {e}")
            raise
    
    def get_client(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get a client by ID"""
        try:
            doc_ref = self.clients_collection.document(client_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get client: {e}")
            return None
    
    def list_clients(
        self,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List clients with pagination"""
        try:
            base_query = self.clients_collection.order_by('created_at', direction=Query.DESCENDING)

            total = _get_query_count(base_query)

            query = base_query
            
            if cursor_doc_id:
                cursor_doc = self.clients_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())
            clients = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                clients.append(data)
            
            return clients, total
        except Exception as e:
            logger.error(f"Failed to list clients: {e}")
            return [], 0
    
    def update_client(self, client_id: str, data: Dict[str, Any]) -> bool:
        """Update a client record"""
        try:
            doc_ref = self.clients_collection.document(client_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated client record: {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update client record: {e}")
            return False
    
    def delete_client(self, client_id: str) -> bool:
        """Delete a client record"""
        try:
            doc_ref = self.clients_collection.document(client_id)
            doc_ref.delete()
            logger.info(f"Deleted client record: {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete client record: {e}")
            return False
    
    def search_clients_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search clients by name (fuzzy matching)"""
        try:
            # Normalize search term
            search_term = name.lower().strip()
            
            # Get all clients and filter in memory (Firestore doesn't support case-insensitive search)
            docs = list(self.clients_collection.stream())
            matches = []
            
            for doc in docs:
                data = doc.to_dict()
                full_name = data.get('full_name', '').lower().strip()
                
                # Simple fuzzy matching: check if search term is in name or vice versa
                if search_term in full_name or full_name in search_term:
                    data['id'] = doc.id
                    matches.append(data)
            
            return matches
        except Exception as e:
            logger.error(f"Failed to search clients by name: {e}")
            return []
    
    # Property Operations
    
    def create_property(self, property_id: str, data: Dict[str, Any]) -> str:
        """Create a new property record"""
        try:
            doc_ref = self.properties_collection.document(property_id)
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            logger.info(f"Created property record: {property_id}")
            return property_id
        except Exception as e:
            logger.error(f"Failed to create property record: {e}")
            raise
    
    def get_property(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Get a property by ID"""
        try:
            doc_ref = self.properties_collection.document(property_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get property: {e}")
            return None
    
    def update_property(self, property_id: str, data: Dict[str, Any]) -> bool:
        """Update a property record"""
        try:
            doc_ref = self.properties_collection.document(property_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated property record: {property_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update property record: {e}")
            return False
    
    def list_properties(
        self,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List properties with pagination and optional agent filter"""
        try:
            if agent_id:
                base_query = self.properties_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).order_by('created_at', direction=Query.DESCENDING)
            else:
                base_query = self.properties_collection.order_by('created_at', direction=Query.DESCENDING)

            total = _get_query_count(base_query)

            query = base_query
            
            if cursor_doc_id:
                cursor_doc = self.properties_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())
            properties = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                properties.append(data)
            
            return properties, total
        except Exception as e:
            logger.error(f"Failed to list properties: {e}")
            return [], 0
    
    def search_properties_by_reference(self, reference: str) -> List[Dict[str, Any]]:
        """Search properties by reference"""
        try:
            # Normalize search term
            search_term = reference.lower().strip()
            
            # Get all properties and filter in memory
            docs = list(self.properties_collection.stream())
            matches = []
            
            for doc in docs:
                data = doc.to_dict()
                prop_reference = data.get('reference', '').lower().strip()
                
                if search_term in prop_reference or prop_reference in search_term:
                    data['id'] = doc.id
                    matches.append(data)
            
            return matches
        except Exception as e:
            logger.error(f"Failed to search properties by reference: {e}")
            return []
    
    # Property File Operations
    
    def create_property_file(self, property_file_id: str, data: Dict[str, Any]) -> str:
        """Create a new property file record"""
        try:
            doc_ref = self.property_files_collection.document(property_file_id)
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            data.setdefault('status', 'INCOMPLETE')
            doc_ref.set(data)
            logger.info(f"Created property file record: {property_file_id}")
            return property_file_id
        except Exception as e:
            logger.error(f"Failed to create property file record: {e}")
            raise
    
    def get_property_file(self, property_file_id: str) -> Optional[Dict[str, Any]]:
        """Get a property file by ID"""
        try:
            doc_ref = self.property_files_collection.document(property_file_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get property file: {e}")
            return None
    
    def list_property_files(
        self,
        page: int = 1,
        page_size: int = 20,
        client_id: Optional[str] = None,
        property_id: Optional[str] = None,
        status: Optional[str] = None,
        transaction_type: Optional[str] = None,
        deal_id: Optional[str] = None,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List property files with filters and pagination"""
        try:
            base_query = self.property_files_collection

            # Apply filters
            if client_id:
                base_query = base_query.where(filter=FieldFilter('client_id', '==', client_id))
            if property_id:
                base_query = base_query.where(filter=FieldFilter('property_id', '==', property_id))
            if deal_id:
                base_query = base_query.where(filter=FieldFilter('dealId', '==', deal_id))
            if status:
                # Normalize status to uppercase to ensure case-insensitive matching
                status_normalized = status.upper() if isinstance(status, str) else status
                logger.info(f"Filtering property files by status: {status_normalized}")
                base_query = base_query.where(filter=FieldFilter('status', '==', status_normalized))
            if transaction_type:
                base_query = base_query.where(filter=FieldFilter('transaction_type', '==', transaction_type))

            base_query = base_query.order_by('created_at', direction=Query.DESCENDING)

            total = _get_query_count(base_query)

            query = base_query

            if cursor_doc_id:
                cursor_doc = self.property_files_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())

            property_ids = {doc.to_dict().get('property_id') for doc in docs if doc.to_dict().get('property_id')}
            properties_map = {
                prop['id']: prop
                for prop in self._fetch_documents_by_ids(self.properties_collection, property_ids)
            }

            property_files = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id

                # Log status for debugging
                if status:
                    logger.debug(f"Property file {data['id']} has status: {data.get('status')}")

                # Enrich with property name/title if property_id exists
                property_id = data.get('property_id')
                property_obj = properties_map.get(property_id) if property_id else None
                if property_obj:
                    # Prefer title, then name, then reference
                    data['property_name'] = property_obj.get('title') or property_obj.get('name') or property_obj.get('reference') or data.get('property_reference')
                else:
                    # No property_id or failed fetch, use property_reference as fallback
                    data['property_name'] = data.get('property_reference')

                property_files.append(data)

            logger.info(f"Returning {len(property_files)} property files (filtered by status: {status})")
            return property_files, total
        except Exception as e:
            logger.error(f"Failed to list property files: {e}")
            return [], 0
    
    def update_property_file(self, property_file_id: str, data: Dict[str, Any]) -> bool:
        """Update a property file record"""
        try:
            doc_ref = self.property_files_collection.document(property_file_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated property file record: {property_file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update property file record: {e}")
            return False
    
    def delete_property_file(self, property_file_id: str) -> bool:
        """Delete a property file record"""
        try:
            doc_ref = self.property_files_collection.document(property_file_id)
            doc_ref.delete()
            logger.info(f"Deleted property file record: {property_file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete property file record: {e}")
            return False
    
    def get_property_files_by_client(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all property files for a client"""
        try:
            query = self.property_files_collection.where(
                filter=FieldFilter('client_id', '==', client_id)
            ).order_by('created_at', direction=Query.DESCENDING)

            docs = list(query.stream())

            property_ids = {doc.to_dict().get('property_id') for doc in docs if doc.to_dict().get('property_id')}
            properties_map = {
                prop['id']: prop
                for prop in self._fetch_documents_by_ids(self.properties_collection, property_ids)
            }

            property_files = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id

                property_obj = properties_map.get(data.get('property_id'))
                if property_obj:
                    data['property_name'] = property_obj.get('title') or property_obj.get('name') or property_obj.get('reference') or data.get('property_reference')
                else:
                    data['property_name'] = data.get('property_reference')

                property_files.append(data)

            return property_files
        except Exception as e:
            logger.error(f"Failed to get property files by client: {e}")
            return []
    
    def get_property_files_by_property(self, property_id: str) -> List[Dict[str, Any]]:
        """Get all property files for a property"""
        try:
            query = self.property_files_collection.where(
                filter=FieldFilter('property_id', '==', property_id)
            ).order_by('created_at', direction=Query.DESCENDING)
            
            docs = list(query.stream())
            property_files = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                property_files.append(data)
            
            return property_files
        except Exception as e:
            logger.error(f"Failed to get property files by property: {e}")
            return []
    
    def get_property_files_by_deal(self, deal_id: str) -> List[Dict[str, Any]]:
        """Get all property files for a deal"""
        try:
            query = self.property_files_collection.where(
                filter=FieldFilter('dealId', '==', deal_id)
            ).order_by('created_at', direction=Query.DESCENDING)
            
            docs = list(query.stream())
            property_files = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                property_files.append(data)
            
            return property_files
        except Exception as e:
            logger.error(f"Failed to get property files by deal: {e}")
            return []
    
    def find_matching_property_file(
        self,
        client_full_name: str,
        property_reference: Optional[str] = None,
        transaction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find matching property files using fuzzy matching with improved logic"""
        try:
            from services.matching_service import MatchingService
            
            # Normalize search terms using matching service
            normalized_name = MatchingService.normalize_name(client_full_name)
            
            # Get all property files and filter in memory
            query = self.property_files_collection
            if transaction_type:
                query = query.where(filter=FieldFilter('transaction_type', '==', transaction_type))
            
            docs = list(query.stream())
            matches = []
            
            for doc in docs:
                data = doc.to_dict()
                file_name = data.get('client_full_name', '')
                file_reference = data.get('property_reference', '') if data.get('property_reference') else ''
                file_transaction_type = data.get('transaction_type')
                
                # Normalize file name for comparison
                normalized_file_name = MatchingService.normalize_name(file_name)
                
                # Use similarity score for name matching (more accurate than substring)
                name_score = MatchingService.similarity_score(normalized_name, normalized_file_name)
                name_match = name_score >= 0.75  # 75% similarity threshold
                
                # Check property reference match if provided
                # If property_reference is None, match by client name only (allow matching)
                property_match = True
                if property_reference:
                    normalized_ref = property_reference.lower().strip()
                    file_ref_normalized = file_reference.lower().strip() if file_reference else ''
                    # Use similarity for property reference
                    if file_ref_normalized:
                        ref_score = MatchingService.similarity_score(normalized_ref, file_ref_normalized)
                        property_match = ref_score >= 0.8  # Higher threshold for property reference
                    else:
                        # If property file has no reference but we're searching with one, allow match
                        # (property reference might be added later or might be optional)
                        property_match = True
                # If property_reference is None, match by name only (property_match = True already set)
                # This allows matching documents even when property reference is missing
                
                # Check transaction type match if specified
                transaction_match = True
                if transaction_type and file_transaction_type:
                    transaction_match = transaction_type.upper() == file_transaction_type.upper()
                
                if name_match and property_match and transaction_match:
                    data['id'] = doc.id
                    data['match_confidence'] = name_score  # Store confidence score
                    matches.append(data)
            
            # Sort by confidence score (highest first)
            matches.sort(key=lambda x: x.get('match_confidence', 0.0), reverse=True)
            
            return matches
        except Exception as e:
            logger.error(f"Failed to find matching property file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def find_or_create_property_file(
        self,
        client_full_name: str,
        property_reference: Optional[str] = None,
        transaction_type: Optional[str] = None,
        client_id: Optional[str] = None,
        property_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find existing property file or create new one.
        Returns the property file dict with 'id' field.
        """
        import uuid
        
        # Try to find existing property file
        existing_files = self.find_matching_property_file(
            client_full_name=client_full_name,
            property_reference=property_reference,
            transaction_type=transaction_type
        )
        
        if existing_files:
            # Return the best match (highest confidence)
            return existing_files[0]
        
        # No existing file found - create new one
        property_file_id = str(uuid.uuid4())
        property_file_data = {
            'id': property_file_id,
            'client_id': client_id,
            'client_full_name': client_full_name,
            'property_id': property_id,
            'property_reference': property_reference,
            'transaction_type': transaction_type or 'BUY',
            'status': 'INCOMPLETE'
        }
        
        self.create_property_file(property_file_id, property_file_data)
        property_file_data['id'] = property_file_id
        
        logger.info(f"Created new property file {property_file_id} for client {client_full_name}")
        return property_file_data
    
    # Agent-based query methods
    
    def list_documents_by_agent(
        self,
        agent_id: str,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List all documents uploaded by a specific agent"""
        try:
            query = self.documents_collection.where(
                filter=FieldFilter('agentId', '==', agent_id)
            ).order_by('created_at', direction=Query.DESCENDING)
            
            if cursor_doc_id:
                cursor_doc = self.documents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())
            documents = []
            for doc in docs:
                data = doc.to_dict()
                data['document_id'] = doc.id
                documents.append(data)
            
            total = -1
            return documents, total
        except Exception as e:
            logger.error(f"Failed to list documents by agent: {e}")
            return [], 0
    
    def list_properties_by_agent(
        self,
        agent_id: str,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None,
        use_deals: bool = False
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List all properties managed by a specific agent.
        If use_deals=True, uses deals collection for more accurate results (properties with active deals).
        Otherwise, uses properties collection directly (properties where agentId is set).
        """
        try:
            if use_deals:
                # Use deals collection to find properties with active deals for this agent
                deals = self.get_deals_by_agent(agent_id)
                property_ids = {deal.get('propertyId') for deal in deals if deal.get('propertyId')}

                if not property_ids:
                    return [], 0

                # Fetch properties by IDs using chunked IN queries
                properties = self._fetch_documents_by_ids(self.properties_collection, property_ids)

                # Sort by created_at descending
                properties.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                
                # Apply pagination
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                paginated_properties = properties[start_idx:end_idx]
                
                total = len(properties)
                return paginated_properties, total
            else:
                # Original method: query properties collection directly
                base_query = self.properties_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).order_by('created_at', direction=Query.DESCENDING)

                total = _get_query_count(base_query)

                query = base_query
                
                if cursor_doc_id:
                    cursor_doc = self.properties_collection.document(cursor_doc_id).get()
                    if cursor_doc.exists:
                        query = query.start_after(cursor_doc)
                elif page > 1:
                    offset = (page - 1) * page_size
                    query = query.offset(offset)
                
                docs = list(query.limit(page_size).stream())
                properties = []
                for doc in docs:
                    data = doc.to_dict()
                    data['id'] = doc.id
                    properties.append(data)
                
                return properties, total
        except Exception as e:
            logger.error(f"Failed to list properties by agent: {e}")
            return [], 0
    
    def list_clients_by_agent(
        self,
        agent_id: str,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List all clients related to deals for a specific agent (optimized using deals collection)"""
        try:
            # Use deals collection for efficient querying
            deals = self.get_deals_by_agent(agent_id)

            # Extract unique client IDs from deals
            client_ids = {deal.get('clientId') for deal in deals if deal.get('clientId')}

            if not client_ids:
                return [], 0

            # Get clients by IDs using chunked IN queries
            clients = self._fetch_documents_by_ids(self.clients_collection, client_ids)

            # Sort by created_at descending
            clients.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_clients = clients[start_idx:end_idx]
            
            total = len(clients)
            return paginated_clients, total
        except Exception as e:
            logger.error(f"Failed to list clients by agent: {e}")
            return [], 0
    
    def get_client_agent(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get the agent who uploaded the most documents for a client"""
        try:
            # Get all documents for this client
            client_docs_query = self.documents_collection.where(
                filter=FieldFilter('clientId', '==', client_id)
            )
            client_docs = list(client_docs_query.stream())
            
            if not client_docs:
                return None
            
            # Count documents by agentId
            agent_counts = {}
            agent_timestamps = {}
            for doc in client_docs:
                data = doc.to_dict()
                agent_id = data.get('agentId')
                if agent_id:
                    agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
                    # Track most recent document timestamp for tie-breaking
                    created_at = data.get('created_at')
                    if created_at:
                        if agent_id not in agent_timestamps or created_at > agent_timestamps[agent_id]:
                            agent_timestamps[agent_id] = created_at
            
            if not agent_counts:
                return None
            
            # Find agent with highest count
            max_count = max(agent_counts.values())
            top_agents = [agent_id for agent_id, count in agent_counts.items() if count == max_count]
            
            # If multiple agents have same count, use most recent document
            if len(top_agents) > 1:
                top_agent = max(top_agents, key=lambda agent_id: agent_timestamps.get(agent_id, datetime.min))
            else:
                top_agent = top_agents[0]
            
            return {
                "id": top_agent,
                "document_count": max_count
            }
        except Exception as e:
            logger.error(f"Failed to get client agent: {e}")
            return None
    
    def get_client_properties(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all unique properties for a client through deals (optimized)"""
        try:
            # Use deals collection for efficient querying
            deals = self.get_deals_by_client(client_id)
            
            # Extract unique property IDs from deals
            property_ids = set()
            for deal in deals:
                property_id = deal.get('propertyId')
                if property_id:
                    property_ids.add(property_id)
            
            if not property_ids:
                # Fallback to property files if no deals found
                property_files = self.get_property_files_by_client(client_id)
                logger.info(f"Found {len(property_files)} property files by client_id for client {client_id}")
                
                # If no property files found by client_id, try searching by client_full_name as fallback
                if not property_files:
                    client = self.get_client(client_id)
                    if client and client.get('full_name'):
                        # Search property files by client_full_name
                        try:
                            query = self.property_files_collection.where(
                                filter=FieldFilter('client_full_name', '==', client.get('full_name'))
                            ).order_by('created_at', direction=Query.DESCENDING)
                            docs = list(query.stream())
                            property_files = []
                            for doc in docs:
                                data = doc.to_dict()
                                data['id'] = doc.id
                                property_files.append(data)
                            logger.info(f"Found {len(property_files)} property files by client_full_name '{client.get('full_name')}' for client {client_id}")
                        except Exception as e:
                            logger.warning(f"Failed to search property files by client_full_name: {e}")
                
                if not property_files:
                    logger.info(f"No property files found for client {client_id}")
                    return []
                
                # Extract property IDs from property files
                for pf in property_files:
                    prop_id = pf.get('property_id')
                    if prop_id:
                        property_ids.add(prop_id)
            
            if not property_ids:
                return []
            
            # Track property files with their IDs and references for fallback lookup
            property_file_lookups = []  # List of (property_id, property_reference) tuples
            property_references_only = set()  # References from files without property_id
            
            for pf in property_files:
                property_id = pf.get('property_id')
                property_reference = pf.get('property_reference')
                
                if property_id:
                    # Store both ID and reference for fallback
                    property_file_lookups.append((property_id, property_reference))
                elif property_reference:
                    # Only reference available
                    property_references_only.add(property_reference)
            
            # Fetch properties by ID first
            properties = []
            property_ids_found = set()
            property_references_handled = set()
            
            # Try to fetch properties by ID
            for property_id, property_reference in property_file_lookups:
                property_obj = self.get_property(property_id)
                if property_obj:
                    properties.append(property_obj)
                    property_ids_found.add(property_id)
                    logger.info(f"Found property {property_id} by ID for client {client_id}")
                    if property_reference:
                        property_references_handled.add(property_reference)
                elif property_reference:
                    # Property not found by ID, try searching by reference as fallback
                    logger.info(f"Property {property_id} not found by ID, will search by reference '{property_reference}' for client {client_id}")
                    property_references_only.add(property_reference)
            
            # Search properties by reference for those without property_id or where property_id lookup failed
            for property_reference in property_references_only:
                if property_reference:
                    matching_properties = self.search_properties_by_reference(property_reference)
                    # Normalize the search reference for comparison
                    search_ref_normalized = property_reference.lower().strip()
                    
                    # First, try to find exact matches
                    exact_match_found = False
                    for prop in matching_properties:
                        prop_id = prop.get('id')
                        if prop_id and prop_id not in property_ids_found:
                            prop_ref = prop.get('reference', '').strip()
                            prop_ref_normalized = prop_ref.lower()
                            
                            # Prefer exact match
                            if prop_ref_normalized == search_ref_normalized:
                                properties.append(prop)
                                property_ids_found.add(prop_id)
                                logger.info(f"Found property {prop_id} by exact reference match '{property_reference}' for client {client_id}")
                                exact_match_found = True
                                break
                    
                    # If no exact match, add the first close match
                    if not exact_match_found:
                        for prop in matching_properties:
                            prop_id = prop.get('id')
                            if prop_id and prop_id not in property_ids_found:
                                prop_ref = prop.get('reference', '').strip()
                                prop_ref_normalized = prop_ref.lower()
                                
                                # Check for close match (reference contains search term or vice versa)
                                if search_ref_normalized in prop_ref_normalized or \
                                   prop_ref_normalized in search_ref_normalized:
                                    properties.append(prop)
                                    property_ids_found.add(prop_id)
                                    logger.info(f"Found property {prop_id} by close reference match '{property_reference}' -> '{prop_ref}' for client {client_id}")
                                    break
            
            logger.info(f"Returning {len(properties)} properties for client {client_id}")
            return properties
        except Exception as e:
            logger.error(f"Failed to get client properties for client {client_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def list_clients_with_relations(
        self,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List clients with their agents and properties in optimized batch query"""
        try:
            # Get clients (with or without agent filter)
            if agent_id:
                clients, total = self.list_clients_by_agent(agent_id, page, page_size, cursor_doc_id)
            else:
                clients, total = self.list_clients(page, page_size, cursor_doc_id)
            
            if not clients:
                return [], total
            
            client_ids = [c['id'] for c in clients]
            
            # Batch fetch all property files for these clients (avoid N queries)
            # Fetch all property files and filter in memory to avoid index requirements
            try:
                all_property_files = list(self.property_files_collection.stream())
                # Group property files by client_id
                property_files_by_client = {}
                for pf_doc in all_property_files:
                    pf_data = pf_doc.to_dict()
                    pf_client_id = pf_data.get('client_id')
                    if pf_client_id and pf_client_id in client_ids:
                        if pf_client_id not in property_files_by_client:
                            property_files_by_client[pf_client_id] = []
                        pf_data['id'] = pf_doc.id
                        property_files_by_client[pf_client_id].append(pf_data)
            except Exception as e:
                logger.warning(f"Error fetching property files in batch: {e}")
                property_files_by_client = {}
            
            # Extract unique properties from property files for each client
            client_properties_map = {}
            for client_id in client_ids:
                properties = []
                property_ids_seen = set()
                
                if client_id in property_files_by_client:
                    for pf in property_files_by_client[client_id]:
                        property_id = pf.get('property_id')
                        if property_id and property_id not in property_ids_seen:
                            # Fetch property details
                            property_data = self.get_property(property_id)
                            if property_data:
                                properties.append(property_data)
                                property_ids_seen.add(property_id)
                
                if properties:
                    client_properties_map[client_id] = properties
            
            # Get all documents to batch-process agent relationships
            all_docs = []
            if client_ids:
                # Query documents for all clients
                for client_id in client_ids:
                    try:
                        client_docs_query = self.documents_collection.where(
                            filter=FieldFilter('clientId', '==', client_id)
                        )
                        client_docs = list(client_docs_query.stream())
                        all_docs.extend([(client_id, doc) for doc in client_docs])
                    except Exception as e:
                        logger.warning(f"Error fetching documents for client {client_id}: {e}")
            
            # Process agent relationships for all clients
            client_agent_map = {}
            for client_id, doc in all_docs:
                data = doc.to_dict()
                agent_id = data.get('agentId')
                if agent_id:
                    if client_id not in client_agent_map:
                        client_agent_map[client_id] = {'id': agent_id, 'count': 0}
                    client_agent_map[client_id]['count'] += 1
            
            # Attach relations to clients
            for client in clients:
                client_id = client['id']
                # Attach agent info (just the ID, frontend will fetch full details if needed)
                if client_id in client_agent_map:
                    client['agent'] = {'id': client_agent_map[client_id]['id']}
                else:
                    client['agent'] = None
                # Attach properties
                client['properties'] = client_properties_map.get(client_id, [])
            
            return clients, total
        except Exception as e:
            logger.error(f"Failed to list clients with relations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], 0
    
    def get_client_full(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client with all related data (agent, properties, property_files) in optimized queries"""
        try:
            client = self.get_client(client_id)
            if not client:
                return None
            
            # Get agent
            agent_info = self.get_client_agent(client_id)
            if agent_info:
                client['agent'] = {'id': agent_info['id']}
            else:
                client['agent'] = None
            
            # Get properties
            properties = self.get_client_properties(client_id)
            client['properties'] = properties
            
            # Get property files
            property_files = self.get_property_files_by_client(client_id)
            client['property_files'] = property_files
            
            return client
        except Exception as e:
            logger.error(f"Failed to get client full: {e}")
            return None
    
    # Agent Operations
    
    def create_agent(self, agent_id: str, data: Dict[str, Any]) -> str:
        """Create a new agent record"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            data['id'] = agent_id
            data['created_at'] = firestore.SERVER_TIMESTAMP
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            data.setdefault('status', 'ACTIVE')
            doc_ref.set(data)
            logger.info(f"Created agent record: {agent_id}")
            return agent_id
        except Exception as e:
            logger.error(f"Failed to create agent record: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent by ID"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get agent: {e}")
            return None
    
    def list_agents(
        self,
        page: int = 1,
        page_size: int = 20,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List agents with pagination"""
        try:
            base_query = self.agents_collection.order_by('created_at', direction=Query.DESCENDING)

            total = _get_query_count(base_query)

            query = base_query
            
            if cursor_doc_id:
                cursor_doc = self.agents_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())
            agents = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                agents.append(data)
            
            return agents, total
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return [], 0
    
    def update_agent(self, agent_id: str, data: Dict[str, Any]) -> bool:
        """Update an agent record"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated agent record: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update agent record: {e}")
            return False
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent record"""
        try:
            doc_ref = self.agents_collection.document(agent_id)
            doc_ref.delete()
            logger.info(f"Deleted agent record: {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete agent record: {e}")
            return False
    
    # Deal Operations
    
    def create_deal(self, deal_id: str, data: Dict[str, Any]) -> str:
        """Create a new deal record"""
        try:
            doc_ref = self.deals_collection.document(deal_id)
            data['id'] = deal_id
            data['createdAt'] = firestore.SERVER_TIMESTAMP
            data['updatedAt'] = firestore.SERVER_TIMESTAMP
            data.setdefault('status', 'ACTIVE')
            data.setdefault('stage', 'LEAD')
            doc_ref.set(data)
            logger.info(f"Created deal record: {deal_id}")
            return deal_id
        except Exception as e:
            logger.error(f"Failed to create deal record: {e}")
            raise
    
    def get_deal(self, deal_id: str) -> Optional[Dict[str, Any]]:
        """Get a deal by ID"""
        try:
            doc_ref = self.deals_collection.document(deal_id)
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get deal: {e}")
            return None
    
    def list_deals(
        self,
        page: int = 1,
        page_size: int = 20,
        agent_id: Optional[str] = None,
        client_id: Optional[str] = None,
        property_id: Optional[str] = None,
        status: Optional[str] = None,
        cursor_doc_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List deals with filters and pagination"""
        try:
            query = self.deals_collection
            
            # Apply filters
            if agent_id:
                query = query.where(filter=FieldFilter('agentId', '==', agent_id))
            if client_id:
                query = query.where(filter=FieldFilter('clientId', '==', client_id))
            if property_id:
                query = query.where(filter=FieldFilter('propertyId', '==', property_id))
            if status:
                query = query.where(filter=FieldFilter('status', '==', status))
            
            query = query.order_by('createdAt', direction=Query.DESCENDING)
            
            if cursor_doc_id:
                cursor_doc = self.deals_collection.document(cursor_doc_id).get()
                if cursor_doc.exists:
                    query = query.start_after(cursor_doc)
            elif page > 1:
                offset = (page - 1) * page_size
                query = query.offset(offset)
            
            docs = list(query.limit(page_size).stream())
            deals = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                deals.append(data)
            
            total = -1
            return deals, total
        except Exception as e:
            logger.error(f"Failed to list deals: {e}")
            return [], 0
    
    def update_deal(self, deal_id: str, data: Dict[str, Any]) -> bool:
        """Update a deal record"""
        try:
            doc_ref = self.deals_collection.document(deal_id)
            data['updatedAt'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            logger.info(f"Updated deal record: {deal_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update deal record: {e}")
            return False
    
    def delete_deal(self, deal_id: str) -> bool:
        """Delete a deal record"""
        try:
            doc_ref = self.deals_collection.document(deal_id)
            doc_ref.delete()
            logger.info(f"Deleted deal record: {deal_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete deal record: {e}")
            return False
    
    def get_deals_by_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all deals for a specific agent"""
        try:
            query = self.deals_collection.where(
                filter=FieldFilter('agentId', '==', agent_id)
            ).order_by('createdAt', direction=Query.DESCENDING)
            
            docs = list(query.stream())
            deals = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                deals.append(data)
            
            return deals
        except Exception as e:
            logger.error(f"Failed to get deals by agent: {e}")
            return []
    
    def get_deals_by_client(self, client_id: str) -> List[Dict[str, Any]]:
        """Get all deals for a specific client"""
        try:
            query = self.deals_collection.where(
                filter=FieldFilter('clientId', '==', client_id)
            ).order_by('createdAt', direction=Query.DESCENDING)
            
            docs = list(query.stream())
            deals = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                deals.append(data)
            
            return deals
        except Exception as e:
            logger.error(f"Failed to get deals by client: {e}")
            return []
    
    def get_deals_by_property(self, property_id: str) -> List[Dict[str, Any]]:
        """Get all deals for a specific property"""
        try:
            query = self.deals_collection.where(
                filter=FieldFilter('propertyId', '==', property_id)
            ).order_by('createdAt', direction=Query.DESCENDING)
            
            docs = list(query.stream())
            deals = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                deals.append(data)
            
            return deals
        except Exception as e:
            logger.error(f"Failed to get deals by property: {e}")
            return []
    
    def find_or_create_deal(
        self,
        agent_id: str,
        client_id: str,
        property_id: str,
        deal_type: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Find existing deal or create a new one for the given agent, client, and property combination"""
        try:
            # First, try to find existing deal
            query = self.deals_collection.where(
                filter=FieldFilter('agentId', '==', agent_id)
            ).where(
                filter=FieldFilter('clientId', '==', client_id)
            ).where(
                filter=FieldFilter('propertyId', '==', property_id)
            ).where(
                filter=FieldFilter('status', '==', 'ACTIVE')
            ).limit(1)
            
            existing_deals = list(query.stream())
            if existing_deals:
                deal_doc = existing_deals[0]
                data = deal_doc.to_dict()
                data['id'] = deal_doc.id
                logger.info(f"Found existing deal: {deal_doc.id}")
                return data
            
            # Create new deal if not found
            import uuid
            deal_id = f"deal_{uuid.uuid4().hex[:12]}"
            deal_data = {
                'agentId': agent_id,
                'clientId': client_id,
                'propertyId': property_id,
                'dealType': deal_type or 'RENT',
                'stage': stage or 'LEAD',
                'status': 'ACTIVE'
            }
            
            self.create_deal(deal_id, deal_data)
            logger.info(f"Created new deal: {deal_id}")
            return self.get_deal(deal_id)
        except Exception as e:
            logger.error(f"Failed to find or create deal: {e}")
            return None

