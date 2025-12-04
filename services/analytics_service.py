"""
Analytics service for aggregating statistics and analytics data
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
from google.cloud import firestore
from google.cloud.firestore import Query, FieldFilter

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from services.firestore_service import FirestoreService

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analytics and statistics aggregation"""
    
    def __init__(self, firestore_service: FirestoreService):
        """Initialize analytics service with Firestore service"""
        self.firestore_service = firestore_service
        self.db = firestore_service.db
        self.documents_collection = firestore_service.documents_collection
        self.clients_collection = firestore_service.clients_collection
        self.properties_collection = firestore_service.properties_collection
        self.property_files_collection = firestore_service.property_files_collection
    
    def _get_document_type(self, doc_data: Dict[str, Any]) -> str:
        """Extract document type from document data"""
        metadata = doc_data.get('metadata', {})
        classification = metadata.get('classification') or doc_data.get('documentType') or metadata.get('ui_category')
        
        if not classification:
            return "Other"
        
        classification_lower = classification.lower()
        
        # Map to standard categories
        if 'id' in classification_lower or 'passport' in classification_lower or 'emirates id' in classification_lower:
            return "IDs"
        elif 'invoice' in classification_lower:
            return "Invoices"
        elif 'proof of payment' in classification_lower or 'payment' in classification_lower or 'receipt' in classification_lower:
            return "Proof of Payment"
        elif 'spa' in classification_lower or 'contract' in classification_lower:
            return "SPA"
        else:
            return "Other"
    
    def _get_processing_status(self, doc_data: Dict[str, Any]) -> str:
        """Get normalized processing status"""
        status = doc_data.get('processing_status', 'pending')
        
        # Map statuses
        if status in ['pending', 'uploading', 'processing']:
            return 'pending'
        elif status == 'need_review':
            return 'need_review'
        elif status == 'failed':
            return 'failed'
        elif status == 'completed':
            return 'completed'
        else:
            return 'pending'
    
    def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get analytics for a specific agent"""
        try:
            # Get all documents for this agent
            query = self.documents_collection.where(
                filter=FieldFilter('agentId', '==', agent_id)
            )
            docs = list(query.stream())
            
            # Get properties for this agent
            properties_query = self.properties_collection.where(
                filter=FieldFilter('agentId', '==', agent_id)
            )
            properties = list(properties_query.stream())
            
            # Get clients (from documents)
            client_ids = set()
            for doc in docs:
                data = doc.to_dict()
                client_id = data.get('clientId')
                if client_id:
                    client_ids.add(client_id)
            
            # Initialize counters
            document_count = len(docs)
            property_count = len(properties)
            client_count = len(client_ids)
            
            document_type_breakdown = defaultdict(int)
            document_status_breakdown = defaultdict(int)
            upload_activity = defaultdict(int)  # date -> count
            
            # Process documents
            for doc in docs:
                data = doc.to_dict()
                
                # Document type
                doc_type = self._get_document_type(data)
                document_type_breakdown[doc_type] += 1
                
                # Document status
                status = self._get_processing_status(data)
                document_status_breakdown[status] += 1
                
                # Upload activity (by date)
                created_at = data.get('created_at')
                if created_at:
                    if isinstance(created_at, datetime):
                        date_key = created_at.date().isoformat()
                    else:
                        # Handle Firestore timestamp
                        date_key = created_at.strftime('%Y-%m-%d') if hasattr(created_at, 'strftime') else str(created_at)[:10]
                    upload_activity[date_key] += 1
            
            # Convert upload activity to list sorted by date
            upload_activity_list = [
                {"date": date, "count": count}
                for date, count in sorted(upload_activity.items())
            ]
            
            # Calculate quality score
            total_docs = document_count
            if total_docs > 0:
                pending = document_status_breakdown.get('pending', 0)
                completed = document_status_breakdown.get('completed', 0)
                need_review = document_status_breakdown.get('need_review', 0)
                failed = document_status_breakdown.get('failed', 0)
                
                quality_score = ((pending + completed) / total_docs * 100) if total_docs > 0 else 0
            else:
                quality_score = 0
            
            return {
                "property_count": property_count,
                "client_count": client_count,
                "document_count": document_count,
                "document_type_breakdown": dict(document_type_breakdown),
                "document_status_breakdown": dict(document_status_breakdown),
                "upload_activity": upload_activity_list[-30:],  # Last 30 days
                "quality_score": round(quality_score, 2)
            }
        except Exception as e:
            logger.error(f"Failed to get agent analytics: {e}")
            return {
                "property_count": 0,
                "client_count": 0,
                "document_count": 0,
                "document_type_breakdown": {},
                "document_status_breakdown": {},
                "upload_activity": [],
                "quality_score": 0
            }
    
    def get_property_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get property analytics"""
        try:
            # Get all properties (limit to 1000 for performance)
            if agent_id:
                properties_query = self.properties_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).limit(1000)
            else:
                properties_query = self.properties_collection.limit(1000)
            
            properties = list(properties_query.stream())
            
            # OPTIMIZED: Limit document query to 5000 for performance
            if agent_id:
                docs_query = self.documents_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).limit(5000)
            else:
                docs_query = self.documents_collection.limit(5000)
            
            docs = list(docs_query.stream())
            
            # Documents per property
            property_doc_counts = defaultdict(int)
            property_doc_types = defaultdict(lambda: defaultdict(int))
            
            for doc in docs:
                data = doc.to_dict()
                property_id = data.get('propertyId')
                if property_id:
                    property_doc_counts[property_id] += 1
                    doc_type = self._get_document_type(data)
                    property_doc_types[property_id][doc_type] += 1
            
            # Build documents per property list
            documents_per_property = []
            for prop in properties:
                prop_data = prop.to_dict()
                prop_id = prop.id
                prop_reference = prop_data.get('reference', prop_data.get('title', 'Unknown'))
                doc_count = property_doc_counts.get(prop_id, 0)
                documents_per_property.append({
                    "property_id": prop_id,
                    "property_reference": prop_reference,
                    "document_count": doc_count
                })
            
            # Overall document types breakdown
            document_type_breakdown = defaultdict(int)
            for doc in docs:
                data = doc.to_dict()
                doc_type = self._get_document_type(data)
                document_type_breakdown[doc_type] += 1
            
            # Property workflow progress (using property_files)
            property_files = list(self.property_files_collection.stream())
            workflow_progress = []
            
            for pf in property_files:
                pf_data = pf.to_dict()
                property_id = pf_data.get('property_id')
                status = pf_data.get('status', 'INCOMPLETE')
                
                # Calculate completion percentage
                has_spa = bool(pf_data.get('spa_document_id'))
                has_invoice = bool(pf_data.get('invoice_document_id'))
                has_id = bool(pf_data.get('id_document_id'))
                has_proof = bool(pf_data.get('proof_of_payment_document_id'))
                
                completed_slots = sum([has_spa, has_invoice, has_id, has_proof])
                progress = (completed_slots / 4) * 100 if completed_slots > 0 else 0
                
                workflow_progress.append({
                    "property_id": property_id,
                    "property_reference": pf_data.get('property_reference', 'Unknown'),
                    "progress": round(progress, 2),
                    "status": status
                })
            
            return {
                "documents_per_property": sorted(documents_per_property, key=lambda x: x['document_count'], reverse=True),
                "document_type_breakdown": dict(document_type_breakdown),
                "workflow_progress": workflow_progress
            }
        except Exception as e:
            logger.error(f"Failed to get property analytics: {e}")
            return {
                "documents_per_property": [],
                "document_type_breakdown": {},
                "workflow_progress": []
            }
    
    def get_client_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get client analytics"""
        try:
            # OPTIMIZED: Limit clients query to 1000 for performance
            clients = list(self.clients_collection.limit(1000).stream())
            
            # OPTIMIZED: Limit document query to 5000 for performance
            if agent_id:
                docs_query = self.documents_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).limit(5000)
            else:
                docs_query = self.documents_collection.limit(5000)
            
            docs = list(docs_query.stream())
            
            # OPTIMIZED: Limit property files query to 1000 for performance
            property_files = list(self.property_files_collection.limit(1000).stream())
            
            # Find clients with missing documents
            clients_missing_docs = []
            
            for pf in property_files:
                pf_data = pf.to_dict()
                client_id = pf_data.get('client_id')
                client_name = pf_data.get('client_full_name', 'Unknown')
                
                if not client_id:
                    continue
                
                # Check what's missing
                missing = []
                if not pf_data.get('spa_document_id'):
                    missing.append('SPA')
                if not pf_data.get('invoice_document_id'):
                    missing.append('Invoice')
                if not pf_data.get('id_document_id'):
                    missing.append('ID')
                if not pf_data.get('proof_of_payment_document_id'):
                    missing.append('Proof of Payment')
                
                if missing:
                    clients_missing_docs.append({
                        "client_id": client_id,
                        "client_name": client_name,
                        "missing_documents": missing
                    })
            
            # Count completed contracts (SPA documents)
            spa_docs = [doc for doc in docs if self._get_document_type(doc.to_dict()) == "SPA"]
            completed_contracts = len(spa_docs)
            
            return {
                "clients_missing_documents": clients_missing_docs,
                "completed_contracts": completed_contracts,
                "total_clients": len(clients)
            }
        except Exception as e:
            logger.error(f"Failed to get client analytics: {e}")
            return {
                "clients_missing_documents": [],
                "completed_contracts": 0,
                "total_clients": 0
            }
    
    def get_document_analytics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get document analytics"""
        try:
            # OPTIMIZED: Limit document query to 5000 for performance
            if agent_id:
                docs_query = self.documents_collection.where(
                    filter=FieldFilter('agentId', '==', agent_id)
                ).limit(5000)
            else:
                docs_query = self.documents_collection.limit(5000)
            
            docs = list(docs_query.stream())
            
            total_documents = len(docs)
            
            # Document type distribution
            document_type_breakdown = defaultdict(int)
            
            # Document status breakdown
            document_status_breakdown = defaultdict(int)
            
            # Processing times
            processing_times = []
            
            # Failed document reasons
            failed_reasons = defaultdict(int)
            
            for doc in docs:
                data = doc.to_dict()
                
                # Document type
                doc_type = self._get_document_type(data)
                document_type_breakdown[doc_type] += 1
                
                # Document status
                status = self._get_processing_status(data)
                document_status_breakdown[status] += 1
                
                # Processing time
                created_at = data.get('created_at')
                updated_at = data.get('updated_at')
                if created_at and updated_at:
                    try:
                        if isinstance(created_at, datetime) and isinstance(updated_at, datetime):
                            processing_time = (updated_at - created_at).total_seconds()
                        else:
                            # Handle Firestore timestamps
                            if hasattr(created_at, 'timestamp') and hasattr(updated_at, 'timestamp'):
                                processing_time = updated_at.timestamp() - created_at.timestamp()
                            else:
                                processing_time = 0
                        
                        if processing_time > 0:
                            processing_times.append(processing_time)
                    except Exception:
                        pass
                
                # Failed reasons
                if status == 'failed':
                    error = data.get('error') or data.get('error_message', 'Unknown error')
                    error_lower = error.lower()
                    
                    if 'not a document' in error_lower or 'invalid' in error_lower:
                        failed_reasons['not a document'] += 1
                    elif 'blur' in error_lower or 'unclear' in error_lower:
                        failed_reasons['blurred'] += 1
                    elif 'category' in error_lower or 'classification' in error_lower:
                        failed_reasons['wrong category'] += 1
                    elif 'confidence' in error_lower or 'low' in error_lower:
                        failed_reasons['low confidence'] += 1
                    else:
                        failed_reasons['other'] += 1
            
            # Calculate average processing time
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            return {
                "total_documents": total_documents,
                "document_type_distribution": dict(document_type_breakdown),
                "document_status_chart": dict(document_status_breakdown),
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "failed_document_reasons": dict(failed_reasons)
            }
        except Exception as e:
            logger.error(f"Failed to get document analytics: {e}")
            return {
                "total_documents": 0,
                "document_type_distribution": {},
                "document_status_chart": {},
                "average_processing_time_seconds": 0,
                "failed_document_reasons": {}
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health overview (admin only)"""
        try:
            # OPTIMIZED: Limit document query to 5000 for performance
            docs = list(self.documents_collection.limit(5000).stream())
            
            # Monthly upload volume (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            monthly_uploads = 0
            for doc in docs:
                data = doc.to_dict()
                created_at = data.get('created_at')
                if created_at:
                    try:
                        if isinstance(created_at, datetime):
                            doc_date = created_at
                        elif hasattr(created_at, 'timestamp'):
                            doc_date = datetime.fromtimestamp(created_at.timestamp())
                        else:
                            continue
                        
                        if doc_date >= thirty_days_ago:
                            monthly_uploads += 1
                    except Exception:
                        pass
            
            # Agent activity ranking
            agent_activity = defaultdict(int)
            agent_properties = defaultdict(int)
            agent_clients = defaultdict(int)
            
            for doc in docs:
                data = doc.to_dict()
                agent_id = data.get('agentId')
                if agent_id:
                    agent_activity[agent_id] += 1
                    
                    # Count unique properties per agent
                    property_id = data.get('propertyId')
                    if property_id:
                        agent_properties[agent_id] = agent_properties[agent_id]  # Just track existence
            
            # OPTIMIZED: Limit properties query to 1000 for performance
            properties = list(self.properties_collection.limit(1000).stream())
            for prop in properties:
                prop_data = prop.to_dict()
                agent_id = prop_data.get('agentId')
                if agent_id:
                    agent_properties[agent_id] += 1
            
            # Get clients per agent (from documents)
            for doc in docs:
                data = doc.to_dict()
                agent_id = data.get('agentId')
                client_id = data.get('clientId')
                if agent_id and client_id:
                    agent_clients[agent_id] = agent_clients[agent_id]  # Track unique clients
            
            # Build agent leaderboard
            agent_leaderboard = []
            for agent_id, doc_count in agent_activity.items():
                agent_leaderboard.append({
                    "agent_id": agent_id,
                    "documents_uploaded": doc_count,
                    "properties_managed": agent_properties.get(agent_id, 0),
                    "clients_managed": len([c for c in agent_clients.keys() if c == agent_id])  # Simplified
                })
            
            agent_leaderboard.sort(key=lambda x: x['documents_uploaded'], reverse=True)
            
            # Storage usage (estimate from file sizes)
            total_storage_bytes = sum(
                doc.to_dict().get('file_size', 0) for doc in docs
            )
            total_storage_mb = total_storage_bytes / (1024 * 1024)
            
            return {
                "total_documents": len(docs),
                "total_failed": len([d for d in docs if self._get_processing_status(d.to_dict()) == 'failed']),
                "monthly_uploads": monthly_uploads,
                "storage_usage_mb": round(total_storage_mb, 2),
                "agent_activity_ranking": agent_leaderboard[:10]  # Top 10
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "total_documents": 0,
                "total_failed": 0,
                "monthly_uploads": 0,
                "storage_usage_mb": 0,
                "agent_activity_ranking": []
            }

