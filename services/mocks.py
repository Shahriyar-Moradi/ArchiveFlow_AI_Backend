"""
Mock services for testing without GCP dependencies
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class MockFirestoreService:
    """Mock Firestore service for testing"""
    
    def __init__(self):
        """Initialize mock storage"""
        self.documents = {}
        self.jobs = {}
        self.flows = {}
        logger.info("Mock Firestore service initialized")
    
    def create_document(self, document_id: str, data: Dict[str, Any]) -> str:
        """Create a mock document"""
        data['created_at'] = datetime.now()
        data['updated_at'] = datetime.now()
        self.documents[document_id] = data
        logger.info(f"Mock: Created document {document_id}")
        return document_id
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a mock document"""
        doc = self.documents.get(document_id)
        if doc:
            doc['document_id'] = document_id
        return doc
    
    def update_document(self, document_id: str, data: Dict[str, Any]) -> bool:
        """Update a mock document"""
        if document_id in self.documents:
            self.documents[document_id].update(data)
            self.documents[document_id]['updated_at'] = datetime.now()
            logger.info(f"Mock: Updated document {document_id}")
            return True
        return False
    
    def update_compliance_check_results(self, document_id: str, compliance_data: Dict[str, Any]) -> bool:
        """Update compliance check results"""
        return self.update_document(document_id, {'compliance_check': compliance_data})
    
    def get_compliance_check_results(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get compliance check results"""
        doc = self.get_document(document_id)
        return doc.get('compliance_check') if doc else None
    
    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """List mock documents with pagination"""
        docs = list(self.documents.values())
        total = len(docs)
        
        # Apply simple filtering
        if filters:
            if filters.get('classification'):
                docs = [d for d in docs if d.get('metadata', {}).get('classification') == filters['classification']]
            if filters.get('flow_id'):
                docs = [d for d in docs if d.get('flow_id') == filters['flow_id']]
        
        # Sort by created_at
        docs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        paginated_docs = docs[start:end]
        
        # Add document_id to each
        for doc in paginated_docs:
            for doc_id, doc_data in self.documents.items():
                if doc_data == doc:
                    doc['document_id'] = doc_id
                    break
        
        return paginated_docs, total
    
    def search_documents(self, search_params: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """Search mock documents"""
        return self.list_documents(
            page=search_params.get('page', 1),
            page_size=search_params.get('page_size', 20)
        )
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a mock document"""
        if document_id in self.documents:
            del self.documents[document_id]
            logger.info(f"Mock: Deleted document {document_id}")
            return True
        return False
    
    # Job operations
    
    def create_job(self, job_id: str, data: Dict[str, Any]) -> str:
        """Create a mock job"""
        data['status'] = 'pending'
        data['created_at'] = datetime.now()
        data['updated_at'] = datetime.now()
        data['processed_documents'] = 0
        data['failed_documents'] = 0
        self.jobs[job_id] = data
        logger.info(f"Mock: Created job {job_id}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a mock job"""
        job = self.jobs.get(job_id)
        if job:
            job['job_id'] = job_id
        return job
    
    def update_job(self, job_id: str, data: Dict[str, Any]) -> bool:
        """Update a mock job"""
        if job_id in self.jobs:
            self.jobs[job_id].update(data)
            self.jobs[job_id]['updated_at'] = datetime.now()
            return True
        return False
    
    def update_job_progress(
        self,
        job_id: str,
        processed: int = 0,
        failed: int = 0,
        status: Optional[str] = None
    ) -> bool:
        """Update mock job progress"""
        if job_id not in self.jobs:
            return False
        
        if processed > 0:
            self.jobs[job_id]['processed_documents'] = self.jobs[job_id].get('processed_documents', 0) + processed
        if failed > 0:
            self.jobs[job_id]['failed_documents'] = self.jobs[job_id].get('failed_documents', 0) + failed
        if status:
            self.jobs[job_id]['status'] = status
            if status in ['completed', 'failed']:
                self.jobs[job_id]['completed_at'] = datetime.now()
        
        self.jobs[job_id]['updated_at'] = datetime.now()
        return True
    
    # Flow operations
    
    def create_flow(self, flow_id: str, data: Dict[str, Any]) -> str:
        """Create a mock flow"""
        data['created_at'] = datetime.now()
        data['updated_at'] = datetime.now()
        data['document_count'] = 0
        self.flows[flow_id] = data
        logger.info(f"Mock: Created flow {flow_id}")
        return flow_id
    
    def get_flow(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get a mock flow"""
        flow = self.flows.get(flow_id)
        if flow:
            flow['flow_id'] = flow_id
        return flow
    
    def list_flows(
        self,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """List mock flows"""
        flows = list(self.flows.values())
        total = len(flows)
        
        # Sort by created_at
        flows.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        paginated_flows = flows[start:end]
        
        # Add flow_id to each
        for flow in paginated_flows:
            for flow_id, flow_data in self.flows.items():
                if flow_data == flow:
                    flow['flow_id'] = flow_id
                    break
        
        return paginated_flows, total
    
    def update_flow(self, flow_id: str, data: Dict[str, Any]) -> bool:
        """Update a mock flow"""
        if flow_id in self.flows:
            self.flows[flow_id].update(data)
            self.flows[flow_id]['updated_at'] = datetime.now()
            return True
        return False
    
    def increment_flow_document_count(self, flow_id: str, increment: int = 1) -> bool:
        """Increment mock flow document count"""
        if flow_id in self.flows:
            self.flows[flow_id]['document_count'] = self.flows[flow_id].get('document_count', 0) + increment
            self.flows[flow_id]['updated_at'] = datetime.now()
            return True
        return False
    
    def get_documents_by_flow_id(
        self,
        flow_id: str,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Dict[str, Any]], int]:
        """Get mock documents by flow_id"""
        docs = [d for d in self.documents.values() if d.get('flow_id') == flow_id]
        total = len(docs)
        
        # Sort by created_at
        docs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        paginated_docs = docs[start:end]
        
        # Add document_id to each
        for doc in paginated_docs:
            for doc_id, doc_data in self.documents.items():
                if doc_data == doc:
                    doc['document_id'] = doc_id
                    break
        
        return paginated_docs, total
    
    def get_category_statistics(self) -> Dict[str, int]:
        """Get mock category statistics"""
        category_counts = {}
        for doc in self.documents.values():
            metadata = doc.get('metadata', {})
            ui_category = metadata.get('ui_category', 'Unknown')
            category_counts[ui_category] = category_counts.get(ui_category, 0) + 1
        
        category_counts['total'] = len(self.documents)
        return category_counts


class MockGCSVoucherService:
    """Mock GCS service for testing"""
    
    def __init__(self):
        """Initialize mock storage"""
        self.files = {}
        self.bucket_name = "mock-bucket"
        logger.info("Mock GCS service initialized")
    
    def upload_file(self, file_data: bytes, destination_path: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Mock file upload"""
        self.files[destination_path] = {
            'data': file_data,
            'metadata': metadata or {},
            'created_at': datetime.now()
        }
        logger.info(f"Mock: Uploaded file to {destination_path}")
        return f"gs://{self.bucket_name}/{destination_path}"
    
    def download_file(self, source_path: str) -> Optional[bytes]:
        """Mock file download"""
        file_info = self.files.get(source_path)
        if file_info:
            return file_info['data']
        return None
    
    def delete_file(self, file_path: str) -> bool:
        """Mock file deletion"""
        if file_path in self.files:
            del self.files[file_path]
            logger.info(f"Mock: Deleted file {file_path}")
            return True
        return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """Mock file listing"""
        return [path for path in self.files.keys() if path.startswith(prefix)]
    
    def get_public_url(self, file_path: str) -> str:
        """Mock public URL generation"""
        return f"https://storage.googleapis.com/{self.bucket_name}/{file_path}"

