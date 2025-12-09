"""
Background task processing for document OCR
"""
import logging
import tempfile
import os
import re
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import BackgroundTasks

sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from services.document_processor import DocumentProcessor
from services.firestore_service import FirestoreService
from services.category_mapper import map_backend_to_ui_category
from gcs_service import GCSVoucherService
from services.mocks import MockFirestoreService, MockGCSVoucherService

logger = logging.getLogger(__name__)

class TaskQueue:
    """Background task processing service"""
    
    def __init__(self):
        """Initialize task queue services"""
        self._document_processor = None
        self._firestore_service = None
        self._gcs_service = None
    
    @property
    def document_processor(self):
        """Lazy initialization of document processor"""
        if self._document_processor is None:
            self._document_processor = DocumentProcessor()
        return self._document_processor
    
    @property
    def firestore_service(self):
        """Lazy initialization of firestore service"""
        if self._firestore_service is None:
            if settings.USE_MOCK_SERVICES:
                self._firestore_service = MockFirestoreService()
            else:
                try:
                    self._firestore_service = FirestoreService()
                except Exception:
                    self._firestore_service = MockFirestoreService()
        return self._firestore_service
    
    @property
    def gcs_service(self):
        """Lazy initialization of GCS service"""
        if self._gcs_service is None:
            if settings.USE_MOCK_SERVICES:
                self._gcs_service = MockGCSVoucherService()
            else:
                try:
                    self._gcs_service = GCSVoucherService()
                except Exception:
                    self._gcs_service = MockGCSVoucherService()
        return self._gcs_service
    
    async def process_document_task(
        self,
        document_id: str,
        gcs_temp_path: str,
        original_filename: str,
        job_id: Optional[str] = None
    ):
        """
        Background task to process a single document
        
        Args:
            document_id: Unique document ID
            gcs_temp_path: GCS path to temporary uploaded file
            original_filename: Original filename
            job_id: Optional job ID for batch processing
        """
        try:
            logger.info(f"Starting background processing for document: {document_id}")
            
            # Initialize result dictionary to avoid UnboundLocalError in finally block
            result = {}
            
            # Get document to retrieve flow_id for count increment
            flow_id = None
            try:
                doc = self.firestore_service.get_document(document_id)
                if doc:
                    flow_id = doc.get('flow_id')
                    logger.info(f"Retrieved flow_id {flow_id} from document {document_id}")
            except Exception as e:
                logger.warning(f"Could not retrieve flow_id from document {document_id}: {e}")
            
            # Fallback: use job_id as flow_id if flow_id is not available (job_id is often the same as flow_id)
            if not flow_id and job_id:
                flow_id = job_id
                logger.info(f"Using job_id {job_id} as flow_id for document {document_id}")
            
            # Update status to processing
            self.firestore_service.update_document(document_id, {
                'processing_status': 'processing'
            })
            
            if job_id:
                self.firestore_service.update_job(job_id, {'status': 'processing'})
            
            # Download file from GCS temp to local temp
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(original_filename).suffix
            )
            temp_file_path = temp_file.name
            temp_file.close()
            
            try:
                # Download from GCS
                bucket = self.gcs_service.bucket
                blob = bucket.blob(gcs_temp_path)
                blob.download_to_filename(temp_file_path)
                logger.info(f"Downloaded file from GCS: {gcs_temp_path}")
                
                # Calculate image hash for duplicate detection (if not already stored)
                import hashlib
                with open(temp_file_path, 'rb') as f:
                    file_content = f.read()
                    image_hash = hashlib.sha256(file_content).hexdigest()
                
                # Update document with image_hash if not already set
                doc = self.firestore_service.get_document(document_id)
                if doc and not doc.get('image_hash'):
                    self.firestore_service.update_document(document_id, {'image_hash': image_hash})
                
                # Process document
                result = self.document_processor.process_document(
                    temp_file_path,
                    original_filename=original_filename
                )
                
                # Check for need_review status (validation indicated unclear document)
                validation_status = result.get('validation_status')
                handled_need_review = False
                
                if validation_status == 'need_review':
                    validation_confidence = result.get('validation_confidence', 0.0)
                    validation_label = result.get('validation_label', '')
                    error_msg = result.get('error', f'Image might be a document but requires review (confidence: {validation_confidence:.4f})')
                    
                    logger.warning(f"Document {document_id} marked for review: {error_msg}")
                    
                    # Store validation metadata in Firestore
                    metadata = {
                        'validation_status': 'need_review',
                        'validation_confidence': validation_confidence,
                        'validation_label': validation_label,
                        'document_type': result.get('document_type', 'Unclear'),
                        'classification': result.get('classification', 'UNKNOWN')
                    }
                    
                    self.firestore_service.update_document(document_id, {
                        'processing_status': 'need_review',
                        'error': error_msg,
                        'metadata': metadata,
                        'validation_confidence': validation_confidence,
                        'validation_label': validation_label
                    })
                    
                    # Update job progress (count as failed for now, but can be reviewed later)
                    if job_id:
                        self.firestore_service.update_job_progress(job_id, failed=1)
                    
                    handled_need_review = True
                
                if result.get('success'):
                    # Upload processed file to organized location
                    organized_path = result.get('organized_path')
                    if organized_path:
                        # Determine which file to upload (PDF if converted, otherwise original)
                        file_to_upload = result.get('pdf_path', temp_file_path)
                        is_pdf = result.get('converted_to_pdf', False)
                        
                        # Generate final filename
                        if result.get('complete_filename'):
                            complete_doc_no = result['complete_filename'].strip()
                            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', complete_doc_no)
                            
                            if is_pdf:
                                final_filename = f"{safe_filename}_0001.pdf"
                            else:
                                original_ext = Path(original_filename).suffix
                                final_filename = f"{safe_filename}_0001{original_ext}"
                        else:
                            filename_without_ext = Path(original_filename).stem
                            if is_pdf:
                                final_filename = f"{filename_without_ext}.pdf"
                            else:
                                original_ext = Path(original_filename).suffix
                                final_filename = f"{filename_without_ext}{original_ext}"
                        
                        organized_key = f"{organized_path}/{final_filename}"
                        
                        # Prepare metadata
                        metadata = {}
                        if result.get('document_no'):
                            metadata['document-no'] = str(result['document_no'])
                        if result.get('classification'):
                            metadata['classification'] = str(result['classification'])
                        if result.get('branch_id'):
                            metadata['branch-id'] = str(result['branch_id'])
                        if result.get('invoice_amount_usd'):
                            metadata['invoice-amount-usd'] = str(result['invoice_amount_usd'])
                        if result.get('invoice_amount_aed'):
                            metadata['invoice-amount-aed'] = str(result['invoice_amount_aed'])
                        if result.get('gold_weight'):
                            metadata['gold-weight'] = str(result['gold_weight'])
                        if result.get('purity'):
                            metadata['purity'] = str(result['purity'])
                        if result.get('document_date'):
                            metadata['document-date'] = str(result['document_date'])
                        if result.get('discount_rate'):
                            metadata['discount-rate'] = str(result['discount_rate'])
                        
                        # Upload to GCS
                        content_type = 'application/pdf' if is_pdf else 'image/jpeg'
                        blob = bucket.blob(organized_key)
                        with open(file_to_upload, 'rb') as file_data:
                            blob.upload_from_file(file_data, content_type=content_type)
                            blob.metadata = metadata
                            blob.patch()
                        
                        logger.info(f"Uploaded file to organized location: {organized_key}")
                        
                        # Map backend classification to UI category
                        backend_classification = result.get('classification')
                        ui_category = map_backend_to_ui_category(backend_classification)
                        
                        # Prepare metadata with all extracted data
                        metadata = {
                            'document_no': result.get('document_no'),
                            'document_date': result.get('document_date'),
                            'branch_id': result.get('branch_id'),
                            'classification': backend_classification,
                            'ui_category': ui_category,
                            'invoice_amount_usd': result.get('invoice_amount_usd'),
                            'invoice_amount_aed': result.get('invoice_amount_aed'),
                            'gold_weight': result.get('gold_weight'),
                            'purity': result.get('purity'),
                            'discount_rate': result.get('discount_rate'),
                            'is_valid_voucher': result.get('is_valid_voucher', False),
                            'needs_attachment': result.get('needs_attachment', False),
                            'document_type': result.get('document_type'),
                            'classification_confidence': result.get('classification_confidence'),
                            'classification_reasoning': result.get('classification_reasoning'),
                            # Include validation metadata
                            'validation_status': result.get('validation_status'),
                            'validation_confidence': result.get('validation_confidence'),
                            'validation_label': result.get('validation_label')
                        }
                        
                        # Include full extracted_data if available (for general documents)
                        if result.get('extracted_data'):
                            extracted_data = result.get('extracted_data', {})
                            # Add any additional fields from extracted_data that aren't already in metadata
                            if isinstance(extracted_data, dict):
                                # Store buyer, seller, items, terms, signatures, etc. if present
                                if 'buyer' in extracted_data:
                                    metadata['buyer'] = extracted_data.get('buyer')
                                if 'seller' in extracted_data:
                                    metadata['seller'] = extracted_data.get('seller')
                                if 'items' in extracted_data:
                                    metadata['items'] = extracted_data.get('items')
                                if 'terms' in extracted_data:
                                    metadata['terms'] = extracted_data.get('terms')
                                if 'signatures' in extracted_data:
                                    metadata['signatures'] = extracted_data.get('signatures')
                                # Store the full extracted_data for reference
                                metadata['full_extracted_data'] = extracted_data
                        
                        # Update Firestore with success and all metadata
                        # IMPORTANT: Update filename with the final processed filename (PDF or image)
                        self.firestore_service.update_document(document_id, {
                            'processing_status': 'completed',
                            'organized_path': organized_path,
                            'gcs_path': f"gs://{settings.GCS_BUCKET_NAME}/{organized_key}",
                            'filename': final_filename,  # Save the final processed filename
                            'metadata': metadata,
                            'processing_method': result.get('method'),
                            'confidence': result.get('confidence'),
                            'extraction_method': result.get('extraction_method')
                        })
                        
                        # PERFORMANCE: Invalidate caches when document is completed
                        # Note: Cache invalidation is handled in main.py via document update hooks
                        # This is a non-blocking operation, so we don't need to wait for it
                        
                        # Property Files: Find or create property file for any of the 4 document types
                        try:
                            # Get document_type from result (already set from classification)
                            document_type = result.get('document_type', '')
                            
                            # Store extracted client name and property reference in metadata
                            if result.get('client_full_name_extracted'):
                                metadata['client_full_name_extracted'] = result.get('client_full_name_extracted')
                            if result.get('property_reference_extracted'):
                                metadata['property_reference_extracted'] = result.get('property_reference_extracted')
                            if result.get('transaction_type'):
                                metadata['transaction_type'] = result.get('transaction_type')
                            
                            # Ensure document_type is in metadata
                            if document_type:
                                metadata['document_type'] = document_type
                            
                            # Update metadata with property file fields
                            self.firestore_service.update_document(document_id, {'metadata': metadata})
                            
                            # Normalize document_type for comparison (handle variations)
                            # Normalize by removing spaces, underscores, and converting to uppercase
                            doc_type_normalized = document_type.upper().strip().replace(' ', '_').replace('-', '_') if document_type else ''
                            # Also create a version with spaces for comparison
                            doc_type_normalized_spaced = document_type.upper().strip() if document_type else ''
                            
                            # Check if this is one of the 4 property file document types
                            # Use multiple normalization variants to catch all cases
                            property_file_doc_types = [
                                'SPA', 
                                'INVOICES', 'INVOICE', 
                                'ID', 
                                'PROOF OF PAYMENT', 'PROOF_OF_PAYMENT', 'PROOF-OF-PAYMENT',
                                'PROOFOFPAYMENT'  # No spaces/separators
                            ]
                            
                            # Check both normalized versions
                            is_property_file_type = (
                                doc_type_normalized in property_file_doc_types or
                                doc_type_normalized_spaced in property_file_doc_types or
                                any(dt in doc_type_normalized or dt in doc_type_normalized_spaced for dt in property_file_doc_types)
                            )
                            
                            # Additional check: if document_type contains keywords, treat as property file type
                            if not is_property_file_type and document_type:
                                doc_lower = document_type.lower()
                                if any(keyword in doc_lower for keyword in ['spa', 'invoice', 'id', 'proof', 'payment']):
                                    logger.info(f"Document type '{document_type}' contains property file keywords - attempting property file matching")
                                    is_property_file_type = True
                            
                            if is_property_file_type:
                                if doc_type_normalized == 'ID' or 'ID' in doc_type_normalized:
                                    logger.info(f"🆔 ID document {document_id} detected - processing for property file matching/creation")
                                    logger.info(f"📋 ID document {document_id} - Extracted data: client_full_name_extracted={result.get('client_full_name_extracted')}, property_reference_extracted={result.get('property_reference_extracted')}")
                                else:
                                    logger.info(f"✅ Property file document type '{document_type}' (normalized: '{doc_type_normalized}') detected - finding or creating property file for document {document_id}")
                                
                                # Validate that we have required data before attempting match
                                has_client_name = bool(result.get('client_full_name_extracted') or metadata.get('client_full_name_extracted'))
                                if not has_client_name:
                                    logger.warning(f"⚠️ Document {document_id} is property file type but no client_full_name_extracted found - will mark as unlinked")
                                
                                try:
                                    self._find_or_create_property_file(document_id, metadata, document_type)
                                    logger.info(f"✅ Completed property file processing for document {document_id}")
                                except Exception as e:
                                    logger.error(f"❌ Error processing property file for document {document_id}: {e}")
                                    if 'ID' in doc_type_normalized:
                                        logger.error(f"❌ ID document {document_id} - Property file processing failed: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            else:
                                logger.debug(f"Document {document_id} with type '{document_type}' is not a property file document type - skipping property file matching")
                        except Exception as e:
                            logger.warning(f"Property file processing error (non-critical): {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        
                        # Increment flow document count when document is successfully moved to organized vouchers
                        if flow_id:
                            try:
                                self.firestore_service.increment_flow_document_count(flow_id)
                                logger.info(f"Incremented document count for flow {flow_id}")
                            except Exception as e:
                                logger.warning(f"Failed to increment document count for flow {flow_id}: {e}")
                        else:
                            logger.warning(f"Could not increment flow document count: flow_id not available for document {document_id}")
                    else:
                        # UNKNOWN classification - mark as failed
                        error_msg = 'Document classified as UNKNOWN - unable to determine document type'
                        logger.error(f"❌ Processing FAILED for document {document_id}: {error_msg}")
                        logger.error(f"   - Classification: {result.get('classification', 'N/A')}")
                        logger.error(f"   - Document type: {result.get('document_type', 'N/A')}")
                        logger.error(f"   - Success: {result.get('success', False)}")
                        logger.error(f"   - Error from result: {result.get('error', 'No error message')}")
                        self.firestore_service.update_document(document_id, {
                            'processing_status': 'failed',
                            'error': error_msg
                        })
                elif not handled_need_review:
                    # Processing failed (skip if we already handled need_review)
                    error_msg = result.get('error', 'Unknown error during processing')
                    logger.error(f"❌ Processing FAILED for document {document_id}: {error_msg}")
                    logger.error(f"   - Result success: {result.get('success', False)}")
                    logger.error(f"   - Classification: {result.get('classification', 'N/A')}")
                    logger.error(f"   - Document type: {result.get('document_type', 'N/A')}")
                    logger.error(f"   - Validation status: {result.get('validation_status', 'N/A')}")
                    logger.error(f"   - Validation confidence: {result.get('validation_confidence', 'N/A')}")
                    logger.error(f"   - Full result keys: {list(result.keys())}")
                    
                    # Include validation metadata if available
                    metadata = {}
                    validation_status = result.get('validation_status')
                    if validation_status:
                        metadata['validation_status'] = validation_status
                        metadata['validation_confidence'] = result.get('validation_confidence')
                        metadata['validation_label'] = result.get('validation_label')
                    
                    update_data = {
                        'processing_status': 'failed',
                        'error': error_msg,
                        'classification': result.get('classification', 'UNKNOWN'),
                        'document_type': result.get('document_type', 'Other')
                    }
                    
                    if metadata:
                        update_data['metadata'] = metadata
                    
                    self.firestore_service.update_document(document_id, update_data)
                
                # Track if we need to move file to failed folder
                processing_failed = not result.get('success') and not handled_need_review
                
                # Prepare error_msg and metadata for file handling if processing failed
                # Extract from result to ensure they're always available
                failed_error_msg = result.get('error', 'Unknown error during processing')
                failed_metadata = {}
                validation_status = result.get('validation_status')
                if validation_status:
                    failed_metadata['validation_status'] = validation_status
                    failed_metadata['validation_confidence'] = result.get('validation_confidence')
                    failed_metadata['validation_label'] = result.get('validation_label')
                
                # Update job progress if batch job
                if job_id:
                    if result.get('success'):
                        self.firestore_service.update_job_progress(job_id, processed=1)
                    else:
                        self.firestore_service.update_job_progress(job_id, failed=1)
                
                # Handle temp file: move to failed folder if processing failed, delete if successful, preserve if need_review
                if processing_failed:
                    # Move failed file to failed folder in GCS
                    if flow_id:
                        try:
                            # Construct failed folder path: organized_vouchers/{flow_id}/failed/{filename}
                            from gcs_service import ORG_PREFIX
                            safe_filename = re.sub(r'[<>:"/\\|?*]', '_', original_filename)
                            failed_key = f"{ORG_PREFIX}/{flow_id}/failed/{safe_filename}"
                            
                            # Copy temp file to failed folder
                            source_blob = bucket.blob(gcs_temp_path)
                            if source_blob.exists():
                                dest_blob = bucket.blob(failed_key)
                                
                                # Copy blob with metadata
                                bucket.copy_blob(source_blob, bucket, dest_blob.name)
                                
                                # Add error metadata
                                dest_blob.metadata = {
                                    'error': failed_error_msg,
                                    'document_id': document_id,
                                    'original_path': gcs_temp_path
                                }
                                if failed_metadata:
                                    for key, value in failed_metadata.items():
                                        dest_blob.metadata[key] = str(value)
                                dest_blob.patch()
                                
                                # Delete original temp file
                                source_blob.delete()
                                
                                logger.info(f"✅ Moved failed file to failed folder: {failed_key}")
                                
                                # Update Firestore with failed file path
                                self.firestore_service.update_document(document_id, {
                                    'gcs_path': f"gs://{settings.GCS_BUCKET_NAME}/{failed_key}",
                                    'failed_path': failed_key
                                })
                            else:
                                logger.warning(f"Temp file {gcs_temp_path} does not exist, cannot move to failed folder")
                        except Exception as e:
                            logger.error(f"Failed to move file to failed folder: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            # Fallback: try to delete temp file
                            try:
                                blob = bucket.blob(gcs_temp_path)
                                blob.delete()
                                logger.info(f"Deleted temp file from GCS (fallback): {gcs_temp_path}")
                            except Exception as del_e:
                                logger.warning(f"Failed to delete temp file from GCS: {del_e}")
                    else:
                        logger.warning(f"Cannot move failed file to failed folder: flow_id not available for document {document_id}")
                        # Fallback: delete temp file
                        try:
                            blob = bucket.blob(gcs_temp_path)
                            blob.delete()
                            logger.info(f"Deleted temp file from GCS (no flow_id): {gcs_temp_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temp file from GCS: {e}")
                elif not handled_need_review:
                    # Processing succeeded - delete temp file
                    try:
                        blob = bucket.blob(gcs_temp_path)
                        blob.delete()
                        logger.info(f"Deleted temp file from GCS: {gcs_temp_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file from GCS: {e}")
                else:
                    # need_review - preserve temp file
                    logger.info(f"Preserving temp file for manual review: {gcs_temp_path}")
                
            finally:
                # Clean up local temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up local temp file: {temp_file_path}")
                
                # Clean up PDF if converted
                if result.get('pdf_path') and result.get('pdf_path') != temp_file_path:
                    pdf_path = result.get('pdf_path')
                    if os.path.exists(pdf_path):
                        os.unlink(pdf_path)
                        logger.info(f"Cleaned up converted PDF: {pdf_path}")
            
            logger.info(f"Completed background processing for document: {document_id}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ EXCEPTION in background task for document {document_id}: {error_msg}")
            import traceback
            logger.error(f"   - Traceback: {traceback.format_exc()}")
            self.firestore_service.update_document(document_id, {
                'processing_status': 'failed',
                'error': error_msg
            })
            
            if job_id:
                self.firestore_service.update_job_progress(job_id, failed=1)
    
    def _find_or_create_property_file(self, document_id: str, metadata: Dict[str, Any], document_type: str):
        """
        Find existing property file or create new one for any document type.
        This method handles all 4 document types: SPA, Invoice, ID, Proof of Payment.
        """
        import uuid
        
        # Normalize document_type for logging
        doc_type_normalized = document_type.upper().strip() if document_type else ''
        is_id_document = doc_type_normalized == 'ID'
        
        if is_id_document:
            logger.info(f"🔍 Processing ID document {document_id} for property file matching/creation")
        
        # Fix: Read fresh metadata from Firestore to ensure we have the latest extracted data
        # This ensures we have the most up-to-date client_full_name_extracted and property_reference_extracted
        try:
            fresh_document = self.firestore_service.get_document(document_id)
            if fresh_document:
                fresh_metadata = fresh_document.get('metadata', {})
                # Merge fresh metadata with parameter metadata, prioritizing fresh data
                # This handles cases where metadata was updated in Firestore but parameter is stale
                if fresh_metadata:
                    # Update parameter metadata with fresh values, but preserve any existing values not in fresh
                    for key, value in fresh_metadata.items():
                        if value:  # Only update if fresh value is not empty/None
                            metadata[key] = value
                    logger.debug(f"Refreshed metadata from Firestore for document {document_id}. Keys: {list(metadata.keys())}")
                else:
                    logger.warning(f"No metadata found in Firestore document {document_id}, using parameter metadata")
            else:
                logger.warning(f"Document {document_id} not found in Firestore, using parameter metadata")
        except Exception as e:
            logger.warning(f"Failed to refresh metadata from Firestore for document {document_id}: {e}. Using parameter metadata.")
        
        # Read client_full_name from metadata (try both fields)
        client_full_name = metadata.get('client_full_name_extracted') or metadata.get('client_full_name')
        
        # Clean up client_full_name - handle empty strings and whitespace
        if client_full_name:
            client_full_name = str(client_full_name).strip()
            if not client_full_name:  # Empty after strip
                client_full_name = None
        
        # Helper function to clean and extract field from metadata
        def clean_metadata_field(value):
            """Clean metadata field: handle None, empty strings, and whitespace"""
            if not value:
                return None
            cleaned = str(value).strip()
            return cleaned if cleaned else None
        
        # For ID documents, property information is not available (ID documents only contain client name)
        if is_id_document:
            property_reference = None
            property_name = None
            property_address = None
            transaction_type = clean_metadata_field(metadata.get('transaction_type')) or 'BUY'  # Default to BUY, but matching won't require it
            logger.info(
                f"📋 [{doc_type_normalized}] Document {document_id} - Extracted data:\n"
                f"   Client Full Name: {client_full_name or '(NOT FOUND)'}\n"
                f"   Property Reference: (not available - ID documents don't contain property info)\n"
                f"   Property Name: (not available - ID documents don't contain property info)\n"
                f"   Property Address: (not available - ID documents don't contain property info)\n"
                f"   Transaction Type: {transaction_type} (optional for ID matching)"
            )
            logger.info(f"ID document {document_id} - Property information not available (ID documents only contain client name)")
        else:
            # Read property fields with cleanup - handle both _extracted and non-_extracted variants
            property_reference = clean_metadata_field(
                metadata.get('property_reference_extracted') or 
                metadata.get('property_reference')
            )
            property_name = clean_metadata_field(
                metadata.get('property_name_extracted') or 
                metadata.get('property_name')
            )
            property_address = clean_metadata_field(
                metadata.get('property_address_extracted') or 
                metadata.get('property_address') or
                metadata.get('address')
            )
            transaction_type = clean_metadata_field(metadata.get('transaction_type')) or 'BUY'
            
            # If property_name is empty, try to construct it from available fields
            if not property_name or property_name.strip() == '':
                if property_reference and property_address:
                    property_name = f"{property_reference} - {property_address}"
                    logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Constructed property_name from property_reference + address: {property_name}")
                elif property_reference:
                    property_name = property_reference
                    logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Using property_reference as property_name: {property_name}")
                elif property_address:
                    property_name = property_address
                    logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Using property_address as property_name: {property_name}")
                else:
                    # Try to get property_name from linked property document if property_id exists
                    if property_id:
                        try:
                            property_doc = self.firestore_service.get_property(property_id)
                            if property_doc:
                                property_name = property_doc.get('title') or property_doc.get('name') or property_doc.get('reference') or None
                                if property_name:
                                    logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Retrieved property_name from property document: {property_name}")
                        except Exception as e:
                            logger.warning(f"⚠️ [{doc_type_normalized}] Document {document_id} - Failed to get property_name from property document: {e}")
            
            logger.info(
                f"📋 [{doc_type_normalized}] Document {document_id} - Extracted data:\n"
                f"   Client Full Name: {client_full_name or '(NOT FOUND)'}\n"
                f"   Property Reference: {property_reference or '(not provided)'}\n"
                f"   Property Name: {property_name or '(not provided)'}\n"
                f"   Property Address: {property_address or '(not provided)'}\n"
                f"   Transaction Type: {transaction_type}"
            )
        
        if not client_full_name:
            logger.warning(
                f"❌ [{doc_type_normalized}] Document {document_id} - Client full name not found. "
                f"Cannot proceed with property file matching/creation."
            )
            if is_id_document:
                logger.warning(
                    f"⚠️ ID document {document_id} cannot create property file - client name extraction failed.\n"
                    f"   Metadata keys available: {list(metadata.keys())}\n"
                    f"   client_full_name_extracted={metadata.get('client_full_name_extracted')}\n"
                    f"   client_full_name={metadata.get('client_full_name')}"
                )
            else:
                logger.warning(
                    f"⚠️ [{doc_type_normalized}] Document {document_id} - Missing client name.\n"
                    f"   Metadata keys available: {list(metadata.keys())}\n"
                    f"   client_full_name_extracted={metadata.get('client_full_name_extracted')}\n"
                    f"   client_full_name={metadata.get('client_full_name')}"
                )
            # Mark as unlinked but don't create property file without client name
            self.firestore_service.update_document(document_id, {'status': 'unlinked'})
            return
        
        # Log successful extraction for debugging
        logger.info(
            f"✅ [{doc_type_normalized}] Document {document_id} - Extracted fields validated:\n"
            f"   Client Full Name: '{client_full_name}' ✓\n"
            f"   Property Reference: {property_reference or '(not provided)'}\n"
            f"   Transaction Type: {transaction_type}"
        )
        
        if is_id_document:
            logger.info(f"✅ ID document {document_id} - Client name extracted successfully: {client_full_name}")
        
        # Get agent_id from document to maintain relationships
        document = self.firestore_service.get_document(document_id)
        agent_id = None
        if document:
            agent_id = document.get('agentId') or document.get('agent_id')
        
        # Normalize document_type (already done above, but keep for consistency)
        if not is_id_document:
            doc_type_normalized = document_type.upper().strip() if document_type else ''
        
        # Find or create client
        clients = self.firestore_service.search_clients_by_name(client_full_name)
        if clients:
            client_id = clients[0]['id']
            if is_id_document:
                logger.info(f"✅ ID document {document_id} - Found existing client {client_id} for {client_full_name}")
            
            # Ensure existing client has agent_id assigned if document has one
            if agent_id:
                existing_client = self.firestore_service.get_client(client_id)
                if existing_client and not existing_client.get('agent_id'):
                    self.firestore_service.update_client(client_id, {'agent_id': agent_id})
                    logger.info(f"Assigned agent {agent_id} to existing client {client_id} from document")
        else:
            # Create new client
            client_id = str(uuid.uuid4())
            client_data = {
                'id': client_id,
                'full_name': client_full_name,
                'created_from': 'document_extraction'
            }
            # Assign agent_id if available from document
            if agent_id:
                client_data['agent_id'] = agent_id
            
            self.firestore_service.create_client(client_id, client_data)
            logger.info(f"Created new client {client_id} for {client_full_name}" + (f" with agent {agent_id}" if agent_id else ""))
            if is_id_document:
                logger.info(f"✅ ID document {document_id} - Created new client {client_id} for {client_full_name}")
        
        # Find or create property
        property_id = None
        if property_reference:
            properties = self.firestore_service.search_properties_by_reference(property_reference)
            if properties:
                property_id = properties[0]['id']
                # Update property with agent_id if it doesn't have one
                if agent_id and not properties[0].get('agentId'):
                    self.firestore_service.update_property(property_id, {'agentId': agent_id})
                    logger.info(f"Updated property {property_id} with agent_id {agent_id}")
            else:
                # Create new property with agent_id
                property_id = str(uuid.uuid4())
                property_data = {
                    'id': property_id,
                    'reference': property_reference
                }
                if agent_id:
                    property_data['agentId'] = agent_id
                self.firestore_service.create_property(property_id, property_data)
                logger.info(f"Created new property {property_id} for reference {property_reference} with agent_id {agent_id}")
        
        # Check if property file already exists for this client+property+transaction_type
        # If property_reference is missing, match only by client name and transaction type
        logger.info(
            f"🔍 [{doc_type_normalized}] Document {document_id} - Searching for matching property files:\n"
            f"   Client: {client_full_name}\n"
            f"   Property Reference: {property_reference or '(not provided)'}\n"
            f"   Property Name: {property_name or '(not provided)'}\n"
            f"   Property Address: {property_address or '(not provided)'}\n"
            f"   Transaction Type: {transaction_type}"
        )
        
        existing_files = self.firestore_service.find_matching_property_file(
            client_full_name=client_full_name,
            property_reference=property_reference,  # Can be None - will match by name only
            property_name=property_name,
            property_address=property_address,
            transaction_type=transaction_type,
            is_id_document=is_id_document  # Pass flag to enable lenient transaction type matching for ID documents
        )
        
        if existing_files:
            # Property file exists - attach document to it
            property_file = existing_files[0]
            property_file_id = property_file.get('id')
            
            if not property_file_id:
                # Safely format error message to avoid ValueError with special characters
                doc_type_str = str(doc_type_normalized)
                doc_id_str = str(document_id)
                logger.error(
                    f"❌ [{doc_type_str}] Document {doc_id_str} - "
                    f"Matched property file has no id field. Property file data: {property_file}"
                )
                # Try to get id from document reference if available
                # If we can't get an id, we can't proceed with attachment
                logger.warning(
                    f"⚠️ [{doc_type_str}] Document {doc_id_str} - "
                    f"Skipping property file attachment due to missing id. Document will remain unlinked."
                )
                return
            
            name_score = property_file.get('name_score', 0.0)
            prop_score = property_file.get('property_score')
            confidence = property_file.get('match_confidence', 0.0)
            
            # Safely format log message to avoid ValueError with special characters
            # Convert all variables to strings to prevent format specifier issues
            doc_type_str = str(doc_type_normalized)
            doc_id_str = str(document_id)
            prop_file_id_str = str(property_file_id)
            confidence_str = f"{confidence:.3f}"
            name_score_str = f"{name_score:.3f}"
            prop_score_str = f"{prop_score:.3f}" if prop_score is not None else 'N/A'
            client_name_str = str(property_file.get('client_full_name', 'N/A'))
            prop_ref_str = str(property_file.get('property_reference', 'N/A'))
            
            logger.info(
                f"✅ [{doc_type_str}] Document {doc_id_str} - Matched with existing property file {prop_file_id_str}:\n"
                f"   Match Confidence: {confidence_str}\n"
                f"   Name Score: {name_score_str}\n"
                f"   Property Score: {prop_score_str}\n"
                f"   Property File Client: {client_name_str}\n"
                f"   Property File Reference: {prop_ref_str}"
            )
            
            # Determine which document field to update based on document type
            doc_type_key = self._get_document_type_key(doc_type_normalized)
            
            if doc_type_key:
                # Update property file with this document
                update_data = {doc_type_key: document_id}
                
                # Check if all 4 documents are now present
                spa_id = property_file.get('spa_document_id') if doc_type_key != 'spa_document_id' else document_id
                invoice_id = property_file.get('invoice_document_id') if doc_type_key != 'invoice_document_id' else document_id
                id_doc_id = property_file.get('id_document_id') if doc_type_key != 'id_document_id' else document_id
                proof_id = property_file.get('proof_of_payment_document_id') if doc_type_key != 'proof_of_payment_document_id' else document_id
                
                # Set the new document ID
                if doc_type_key == 'spa_document_id':
                    spa_id = document_id
                elif doc_type_key == 'invoice_document_id':
                    invoice_id = document_id
                elif doc_type_key == 'id_document_id':
                    id_doc_id = document_id
                elif doc_type_key == 'proof_of_payment_document_id':
                    proof_id = document_id
                
                # Check completion status
                if spa_id and invoice_id and id_doc_id and proof_id:
                    update_data['status'] = 'COMPLETE'
                else:
                    update_data['status'] = 'INCOMPLETE'
                
                self.firestore_service.update_property_file(property_file_id, update_data)
                
                # Create or find deal for this agent + client + property combination
                deal = None
                if agent_id and client_id and property_id:
                    deal = self.firestore_service.find_or_create_deal(
                        agent_id=agent_id,
                        client_id=client_id,
                        property_id=property_id,
                        deal_type=transaction_type,
                        stage='LEAD'
                    )
                    if deal:
                        logger.info(f"Created/found deal {deal.get('id')} for agent {agent_id}, client {client_id}, property {property_id}")
                
                # Update document status with all relationships
                document_update = {
                    'client_id': client_id,
                    'property_file_id': property_file_id,
                    'status': 'linked'
                }
                if property_id:
                    document_update['property_id'] = property_id
                if deal:
                    document_update['dealId'] = deal.get('id')
                self.firestore_service.update_document(document_id, document_update)
                
                # Update property file with agent_id, dealId, and property_name if not set
                property_file_update = {}
                if agent_id and not property_file.get('agent_id'):
                    property_file_update['agent_id'] = agent_id
                if deal and not property_file.get('dealId'):
                    property_file_update['dealId'] = deal.get('id')
                
                # Update property_name if it's missing in the existing property file
                existing_property_name = property_file.get('property_name')
                if not existing_property_name or existing_property_name.strip() == '':
                    if property_name and property_name.strip():
                        property_file_update['property_name'] = property_name
                        logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Updating property file {property_file_id} with property_name: {property_name}")
                
                if property_file_update:
                    self.firestore_service.update_property_file(property_file_id, property_file_update)
                    if agent_id:
                        logger.info(f"Updated property file {property_file_id} with agent_id {agent_id}")
                    if deal:
                        logger.info(f"Updated property file {property_file_id} with dealId {deal.get('id')}")
                    if 'property_name' in property_file_update:
                        logger.info(f"Updated property file {property_file_id} with property_name: {property_file_update['property_name']}")
                
                # Safely format log message to avoid ValueError with special characters
                doc_type_str = str(doc_type_normalized)
                doc_id_str = str(document_id)
                prop_file_id_str = str(property_file_id)
                doc_type_safe = str(document_type)
                status_safe = str(update_data.get('status', 'N/A'))
                logger.info(
                    f"✅ [{doc_type_str}] Document {doc_id_str} - Successfully attached to property file {prop_file_id_str}\n"
                    f"   Document type: {doc_type_safe}\n"
                    f"   Property file status: {status_safe}"
                )
        else:
            # No property file exists - create new one with this document
            # Safely format log message to avoid ValueError with special characters
            doc_type_str = str(doc_type_normalized)
            doc_id_str = str(document_id)
            client_name_safe = str(client_full_name) if client_full_name else '(not provided)'
            prop_ref_safe = str(property_reference) if property_reference else '(not provided)'
            prop_name_safe = str(property_name) if property_name else '(not provided)'
            trans_type_safe = str(transaction_type) if transaction_type else '(not provided)'
            logger.info(
                f"📝 [{doc_type_str}] Document {doc_id_str} - No matching property file found\n"
                f"   Client: {client_name_safe}\n"
                f"   Property Reference: {prop_ref_safe}\n"
                f"   Property Name: {prop_name_safe}\n"
                f"   Transaction Type: {trans_type_safe}\n"
                f"   Creating new property file..."
            )
            self._create_property_file_from_document(document_id, metadata, document_type, client_id, property_id)
    
    def _get_document_type_key(self, doc_type_normalized: str) -> Optional[str]:
        """Get the property file field key for a document type"""
        if doc_type_normalized == 'SPA':
            return 'spa_document_id'
        elif doc_type_normalized in ['INVOICES', 'INVOICE']:
            return 'invoice_document_id'
        elif doc_type_normalized == 'ID':
            return 'id_document_id'
        elif doc_type_normalized in ['PROOF OF PAYMENT', 'PROOF_OF_PAYMENT']:
            return 'proof_of_payment_document_id'
        return None
    
    def _create_property_file_from_document(self, document_id: str, metadata: Dict[str, Any], document_type: str, client_id: str, property_id: Optional[str]):
        """
        Create a new property file from any document type (SPA, Invoice, ID, Proof of Payment).
        This is a generic version that works for all 4 document types.
        """
        import uuid
        
        client_full_name = metadata.get('client_full_name_extracted') or metadata.get('client_full_name')
        property_reference = metadata.get('property_reference_extracted') or metadata.get('property_reference')
        property_name = metadata.get('property_name_extracted') or metadata.get('property_name')
        property_address = metadata.get('property_address_extracted') or metadata.get('property_address') or metadata.get('address')
        transaction_type = metadata.get('transaction_type') or 'BUY'
        
        # Normalize document_type
        doc_type_normalized = document_type.upper().strip() if document_type else ''
        is_id_document = doc_type_normalized == 'ID'
        
        # Ensure property_name is always set - construct from available data if missing
        if not property_name or property_name.strip() == '':
            if property_reference and property_address:
                property_name = f"{property_reference} - {property_address}"
                logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Constructed property_name for property file: {property_name}")
            elif property_reference:
                property_name = property_reference
                logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Using property_reference as property_name: {property_name}")
            elif property_address:
                property_name = property_address
                logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Using property_address as property_name: {property_name}")
            elif property_id:
                # Try to get property_name from linked property document
                try:
                    property_doc = self.firestore_service.get_property(property_id)
                    if property_doc:
                        property_name = property_doc.get('title') or property_doc.get('name') or property_doc.get('reference') or None
                        if property_name:
                            logger.info(f"🔧 [{doc_type_normalized}] Document {document_id} - Retrieved property_name from property document: {property_name}")
                except Exception as e:
                    logger.warning(f"⚠️ [{doc_type_normalized}] Document {document_id} - Failed to get property_name from property document: {e}")
            
            # Last resort: construct from client name and transaction type
            if not property_name or property_name.strip() == '':
                if client_full_name:
                    property_name = f"{client_full_name} - {transaction_type} Property"
                    logger.warning(f"⚠️ [{doc_type_normalized}] Document {document_id} - Using fallback property_name: {property_name}")
                else:
                    property_name = f"{transaction_type} Property"
                    logger.warning(f"⚠️ [{doc_type_normalized}] Document {document_id} - Using minimal fallback property_name: {property_name}")
        
        # Get the document type key
        doc_type_key = self._get_document_type_key(doc_type_normalized)
        
        if not doc_type_key:
            logger.warning(f"Unknown document type {document_type} - cannot create property file")
            if is_id_document:
                logger.error(f"❌ ID document {document_id} - Cannot create property file: unknown document type")
            return
        
        # Get agent_id from document to maintain relationships
        document = self.firestore_service.get_document(document_id)
        agent_id = None
        if document:
            agent_id = document.get('agentId') or document.get('agent_id')
        
        # Create or find deal for this agent + client + property combination
        deal = None
        if agent_id and client_id and property_id:
            deal = self.firestore_service.find_or_create_deal(
                agent_id=agent_id,
                client_id=client_id,
                property_id=property_id,
                deal_type=transaction_type,
                stage='LEAD'
            )
            if deal:
                logger.info(f"Created/found deal {deal.get('id')} for agent {agent_id}, client {client_id}, property {property_id}")
        
        # Create new property file
        property_file_id = str(uuid.uuid4())
        property_file_data = {
            'id': property_file_id,
            'client_id': client_id,
            'client_full_name': client_full_name,
            'property_id': property_id,
            'property_reference': property_reference,
            'property_name': property_name,
            'transaction_type': transaction_type,
            'status': 'INCOMPLETE',
            'created_from_document_type': doc_type_normalized  # Track which document type created this
        }
        
        # Add agent_id and dealId to property file
        if agent_id:
            property_file_data['agent_id'] = agent_id
        if deal:
            property_file_data['dealId'] = deal.get('id')
        
        # Set the appropriate document field based on type
        property_file_data[doc_type_key] = document_id
        
        self.firestore_service.create_property_file(property_file_id, property_file_data)
        
        # Update document status with all relationships
        document_update = {
            'client_id': client_id,
            'property_file_id': property_file_id,
            'status': 'linked'
        }
        if property_id:
            document_update['property_id'] = property_id
        if deal:
            document_update['dealId'] = deal.get('id')
        self.firestore_service.update_document(document_id, document_update)
        
        logger.info(f"Successfully created property file {property_file_id} from {document_type} document {document_id}")
        if is_id_document:
            logger.info(f"✅ ID document {document_id} - Successfully created property file {property_file_id} for client {client_full_name} (property: {property_reference or 'N/A'})")
    
    
    def add_process_task(
        self,
        background_tasks: BackgroundTasks,
        document_id: str,
        gcs_temp_path: str,
        original_filename: str,
        job_id: Optional[str] = None
    ):
        """Add a document processing task to background tasks"""
        background_tasks.add_task(
            self.process_document_task,
            document_id,
            gcs_temp_path,
            original_filename,
            job_id
        )
