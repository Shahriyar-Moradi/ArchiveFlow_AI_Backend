"""
Example: How to integrate DynamoDB into your existing main.py

This file shows the key changes needed to replace in-memory storage with DynamoDB
"""

# ============================================
# STEP 1: Import DynamoDB Service
# ============================================
from dynamodb_service import dynamodb_service
import hashlib
from decimal import Decimal

# ============================================
# STEP 2: Update Upload Endpoint
# ============================================

# BEFORE (main.py line ~891):
"""
@app.post("/api/aws/upload")
async def aws_upload_file(file: UploadFile = File(...), batch_id: Optional[str] = None):
    # ... existing S3 upload code ...
    
    # Track AWS batch processing
    final_batch_id = s3_result['batch_id']
    if final_batch_id not in aws_processing_status:
        aws_processing_status[final_batch_id] = {
            'total_documents': 0,
            'processed': 0,
            'completed': 0,
            'errors': 0,
            'documents': {}
        }
    aws_processing_status[final_batch_id]['total_documents'] += 1
"""

# AFTER (add this):
async def aws_upload_file_NEW(file, batch_id=None):
    """Updated upload with DynamoDB integration"""
    
    # Read file content
    content = await file.read()
    document_id = str(uuid.uuid4())
    
    # Calculate content hash for duplicate detection
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Check for duplicates BEFORE uploading
    duplicate_check = dynamodb_service.find_duplicate_by_hash(content_hash)
    if duplicate_check['is_duplicate']:
        return JSONResponse(content={
            "success": False,
            "error": "Duplicate file detected",
            "message": f"This file was already uploaded as {duplicate_check['documents'][0].get('document_no', 'N/A')}",
            "original_document": duplicate_check['documents'][0]
        })
    
    # Upload to S3 (existing code)
    file_type = file.filename.split('.')[-1].lower() if '.' in file.filename else 'jpg'
    s3_result = s3_service.upload_image_to_temp(
        image_data=content,
        filename=file.filename,
        file_type=file_type,
        batch_id=batch_id
    )
    
    if s3_result['success']:
        # Create/update batch in DynamoDB
        final_batch_id = s3_result['batch_id']
        
        batch_exists = dynamodb_service.get_batch(final_batch_id)
        if not batch_exists['success']:
            # Create new batch
            dynamodb_service.create_batch(
                batch_id=final_batch_id,
                batch_name=f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                branch_id=os.getenv('BRANCH_ID', '01'),
                source='web',
                upload_method='aws_s3'
            )
        
        # Create document in DynamoDB
        dynamodb_service.create_document({
            'document_id': document_id,
            'batch_id': final_batch_id,
            'filename': file.filename,
            'file_type': f'image/{file_type}',
            'file_size': len(content),
            's3_key': s3_result['s3_key'],
            'source': 'web',
            'uploaded_by': 'user',  # Add actual user ID
            'content_hash': content_hash
        })
        
        # Log upload event
        dynamodb_service.log_event(
            document_id=document_id,
            batch_id=final_batch_id,
            event_type='upload_completed',
            status='pending',
            message=f'File {file.filename} uploaded to S3',
            details={
                's3_key': s3_result['s3_key'],
                's3_url': s3_result['s3_url'],
                'file_size': len(content)
            }
        )
        
        # Send SQS message (existing code)
        sqs_result = sqs_service.send_upload_message(
            upload_result=s3_result,
            document_id=document_id,
            batch_id=final_batch_id
        )
        
        return JSONResponse(content={
            "success": True,
            "message": "File uploaded successfully",
            "document_id": document_id,
            "filename": file.filename,
            "batch_id": final_batch_id,
            "s3_details": s3_result,
            "sqs_details": sqs_result
        })


# ============================================
# STEP 3: Update Lambda Status Monitor
# ============================================

# BEFORE (main.py line ~293):
"""
async def aws_lambda_status_monitor():
    while True:
        result = sqs_service.receive_processed_messages(aws_config.processed_queue_url, max_messages=10)
        
        if result['success'] and result['messages']:
            for msg in result['messages']:
                batch_id = msg.get('batch_id')
                document_id = msg.get('document_id')
                
                # Update in-memory status
                if batch_id not in aws_processing_status:
                    aws_processing_status[batch_id] = {...}
"""

# AFTER (add this):
async def aws_lambda_status_monitor_NEW():
    """Updated Lambda monitor with DynamoDB"""
    from aws_config import aws_config
    
    while True:
        try:
            # Poll processed messages from Lambda
            result = sqs_service.receive_processed_messages(
                aws_config.processed_queue_url, 
                max_messages=10
            )
            
            if result['success'] and result['messages']:
                for msg in result['messages']:
                    batch_id = msg.get('batch_id')
                    document_id = msg.get('document_id')
                    
                    if not batch_id or not document_id:
                        continue
                    
                    # Update document in DynamoDB
                    processing_status = 'completed' if msg.get('lambda_success') else 'failed'
                    
                    # Prepare OCR data
                    ocr_data = {
                        'document_no': msg.get('document_no'),
                        'document_type': msg.get('classification'),
                        'document_date': msg.get('document_date'),
                        's3_key_organized': msg.get('organized_key'),
                        'year': msg.get('year'),
                        'month': msg.get('month'),
                        'date_str': msg.get('date_str'),
                        'branch_id': msg.get('branch_id'),
                        'text_content': msg.get('text_content', ''),
                        'confidence_score': Decimal(str(msg.get('confidence_score', 0.0)))
                    }
                    
                    # Update document
                    dynamodb_service.update_document_processing(
                        document_id=document_id,
                        batch_id=batch_id,
                        status=processing_status,
                        ocr_data=ocr_data if processing_status == 'completed' else None,
                        error=msg.get('error')
                    )
                    
                    # Log processing event
                    dynamodb_service.log_event(
                        document_id=document_id,
                        batch_id=batch_id,
                        event_type='lambda_processing_completed',
                        status=processing_status,
                        message=f"Lambda processing {processing_status}",
                        details=msg
                    )
                    
                    # Update batch status if all documents processed
                    batch_result = dynamodb_service.get_batch(batch_id)
                    if batch_result['success']:
                        batch_data = batch_result['batch']
                        total = batch_data['total_documents']
                        processed = batch_data['processed_documents']
                        failed = batch_data['failed_documents']
                        
                        # Check if batch complete
                        if processed + failed >= total:
                            batch_status = 'completed' if failed == 0 else 'completed_with_errors'
                            dynamodb_service.update_batch_status(
                                batch_id=batch_id,
                                status=batch_status,
                                processed_count=processed,
                                failed_count=failed
                            )
                    
                    # Broadcast WebSocket update (keep existing code)
                    await manager.broadcast(json.dumps({
                        "type": "aws_lambda_update",
                        "batch_id": batch_id,
                        "document_id": document_id,
                        "status": processing_status,
                        "message": msg
                    }))
                    
                    # Delete processed message from queue
                    if msg.get('receipt_handle'):
                        sqs_service.delete_message(msg['receipt_handle'])
            
            # Sleep before next poll
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"AWS Lambda monitor error: {e}")
            await asyncio.sleep(10)


# ============================================
# STEP 4: Update API Endpoints
# ============================================

# GET Batch Status - BEFORE (main.py line ~1344):
"""
@app.get("/api/aws/batch/{batch_id}/status")
async def get_aws_batch_status(batch_id: str):
    if batch_id in aws_processing_status:
        status = aws_processing_status[batch_id]
        return JSONResponse(content=status)
"""

# AFTER:
@app.get("/api/aws/batch/{batch_id}/status")
async def get_aws_batch_status_NEW(batch_id: str):
    """Get batch status from DynamoDB"""
    
    # Get batch from DynamoDB
    batch_result = dynamodb_service.get_batch(batch_id)
    
    if not batch_result['success']:
        return JSONResponse(content={
            "success": False,
            "batch_id": batch_id,
            "status": "not_found",
            "message": "Batch not found"
        })
    
    batch = batch_result['batch']
    
    # Get all documents in batch
    docs_result = dynamodb_service.list_documents_in_batch(batch_id, limit=1000)
    documents = docs_result.get('documents', [])
    
    # Calculate detailed statistics
    completed_docs = [d for d in documents if d['processing_status'] == 'completed']
    failed_docs = [d for d in documents if d['processing_status'] == 'failed']
    processing_docs = [d for d in documents if d['processing_status'] == 'processing']
    pending_docs = [d for d in documents if d['processing_status'] == 'pending']
    
    return JSONResponse(content={
        "success": True,
        "batch_id": batch_id,
        "status": batch['status'],
        "batch_name": batch.get('batch_name'),
        "created_at": batch['created_at'],
        "completed_at": batch.get('completed_at'),
        "branch_id": batch.get('branch_id'),
        "total_documents": batch['total_documents'],
        "processed_documents": batch['processed_documents'],
        "failed_documents": batch['failed_documents'],
        "pending_documents": len(pending_docs),
        "processing_documents": len(processing_docs),
        "completed_documents": len(completed_docs),
        "documents": documents,
        "processing_summary": {
            "completion_rate": (len(completed_docs) / max(1, len(documents))) * 100,
            "error_rate": (len(failed_docs) / max(1, len(documents))) * 100
        }
    })


# GET Recent Batches - NEW ENDPOINT:
@app.get("/api/batches/recent")
async def get_recent_batches(status: Optional[str] = None, limit: int = 20):
    """List recent batches from DynamoDB"""
    
    if status:
        result = dynamodb_service.list_batches_by_status(status, limit)
    else:
        # Get completed batches by default
        result = dynamodb_service.list_batches_by_status('completed', limit)
    
    if result['success']:
        return JSONResponse(content={
            "success": True,
            "batches": result['batches'],
            "count": result['count']
        })
    else:
        return JSONResponse(content={
            "success": False,
            "error": result.get('error'),
            "batches": []
        })


# GET Document History - NEW ENDPOINT:
@app.get("/api/document/{document_id}/history")
async def get_document_history(document_id: str, batch_id: Optional[str] = None):
    """Get processing history for a document"""
    
    # Get document details
    if batch_id:
        doc_result = dynamodb_service.get_document(document_id, batch_id)
        document = doc_result.get('document')
    else:
        document = None
    
    # Get processing history
    history_result = dynamodb_service.get_document_history(document_id, limit=50)
    
    return JSONResponse(content={
        "success": True,
        "document_id": document_id,
        "document": document,
        "history": history_result.get('logs', []),
        "event_count": history_result.get('count', 0)
    })


# ============================================
# STEP 5: Remove In-Memory Storage
# ============================================

# REMOVE these lines from main.py (line ~121-130):
"""
# Global state for batch jobs (in production, use Redis or database)
batch_jobs: Dict[str, BatchJob] = {}
processing_queue: asyncio.Queue = asyncio.Queue()

# Document tracking to prevent duplicates
processed_documents: Dict[str, Dict[str, Any]] = {}
PROCESSED_DOCS_FILE = BASE_DIR / "processed_documents.json"

# AWS processing status tracking
aws_processing_status: Dict[str, Dict[str, Any]] = {}
"""

# REPLACE with:
"""
# DynamoDB service (replaces in-memory storage)
from dynamodb_service import dynamodb_service

# Keep processing queue for local processing only
processing_queue: asyncio.Queue = asyncio.Queue()
"""


# ============================================
# USAGE EXAMPLES
# ============================================

def example_usage():
    """Examples of using DynamoDB service"""
    
    # Example 1: Create batch
    batch = dynamodb_service.create_batch(
        batch_id='batch-123',
        batch_name='Morning Uploads',
        branch_id='01'
    )
    
    # Example 2: Create document
    doc = dynamodb_service.create_document({
        'document_id': 'doc-456',
        'batch_id': 'batch-123',
        'filename': 'voucher.jpg',
        'file_type': 'image/jpeg',
        's3_key': 'temp/batch-123/voucher.jpg',
        'content_hash': 'sha256-abc...'
    })
    
    # Example 3: Check for duplicates
    duplicate = dynamodb_service.find_duplicate_by_hash('sha256-abc...')
    if duplicate['is_duplicate']:
        print("Duplicate found!")
    
    # Example 4: Update with OCR results
    result = dynamodb_service.update_document_processing(
        document_id='doc-456',
        batch_id='batch-123',
        status='completed',
        ocr_data={
            'document_no': 'MRT01-85702',
            'document_type': 'MRT',
            'confidence_score': Decimal('0.95')
        }
    )
    
    # Example 5: Query documents
    docs = dynamodb_service.query_documents_by_type_and_date(
        document_type='MRT',
        start_date='2025-06-01',
        end_date='2025-06-30'
    )
    print(f"Found {docs['count']} MRT documents")


# ============================================
# IMPORTANT NOTES
# ============================================
"""
1. Always use Decimal for numeric values:
   from decimal import Decimal
   score = Decimal('0.95')  # ✅ Correct
   score = 0.95  # ❌ Wrong

2. Check for duplicates BEFORE uploading to S3:
   - Use find_duplicate_by_hash() for file content
   - Use find_document_by_number() for document numbers

3. Update batch counters automatically:
   - create_document() increments total_documents
   - update_document_processing() updates processed/failed counts

4. Log important events:
   - Use log_event() for upload, processing, errors
   - Provides audit trail and debugging info

5. Handle errors gracefully:
   - All service methods return {'success': bool, ...}
   - Always check 'success' before using data
"""


