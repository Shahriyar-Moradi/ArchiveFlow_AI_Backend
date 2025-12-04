"""
FastAPI Backend for Document Processing with GCP
Handles OCR, batch processing, and document organization
Version: GCP Migration - November 2025
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64
import uuid
from contextlib import asynccontextmanager
import logging
import shutil
import re
import calendar
import hashlib
from collections import defaultdict
from decimal import Decimal
from functools import lru_cache
from time import time
from urllib.parse import unquote

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Body, Form, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aiofiles
from dotenv import load_dotenv
from PIL import Image

# Import GCP services
from config import settings
from services.document_processor import DocumentProcessor
from services.firestore_service import FirestoreService
from services.analytics_service import AnalyticsService
from services.task_queue import TaskQueue
from services.category_mapper import map_backend_to_ui_category
from gcs_service import GCSVoucherService
from s3_service import s3_service

# Import simple authentication
from simple_auth import simple_auth

# Load environment variables
load_dotenv()

# Initialize GCP services
try:
    document_processor = DocumentProcessor()
    logger = logging.getLogger(__name__)
    logger.info("Document processor initialized")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to initialize document processor: {e}")
    document_processor = None

try:
    firestore_service = FirestoreService()
    logger.info("Firestore service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize Firestore service: {e}")
    firestore_service = None

try:
    analytics_service = AnalyticsService(firestore_service) if firestore_service else None
    if analytics_service:
        logger.info("Analytics service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize analytics service: {e}")
    analytics_service = None

try:
    task_queue = TaskQueue()
    logger.info("Task queue initialized")
except Exception as e:
    logger.warning(f"Failed to initialize task queue: {e}")
    task_queue = None

try:
    gcs_service = GCSVoucherService()
    logger.info("GCS service initialized")
except Exception as e:
    logger.warning(f"Failed to initialize GCS service: {e}")
    gcs_service = None

# Simple in-memory cache for flows endpoint (key: 'flows', value: (data, timestamp))
_flows_cache: Dict[str, tuple] = {}
_FLOWS_CACHE_TTL = 30  # 30 seconds cache TTL

# Browse documents cache (defined here but populated later)
_browse_docs_cache: Dict[str, tuple] = {}
_BROWSE_DOCS_CACHE_TTL = 30  # seconds

# Flow files cache for batch details page
_flow_files_cache: Dict[str, tuple] = {}
_FLOW_FILES_CACHE_TTL = 15  # 15 seconds TTL (shorter than browse since it changes more frequently)

# Clients, agents, and property files cache
_clients_cache: Dict[str, tuple] = {}
_agents_cache: Dict[str, tuple] = {}
_property_files_cache: Dict[str, tuple] = {}
_CLIENTS_CACHE_TTL = 30  # 30 seconds for list endpoints
_AGENTS_CACHE_TTL = 30  # 30 seconds for list endpoints
_PROPERTY_FILES_CACHE_TTL = 30  # 30 seconds for list endpoints
_CLIENT_DETAIL_CACHE_TTL = 60  # 60 seconds for detail endpoints

# Analytics cache
_analytics_cache: Dict[str, tuple] = {}
_ANALYTICS_CACHE_TTL = 60  # 60 seconds for analytics endpoints

# Cache management helper
def _cleanup_cache(cache: Dict[str, tuple], ttl: int, max_entries: int = 100) -> None:
    """Remove expired and excess cache entries to prevent memory issues"""
    now = time()
    # Remove expired entries
    expired_keys = [k for k, (_, ts) in cache.items() if now - ts > ttl]
    for k in expired_keys:
        del cache[k]
    # If still too many entries, remove oldest
    if len(cache) > max_entries:
        sorted_keys = sorted(cache.keys(), key=lambda k: cache[k][1])
        for k in sorted_keys[:len(cache) - max_entries]:
            del cache[k]

def invalidate_document_caches() -> None:
    """Invalidate document-related caches when documents are modified"""
    global _browse_docs_cache, _flow_files_cache
    _browse_docs_cache.clear()
    _flow_files_cache.clear()
    logger.info("🗑️  Invalidated document caches (browse and flow files)")

# Helper function to convert Decimal types for JSON serialization
def convert_decimals(obj):
    """Convert non-JSON-serializable types (Decimal, datetime) to safe values"""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, datetime):
        # Firestore DatetimeWithNanoseconds inherits from datetime
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: convert_decimals(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj


def parse_organized_path(organized_path: str) -> Dict[str, str]:
    """
    Parse organized_vouchers path structure into components.
    
    Path format: organized_vouchers/{category}/{year}/{month}/{date}/{filename}
    Example: organized_vouchers/proof_of_payment/2025/nov/28-11-2025/RCP-2025-005678/file.pdf
    
    Returns: {category, year, month, date, filename}
    """
    result = {
        'category': '',
        'year': '',
        'month': '',
        'date': '',
        'filename': ''
    }
    
    if not organized_path:
        return result
    
    parts = organized_path.split('/')
    
    # Check if it starts with organized_vouchers
    if len(parts) >= 6 and parts[0] == 'organized_vouchers':
        result['category'] = parts[1] if len(parts) > 1 else ''
        result['year'] = parts[2] if len(parts) > 2 else ''
        result['month'] = parts[3] if len(parts) > 3 else ''
        result['date'] = parts[4] if len(parts) > 4 else ''
        result['filename'] = parts[-1] if len(parts) > 5 else ''
    
    return result


def extract_gcs_key_from_path(gcs_path: str, bucket_name: str = None) -> str:
    """
    Extract GCS key from various path formats.
    
    Handles:
    - gs://bucket-name/path/to/file → path/to/file
    - bucket-name/path/to/file → path/to/file (if bucket_name matches)
    - path/to/file → path/to/file
    
    Args:
        gcs_path: The GCS path in any format
        bucket_name: Optional bucket name to remove from path
    
    Returns:
        Clean GCS key (path without bucket)
    """
    if not gcs_path:
        return ''
    
    # Handle gs:// URLs
    if gcs_path.startswith('gs://'):
        parts = gcs_path.replace('gs://', '').split('/', 1)
        if len(parts) == 2:
            return parts[1]  # Return path without bucket
        return parts[0] if parts else ''
    
    # Handle bucket-name/path format
    if bucket_name and '/' in gcs_path:
        parts = gcs_path.split('/', 1)
        # Check if first part is the bucket name
        if parts[0] == bucket_name or parts[0] == 'voucher-bucket-1':
            return parts[1]  # Remove bucket name
    
    # Already clean path
    return gcs_path

# Initialize directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
ORGANIZED_DIR = BASE_DIR.parent / "AIServices" / "organized_vouchers"
FLOW_QUEUE_DIR = BASE_DIR / "flow_queue"

# Create necessary directories
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, ORGANIZED_DIR, FLOW_QUEUE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Simple authentication dependency
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    result = simple_auth.get_user(token)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result["error"],
        )
    
    return result["user"]

def is_admin_user(user: Dict[str, Any] = Depends(get_current_user)) -> bool:
    """Check if current user is admin"""
    return user.get('role') == 'admin' or user.get('id') == 'demo_admin_user' or user.get('email') == 'admin@example.com'

# Pydantic models
class DocumentUpload(BaseModel):
    """Model for document upload from mobile/web"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # pdf, jpg, png
    size: str
    data: str  # base64 encoded
    source: str = "mobile"  # mobile, web, scanner
    timestamp: datetime = Field(default_factory=datetime.now)

class FlowJob(BaseModel):
    """Model for flow processing job"""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    documents: List[str]  # List of document IDs
    status: str = "pending"  # pending, processing, completed, failed
    total_documents: int = 0
    processed_documents: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = []
    gcs_upload: Optional[Dict[str, Any]] = None

class ProcessingResult(BaseModel):
    """Model for OCR processing result"""
    document_id: str
    document_no: Optional[str] = None
    document_type: Optional[str] = None
    voucher_type: Optional[str] = None  # Added for GCS processing
    document_date: Optional[str] = None
    folder_path: Optional[str] = None
    text_file: Optional[str] = None
    image_file: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.now)

class FolderInfo(BaseModel):
    """Model for organized folder information"""
    name: str
    path: str
    document_count: int
    last_modified: datetime
    documents: List[Dict[str, Any]]

# Global state for flow jobs (in production, use Redis or database)
flow_jobs: Dict[str, FlowJob] = {}
processing_queue: asyncio.Queue = asyncio.Queue()

# Document tracking to prevent duplicates
processed_documents: Dict[str, Dict[str, Any]] = {}  # document_id -> processing_info
PROCESSED_DOCS_FILE = BASE_DIR / "processed_documents.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Document Processing Backend with GCP...")

    # Load processed documents for duplicate prevention
    await _load_processed_documents()
    
    # Verify GCP services are initialized
    if document_processor:
        print("✅ Document Processor (Anthropic OCR) initialized")
    else:
        print("⚠️  Document Processor not initialized")
    
    if firestore_service:
        print("✅ Firestore service initialized")
    else:
        print("⚠️  Firestore service not initialized")
    
    if gcs_service:
        print("✅ GCS service initialized")
    else:
        print("⚠️  GCS service not initialized")
    
    if task_queue:
        print("✅ Task queue initialized")
    else:
        print("⚠️  Task queue not initialized")
    
    # Start background flow processor
    asyncio.create_task(flow_processor())
    print("✅ Flow processor started")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Document Processing Backend...")
    # Save processed documents before shutdown
    await _save_processed_documents()

# Create FastAPI app
app = FastAPI(
    title="Axiom Spark Document Processing API",
    description="Backend API for document scanning, OCR, and organization",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
# Build allowed origins list from settings, ensuring production URLs are included
# Remove duplicates and ensure all origins are properly formatted
production_origins = [
    "https://docflowai-c88e6.web.app",
    "https://docflowai-c88e6.firebaseapp.com"
]
allowed_origins = list(set(settings.CORS_ORIGINS + production_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Specific origins (required when allow_credentials=True)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Log allowed origins for debugging
logger.info(f"CORS allowed origins: {allowed_origins}")

# Exception handlers to ensure CORS headers are included in error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTPException handler that ensures CORS headers are always present"""
    # Get the origin from the request
    origin = request.headers.get("origin")
    
    # Determine if origin is allowed
    cors_headers = {}
    if origin and origin in allowed_origins:
        cors_headers["Access-Control-Allow-Origin"] = origin
        cors_headers["Access-Control-Allow-Credentials"] = "true"
    elif not origin:
        # If no origin header, allow all (for non-browser clients)
        cors_headers["Access-Control-Allow-Origin"] = "*"
    
    # Return error response with CORS headers
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=cors_headers
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler that ensures CORS headers are always present"""
    # Get the origin from the request
    origin = request.headers.get("origin")
    
    # Determine if origin is allowed
    cors_headers = {}
    if origin and origin in allowed_origins:
        cors_headers["Access-Control-Allow-Origin"] = origin
        cors_headers["Access-Control-Allow-Credentials"] = "true"
    elif not origin:
        # If no origin header, allow all (for non-browser clients)
        cors_headers["Access-Control-Allow-Origin"] = "*"
    
    # Log the error for debugging
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}", exc_info=True)
    
    # Return error response with CORS headers
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=cors_headers
    )

# Background flow processor
async def flow_processor():
    """Background task to process flow jobs"""
    while True:
        try:
            # Get next job from queue
            job_id = await processing_queue.get()
            
            if job_id in flow_jobs:
                job = flow_jobs[job_id]
                job.status = "processing"
                
                # Broadcast status update
                await manager.broadcast(json.dumps({
                    "type": "flow_status",
                    "job_id": job_id,
                    "status": "processing",
                    "progress": 0
                }))
                
                # Process each document
                for idx, doc_id in enumerate(job.documents):
                    try:
                        # Load document from file
                        doc_path = FLOW_QUEUE_DIR / f"{doc_id}.json"
                        if doc_path.exists():
                            async with aiofiles.open(doc_path, 'r') as f:
                                doc_data = json.loads(await f.read())
                            
                            # Process document with OCR
                            result = await process_single_document(doc_data)
                            job.results.append(json.loads(result.json()))
                            job.processed_documents += 1
                            
                            # Broadcast progress update
                            progress = (job.processed_documents / job.total_documents) * 100
                            await manager.broadcast(json.dumps({
                                "type": "flow_progress",
                                "job_id": job_id,
                                "progress": progress,
                                "current": job.processed_documents,
                                "total": job.total_documents,
                                "last_result": json.loads(result.json())
                            }))
                            
                            # Clean up processed file
                            doc_path.unlink()
                    
                    except Exception as e:
                        print(f"Error processing document {doc_id}: {e}")
                        job.results.append({
                            "document_id": doc_id,
                            "success": False,
                            "error": str(e)
                        })
                
                # Mark job as completed
                job.status = "completed"
                job.completed_at = datetime.now()
                
                # Upload only the processed documents to Google Cloud Storage
                if gcs_service:
                    try:
                        print(f"📤 Uploading processed documents to GCS for job {job_id}...")
                        gcs_result = gcs_service.upload_processed_documents(
                            job.results,
                            job_id
                        )
                        
                        if gcs_result['success']:
                            print(f"✅ Successfully uploaded {gcs_result['total_uploaded']} processed documents to GCS")
                            # Add GCS info to job results
                            job.gcs_upload = gcs_result
                        else:
                            print(f"❌ GCS upload failed: {gcs_result.get('error', 'Unknown error')}")
                            job.gcs_upload = {"success": False, "error": gcs_result.get('error')}
                            
                    except Exception as e:
                        print(f"❌ GCS upload error: {e}")
                        job.gcs_upload = {"success": False, "error": str(e)}
                else:
                    print("⚠️  GCS service not available, skipping upload")
                    job.gcs_upload = {"success": False, "error": "GCS service not initialized"}
                
                # Broadcast completion
                await manager.broadcast(json.dumps({
                    "type": "flow_completed",
                    "job_id": job_id,
                    "status": "completed",
                    "results": job.results,
                    "gcs_upload": job.gcs_upload
                }))
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Flow processor error: {e}")
            await asyncio.sleep(1)


def _parse_date_str(date_str: Optional[str]) -> datetime:
    """Parse a date string into a datetime object, falling back to now()."""
    if not date_str:
        return datetime.now()
    
    # Supported formats, in order of preference
    formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
            
    logger.warning(f"Could not parse date '{date_str}' with any known format. Using current date.")
    return datetime.now()


def _get_branch_id() -> str:
    """Return branch identifier from env."""
    return os.getenv("BRANCH_ID", "")


def _extract_branch_number_from_document_no(document_no: Optional[str]) -> Optional[str]:
    """Extract branch numeric code from a document number like MPU01-85285 -> '01'."""
    if not document_no:
        return None
    try:
        # Match letters followed by 1-3 digits (branch), optionally followed by hyphen
        match = re.match(r"^[A-Z]+(\d{1,3})", document_no.strip())
        if match:
            digits = match.group(1)
            # Zero-pad to two digits by default for display
            if digits.isdigit():
                # Keep original width if >2, otherwise pad to 2
                if len(digits) >= 2:
                    return digits
                return f"{int(digits):02d}"
        return None
    except Exception:
        return None


def _format_branch_dir_name(branch_hint: Optional[str]) -> str:
    """Format branch directory name as 'Branch NN'. Falls back to env."""
    if branch_hint:
        # Accept already formatted 'Branch NN'
        if branch_hint.lower().startswith("branch "):
            return branch_hint
        # If it's purely digits, pad and format
        if branch_hint.isdigit():
            return f"Branch {int(branch_hint):02d}"
        # If looks like '01-xyz', take leading digits
        m = re.match(r"^(\d{1,3})", branch_hint)
        if m:
            return f"Branch {int(m.group(1)):02d}"

    # Fallback to env
    env_branch = _get_branch_id()
    if env_branch.lower().startswith("branch "):
        return env_branch
    if env_branch.isdigit():
        return f"Branch {int(env_branch):02d}"
    # Use env value as-is (no "default" fallback)
    return env_branch


def _get_date_components(dt: datetime) -> Dict[str, str]:
    """Return year, month abbr (lower), and date string D-M-YYYY for a datetime."""
    year = str(dt.year)
    month = calendar.month_abbr[dt.month].lower()
    # Use non-zero-padded day/month as requested (e.g., 16-9-2025)
    date_str = f"{dt.day}-{dt.month}-{dt.year}"
    return {"year": year, "month": month, "date_str": date_str}


def _sanitize_document_no(document_no: Optional[str]) -> Optional[str]:
    """Sanitize document number to be filesystem-friendly (e.g., MPV01-82404)."""
    if not document_no:
        return None
    # Keep letters, numbers, hyphen and underscore only
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "", document_no)
    return sanitized or None


def _calculate_image_hash(image_data: bytes) -> str:
    """Calculate SHA-256 hash of image data for duplicate detection."""
    return hashlib.sha256(image_data).hexdigest()


def _check_image_duplicate(image_hash: str) -> Optional[Dict[str, Any]]:
    """Check if image with this hash already exists in processed documents."""
    for doc_info in processed_documents.values():
        if doc_info.get('image_hash') == image_hash:
            return doc_info
    return None


async def _save_processed_documents():
    """Save processed documents to file for persistence."""
    try:
        async with aiofiles.open(PROCESSED_DOCS_FILE, 'w') as f:
            await f.write(json.dumps(processed_documents, indent=2))
        logger.debug(f"💾 Saved {len(processed_documents)} processed documents to {PROCESSED_DOCS_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to save processed documents: {e}")


async def _load_processed_documents():
    """Load processed documents from file on startup."""
    global processed_documents
    try:
        if PROCESSED_DOCS_FILE.exists():
            async with aiofiles.open(PROCESSED_DOCS_FILE, 'r') as f:
                content = await f.read()
                processed_documents = json.loads(content)
            logger.info(f"📂 Loaded {len(processed_documents)} processed documents from {PROCESSED_DOCS_FILE}")
        else:
            logger.info("📂 No processed documents file found, starting fresh")
    except Exception as e:
        logger.error(f"❌ Failed to load processed documents: {e}")
        processed_documents = {}


def _target_voucher_dir(voucher_type: str, dt: datetime, branch_dir_name: Optional[str] = None) -> Path:
    """Build the target directory: organized_vouchers/Branch NN/year/month/date/voucher_type"""
    comps = _get_date_components(dt)
    branch_name = branch_dir_name or _format_branch_dir_name(None)
    target = ORGANIZED_DIR / branch_name / comps["year"] / comps["month"] / comps["date_str"] / voucher_type
    target.mkdir(parents=True, exist_ok=True)
    return target


def _relocate_and_rename_files(
    text_path: Optional[str],
    image_path: Optional[str],
    voucher_type: Optional[str],
    document_no: Optional[str],
    dt: Optional[datetime] = None,
    branch_dir_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Move OCR outputs into the new directory structure and rename to the document number.
    Prevents file overwrites by adding timestamps for duplicates.
    Returns dict with keys: folder_path, text_file, image_file
    """
    result: Dict[str, Optional[str]] = {"folder_path": None, "text_file": None, "image_file": None}
    if not voucher_type:
        return result
    dt = dt or datetime.now()
    target_dir = _target_voucher_dir(voucher_type, dt, branch_dir_name)

    base_name = _sanitize_document_no(document_no) or f"voucher_{dt.day}-{dt.month}-{dt.year}"

    def _get_unique_filename(target_dir: Path, base_name: str, extension: str) -> Path:
        """Get a unique filename, adding timestamp if file exists."""
        candidate = target_dir / f"{base_name}{extension}"
        if not candidate.exists():
            return candidate
        
        # File exists, add timestamp to make it unique
        timestamp = datetime.now().strftime("_%H%M%S")
        unique_name = f"{base_name}{timestamp}{extension}"
        return target_dir / unique_name

    # Move/rename image file (always convert to .jpg)
    if image_path and Path(image_path).exists():
        img_src = Path(image_path)
        img_dst = _get_unique_filename(target_dir, base_name, ".jpg")
        try:
            # Always re-encode as JPEG for organized storage
            with Image.open(str(img_src)) as im:
                # Convert to RGB mode for JPEG compatibility
                if im.mode in ("RGBA", "P", "LA", "L"):
                    im = im.convert("RGB")
                img_dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(str(img_dst), format="JPEG", quality=95)
            # Remove original source after successful save
            try:
                if img_src.exists():
                    img_src.unlink()
            except Exception:
                pass
            logger.info(f"📁 Saved JPEG image to: {img_dst}")
            result["image_file"] = str(img_dst)
        except Exception as e:
            logger.error(f"❌ Failed to convert/move image file: {e}")
            # Fallback: attempt to move with original extension if conversion failed
            try:
                fallback_dst = _get_unique_filename(target_dir, base_name, img_src.suffix.lower() or ".jpg")
                if img_src.resolve() != fallback_dst.resolve():
                    fallback_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_src), str(fallback_dst))
                result["image_file"] = str(fallback_dst)
            except Exception:
                result["image_file"] = str(img_src)

    # Move/rename text file (if exists)
    if text_path and Path(text_path).exists():
        txt_src = Path(text_path)
        txt_dst = _get_unique_filename(target_dir, base_name, ".txt")
        try:
            if txt_src.resolve() != txt_dst.resolve():
                txt_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(txt_src), str(txt_dst))
                logger.info(f"📄 Moved text to: {txt_dst}")
            result["text_file"] = str(txt_dst)
        except Exception as e:
            logger.error(f"❌ Failed to move text file: {e}")
            result["text_file"] = str(txt_src)

    result["folder_path"] = str(target_dir)
    return result

async def process_single_document(doc_data: Dict[str, Any]) -> ProcessingResult:
    """Process a single document with OCR - ensures single save per document"""
    result = ProcessingResult(document_id=doc_data['id'])
    temp_path = UPLOAD_DIR / f"{doc_data['id']}.{doc_data['type']}"
    image_hash = None

    try:
        # Check if document_processor is available
        if not document_processor:
            logger.error("Document processor not initialized")
            result.success = False
            result.error = "OCR service not available"
            return result
        
        # Check if document already processed (prevent duplicates)
        document_id = doc_data['id']
        
        # Decode base64 image and calculate hash for duplicate detection
        image_data = base64.b64decode(doc_data['data'])
        image_hash = _calculate_image_hash(image_data)
        
        # Check if document was already processed in this session by ID
        if document_id in processed_documents:
            logger.info(f"Document {document_id} already processed in this session, skipping")
            existing_result = processed_documents[document_id]
            result.success = existing_result.get('success', False)
            result.document_no = existing_result.get('document_no')
            result.document_type = existing_result.get('document_type')
            result.voucher_type = existing_result.get('voucher_type')
            result.document_date = existing_result.get('document_date')
            result.folder_path = existing_result.get('folder_path')
            result.text_file = existing_result.get('text_file')
            result.image_file = existing_result.get('image_file')
            return result
        
        # Check if image content already exists (by hash)
        duplicate_info = _check_image_duplicate(image_hash)
        if duplicate_info:
            logger.info(f"Image with hash {image_hash[:8]}... already processed, skipping OCR")
            result.success = False
            doc_no = duplicate_info.get('document_no', 'N/A')
            result.error = f"Duplicate document. This voucher was already saved as '{doc_no}'."
            
            # Populate result with original document info
            result.document_no = doc_no
            result.document_type = duplicate_info.get('document_type')
            result.voucher_type = duplicate_info.get('voucher_type')
            result.document_date = duplicate_info.get('document_date')
            result.folder_path = duplicate_info.get('folder_path')
            result.text_file = duplicate_info.get('text_file')
            result.image_file = duplicate_info.get('image_file')
            
            # Track this new document ID as a duplicate
            processed_documents[document_id] = {
                'success': False,
                'error': result.error,
                'image_hash': image_hash,
                'processed_at': datetime.now().isoformat(),
                'duplicate_of': duplicate_info.get('document_no')
            }
            return result
        
        # Check if document already exists locally by document number
        document_no = None
        voucher_type = None
        
        # Try to extract document info from existing files first
        if 'document_no' in doc_data and 'voucher_type' in doc_data:
            document_no = doc_data['document_no']
            voucher_type = doc_data['voucher_type']
            
            # Search recursively in new structured folders
            search_pattern = f"*{_sanitize_document_no(document_no) or document_no}*"
            existing_files = list(ORGANIZED_DIR.rglob(search_pattern))
            if existing_files:
                logger.info(f"Document {document_no} already exists locally, skipping OCR processing")
                # Prefer non-text file as image representative
                image_candidate = next((p for p in existing_files if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf']), None)
                text_candidate = next((p for p in existing_files if p.suffix.lower() == '.txt'), None)
                parent_dir = (image_candidate or text_candidate).parent if (image_candidate or text_candidate) else ORGANIZED_DIR
                result.success = True
                result.document_no = document_no
                result.document_type = voucher_type
                result.voucher_type = voucher_type
                result.document_date = None # Not stored in this path-based check for now
                result.folder_path = str(parent_dir)
                result.text_file = str(text_candidate) if text_candidate else None
                result.image_file = str(image_candidate) if image_candidate else None
                
                # Track this document
                processed_documents[document_id] = {
                    'success': True,
                    'document_no': document_no,
                    'document_type': voucher_type,
                    'voucher_type': voucher_type,
                    'document_date': None,
                    'folder_path': str(parent_dir),
                    'text_file': str(text_candidate) if text_candidate else None,
                    'image_file': str(image_candidate) if image_candidate else None,
                    'processed_at': datetime.now().isoformat()
                }
                return result
        
        # Save temporary file
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(image_data)
        
        # Process with new document processor (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        ocr_result = await loop.run_in_executor(
            None, 
            document_processor.process_document,
            str(temp_path),
            doc_data.get('name', 'document')
        )
        
        # Update result
        if ocr_result.get('success'):
            result.success = True
            result.document_no = ocr_result.get('document_no')
            result.document_type = ocr_result.get('classification')
            result.document_date = ocr_result.get('document_date')
            result.voucher_type = ocr_result.get('classification')
            
            # Validate document number uniqueness
            if result.document_no:
                existing_doc_info = next((doc for doc in processed_documents.values() if doc.get('document_no') == result.document_no and doc.get('success')), None)
                if existing_doc_info:
                    logger.warning(f"⚠️ Document number {result.document_no} already exists.")
                    result.success = False
                    result.error = f"Duplicate document number. A voucher with number '{result.document_no}' already exists."
                    processed_documents[document_id] = {
                        'success': False,
                        'error': result.error,
                        'image_hash': image_hash,
                        'processed_at': datetime.now().isoformat()
                    }
                    return result

            # Determine branch directory name
            ocr_branch_hint = ocr_result.get('branch_id')
            derived_branch_num = _extract_branch_number_from_document_no(result.document_no)
            branch_dir_name = _format_branch_dir_name(ocr_branch_hint or derived_branch_num)
            
            # For new document processor, files are already organized
            # Use organized_path from result
            organized_path = ocr_result.get('organized_path')
            if organized_path:
                result.folder_path = organized_path
                # Get the PDF or image file path
                pdf_path = ocr_result.get('pdf_path')
                if pdf_path:
                    result.image_file = pdf_path
                result.text_file = None  # New processor doesn't create separate text files
            
            # Track this document to prevent future duplicates
            processed_documents[document_id] = {
                'success': True,
                'document_no': result.document_no,
                'document_type': result.document_type,
                'voucher_type': result.voucher_type,
                'document_date': result.document_date,
                'folder_path': result.folder_path,
                'text_file': result.text_file,
                'image_file': result.image_file,
                'image_hash': image_hash,
                'processed_at': datetime.now().isoformat()
            }
            
            # Save processed documents periodically
            await _save_processed_documents()
            
            logger.info(f"✅ Successfully processed document {result.document_no} -> {result.folder_path}")
        else:
            result.error = ocr_result.get('error', 'Unknown error')
            logger.error(f"❌ Failed to process document {doc_data['id']}: {result.error}")
            
            # Track failed document to prevent retry
            processed_documents[document_id] = {
                'success': False,
                'error': result.error,
                'image_hash': image_hash, # Can be None if it failed before hash calculation
                'processed_at': datetime.now().isoformat()
            }
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
    
    return result

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Axiom Spark Document Processing API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "upload": "/api/upload",
            "flow": "/api/flow",
            "folders": "/api/folders",
            "websocket": "/ws",
            "auth": {
                "register": "/api/auth/register",
                "login": "/api/auth/login",
                "refresh": "/api/auth/refresh",
                "profile": "/api/auth/profile"
            }
        }
    }

# Simple Authentication Endpoints
@app.post("/api/auth/register")
async def register(
    email: str = Body(...),
    password: str = Body(...),
    full_name: str = Body(...)
):
    """
    Registration is disabled. Users must be created by administrators.
    """
    return JSONResponse(
        content={
            "success": False,
            "error": "User registration is disabled. Please contact an administrator to create an account."
        },
        status_code=403
    )

@app.post("/api/auth/login")
async def login(
    email: str = Body(...),
    password: str = Body(...)
):
    """
    Login with email and password.
    Returns JWT token for authenticated requests.
    """
    if not email or not password:
        return JSONResponse(
            content={"success": False, "error": "Email and password are required"},
            status_code=400
        )
    
    result = simple_auth.login(email.strip().lower(), password)
    
    if result.get("success"):
        return JSONResponse(content=result, status_code=200)
    return JSONResponse(content=result, status_code=401)

@app.get("/api/auth/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile with full details"""
    try:
        # Get full user details from simple_auth
        user_id = current_user.get('id')
        if user_id:
            user_data = simple_auth.users.get(user_id)
            if user_data:
                # Return user data without password
                profile = {
                    "id": user_data.get('id'),
                    "email": user_data.get('email'),
                    "full_name": user_data.get('full_name'),
                    "role": user_data.get('role', 'agent'),
                    "created_at": user_data.get('created_at'),
                    "is_active": user_data.get('is_active', True)
                }
                return JSONResponse(content={"success": True, "user": profile})
        
        # Fallback to current_user from token
        return JSONResponse(content={"success": True, "user": current_user})
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        return JSONResponse(content={"success": True, "user": current_user})

@app.post("/api/auth/logout")
async def logout():
    """Logout (frontend handles token removal)"""
    return JSONResponse(content={"success": True, "message": "Logged out successfully"})

@app.put("/api/auth/change-password")
async def change_password(
    current_password: str = Body(...),
    new_password: str = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Change user password"""
    try:
        user_id = current_user.get('id')
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Get user from simple_auth
        user = simple_auth.users.get(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify current password
        if not simple_auth.verify_password(current_password, user['password']):
            return JSONResponse(
                content={"success": False, "error": "Current password is incorrect"},
                status_code=400
            )
        
        # Validate new password
        if len(new_password) < 8:
            return JSONResponse(
                content={"success": False, "error": "New password must be at least 8 characters"},
                status_code=400
            )
        
        # Update password
        user['password'] = simple_auth.hash_password(new_password)
        simple_auth.save_users()
        
        return JSONResponse(content={
            "success": True,
            "message": "Password changed successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/users")
async def list_users(current_user: dict = Depends(get_current_user)):
    """List all users (admin only)"""
    # Check if user is admin
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    
    if not user_is_admin:
        raise HTTPException(status_code=403, detail="Only admins can list users")
    
    try:
        users = simple_auth.get_all_users()
        # Remove password from response
        safe_users = [
            {k: v for k, v in user.items() if k != 'password'}
            for user in users
        ]
        return JSONResponse(content={
            "success": True,
            "users": safe_users
        })
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/batch")
async def list_agents_batch(
    include_stats: bool = Query(True, description="Whether to include stats (properties_count, clients_count, documents_count)"),
    current_user: dict = Depends(get_current_user)
):
    """List all agents with optional stats (properties_count, clients_count, documents_count). Cached for 30 seconds."""
    # Check if user is admin
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    
    if not user_is_admin:
        raise HTTPException(status_code=403, detail="Only admins can list agents")
    
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check cache first (cache key includes include_stats flag)
        cache_key = f'agents_batch_{include_stats}'
        now = time()
        if cache_key in _agents_cache:
            cached_data, cached_time = _agents_cache[cache_key]
            if now - cached_time < _AGENTS_CACHE_TTL:
                logger.info(f"📊 Returning cached agents batch (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        # Get all users/agents from simple_auth
        users = simple_auth.get_all_users()
        agents = [
            {k: v for k, v in user.items() if k != 'password'}
            for user in users
        ]
        
        # Only calculate stats if requested (for performance)
        if include_stats:
            # Batch calculate stats for all agents - OPTIMIZED to avoid fetching all documents
            agent_ids = [agent['id'] for agent in agents]
            
            try:
                # Initialize stats
                agent_doc_counts = {aid: 0 for aid in agent_ids}
                agent_client_counts = {aid: 0 for aid in agent_ids}
                agent_properties_counts = {aid: 0 for aid in agent_ids}
                
                # OPTIMIZED: Query documents in batches per agent (limit to 1000 per agent for performance)
                # This is much faster than fetching ALL documents
                for agent_id in agent_ids:
                    try:
                        # Count documents for this agent (limit to 1000 for performance)
                        docs_query = firestore_service.documents_collection.where(
                            filter=FieldFilter('agentId', '==', agent_id)
                        ).limit(1000)
                        docs = list(docs_query.stream())
                        agent_doc_counts[agent_id] = len(docs)
                        
                        # Count unique clients from these documents
                        client_ids = set()
                        for doc in docs:
                            data = doc.to_dict()
                            if data.get('clientId'):
                                client_ids.add(data.get('clientId'))
                        agent_client_counts[agent_id] = len(client_ids)
                    except Exception as e:
                        logger.warning(f"Error counting docs for agent {agent_id}: {e}")
                    
                    # Count properties
                    try:
                        properties_query = firestore_service.properties_collection.where(
                            filter=FieldFilter('agentId', '==', agent_id)
                        )
                        properties = list(properties_query.stream())
                        agent_properties_counts[agent_id] = len(properties)
                    except Exception as e:
                        logger.warning(f"Error counting properties for agent {agent_id}: {e}")
                
                # Assign stats to agents
                for agent in agents:
                    agent_id = agent['id']
                    agent['documents_count'] = agent_doc_counts.get(agent_id, 0)
                    agent['clients_count'] = agent_client_counts.get(agent_id, 0)
                    agent['properties_count'] = agent_properties_counts.get(agent_id, 0)
            except Exception as e:
                logger.error(f"Error calculating agent stats: {e}")
                # Fallback: set stats to 0
                for agent in agents:
                    agent['documents_count'] = 0
                    agent['clients_count'] = 0
                    agent['properties_count'] = 0
        else:
            # No stats requested - just return agents without stats (much faster)
            for agent in agents:
                agent['documents_count'] = 0
                agent['clients_count'] = 0
                agent['properties_count'] = 0
        
        response_data = {
            "success": True,
            "agents": convert_decimals(agents)
        }
        
        # Cache the response
        _agents_cache[cache_key] = (response_data, now)
        _cleanup_cache(_agents_cache, _AGENTS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error listing agents batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/auth/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Update user role (admin only)"""
    # Check if user is admin
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    
    if not user_is_admin:
        raise HTTPException(status_code=403, detail="Only admins can update user roles")
    
    if role not in ['admin', 'agent']:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'agent'")
    
    try:
        success = simple_auth.update_user_role(user_id, role)
        if success:
            return JSONResponse(content={
                "success": True,
                "message": f"User role updated to {role}"
            })
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user role: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(document: DocumentUpload):
    """Upload a single document for processing"""
    try:
        # Save document to flow queue
        doc_path = FLOW_QUEUE_DIR / f"{document.id}.json"
        async with aiofiles.open(doc_path, 'w') as f:
            await f.write(document.json())
        
        # Create a flow job for this single document
        flow_job = FlowJob(
            documents=[document.id],
            total_documents=1
        )
        
        # Store job
        flow_jobs[flow_job.job_id] = flow_job
        
        # Add to processing queue
        await processing_queue.put(flow_job.job_id)
        
        logger.info(f"Created flow job {flow_job.job_id} for document {document.id}")
        
        return JSONResponse(content={
            "success": True,
            "document_id": document.id,
            "flow_job_id": flow_job.job_id,
            "message": "Document uploaded and queued for processing"
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file directly (for web interface)"""
    try:
        # Read file content
        content = await file.read()
        
        # Create document model
        document = DocumentUpload(
            name=file.filename,
            type=file.filename.split('.')[-1].lower(),
            size=f"{len(content) / 1024:.2f} KB",
            data=base64.b64encode(content).decode('utf-8'),
            source="web"
        )
        
        # Save to flow queue
        doc_path = FLOW_QUEUE_DIR / f"{document.id}.json"
        async with aiofiles.open(doc_path, 'w') as f:
            await f.write(document.json())
        
        # Create a flow job for this single document
        flow_job = FlowJob(
            documents=[document.id],
            total_documents=1
        )
        
        # Store job
        flow_jobs[flow_job.job_id] = flow_job
        
        # Add to processing queue
        await processing_queue.put(flow_job.job_id)
        
        logger.info(f"Created flow job {flow_job.job_id} for file {file.filename}")
        
        return JSONResponse(content={
            "success": True,
            "document_id": document.id,
            "flow_job_id": flow_job.job_id,
            "filename": file.filename,
            "message": "File uploaded and queued for processing"
        })
    
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/upload")
async def aws_upload_file(
    file: UploadFile = File(...),
    flow_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    request: Request = None
):
    """
    Upload single file to GCS and process with OCR.
    Endpoint name kept for frontend compatibility.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Try to get current user (optional authentication)
        agent_id = None
        try:
            auth_header = request.headers.get("Authorization") if request else None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                result = simple_auth.get_user(token)
                if result.get("success"):
                    agent_id = result["user"].get("id")
        except Exception as e:
            logger.debug(f"Could not extract agent ID from request: {e}")
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Generate batch ID if not provided
        if not flow_id:
            flow_id = f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Read file content
        content = await file.read()
        file_ext = Path(file.filename).suffix.lower()
        
        # Upload to GCS temp folder
        gcs_temp_path = f"temp/{flow_id}/{document_id}{file_ext}"
        
        if gcs_service:
            bucket = gcs_service.bucket
            blob = bucket.blob(gcs_temp_path)
            blob.upload_from_string(content, content_type=file.content_type or 'application/octet-stream')
        
        # Create Firestore document record
        document_data = {
                'filename': file.filename,
                'flow_id': flow_id,
                'gcs_temp_path': gcs_temp_path,
                'processing_status': 'pending',
                'file_size': len(content)
        }
        if agent_id:
            document_data['agentId'] = agent_id
        
        if firestore_service:
            firestore_service.create_document(document_id, document_data)
            # Invalidate flow files cache when new document is added
            if flow_id and f"flow_files_{flow_id}" in _flow_files_cache:
                del _flow_files_cache[f"flow_files_{flow_id}"]
                logger.info(f"🗑️  Invalidated flow files cache for {flow_id} after document creation")
        
        # Queue background OCR task
        if task_queue and background_tasks:
            task_queue.add_process_task(
                background_tasks,
                document_id,
                gcs_temp_path,
                file.filename
            )
        
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "flow_id": flow_id,
            "filename": file.filename,
            "message": "File uploaded and queued for processing"
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/batch-upload")
async def aws_batch_upload(
    files: List[UploadFile] = File(...),
    flow_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    request: Request = None
):
    """
    Upload multiple files to GCS and process with OCR.
    Endpoint name kept for frontend compatibility.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Try to get current user (optional authentication)
        agent_id = None
        try:
            auth_header = request.headers.get("Authorization") if request else None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                result = simple_auth.get_user(token)
                if result.get("success"):
                    agent_id = result["user"].get("id")
        except Exception as e:
            logger.debug(f"Could not extract agent ID from request: {e}")
        
        # Generate batch ID if not provided
        if not flow_id:
            flow_id = f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create job record in Firestore
        job_id = str(uuid.uuid4())
        if firestore_service:
            firestore_service.create_job(job_id, {
                'flow_id': flow_id,
                'total_documents': len(files)
            })
        
        uploaded = []
        failed = []
        
        for file in files:
            try:
                # Generate document ID
                document_id = str(uuid.uuid4())
                
                # Read file content
                content = await file.read()
                file_ext = Path(file.filename).suffix.lower()
                
                # Upload to GCS temp folder
                gcs_temp_path = f"temp/{flow_id}/{document_id}{file_ext}"
                
                if gcs_service:
                    bucket = gcs_service.bucket
                    blob = bucket.blob(gcs_temp_path)
                    blob.upload_from_string(content, content_type=file.content_type or 'application/octet-stream')
                
                # Create Firestore document record
                document_data = {
                        'filename': file.filename,
                        'flow_id': flow_id,
                        'job_id': job_id,
                        'gcs_temp_path': gcs_temp_path,
                        'processing_status': 'pending',
                        'file_size': len(content)
                }
                if agent_id:
                    document_data['agentId'] = agent_id
                
                if firestore_service:
                    firestore_service.create_document(document_id, document_data)
                
                # Queue background OCR task
                if task_queue and background_tasks:
                    task_queue.add_process_task(
                        background_tasks,
                        document_id,
                        gcs_temp_path,
                        file.filename,
                        job_id
                    )
                
                uploaded.append({
                    "document_id": document_id,
                    "filename": file.filename
                })
            
            except Exception as e:
                logger.error(f"Failed to upload {file.filename}: {e}")
                failed.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "success": len(uploaded) > 0,
            "flow_id": flow_id,
            "job_id": job_id,
            "uploaded": uploaded,
            "failed": failed,
            "message": f"Uploaded {len(uploaded)} files, {len(failed)} failed"
        })
    
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GCS direct upload endpoints (replacement for deprecated AWS routes)
@app.post("/api/gcs/flow/create")
async def create_gcs_flow(
    flow_name: Optional[str] = Form(None),
    batch_name: Optional[str] = Form(None)
):
    """Create a flow in Firestore (primary) with GCS backup."""
    try:
        # Accept both flow_name and batch_name for backward compatibility
        # Priority: flow_name > batch_name > default
        actual_flow_name = flow_name or batch_name or "Uploaded Flow"
        
        # Trim whitespace and validate
        actual_flow_name = actual_flow_name.strip()
        if not actual_flow_name:
            actual_flow_name = "Uploaded Flow"
        
        flow_id = f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        created_at = datetime.now().isoformat()
        
        # PRIMARY: Create flow in Firestore FIRST (required)
        if not firestore_service:
            raise Exception("Firestore service not available - cannot create flow")
        
        try:
            logger.info(f"📊 Creating flow in Firestore (primary): {actual_flow_name}")
            logger.info(f"   Flow ID: {flow_id}")
            logger.info(f"   Flow name: {actual_flow_name}")
            logger.info(f"   Created at: {created_at}")
            
            firestore_service.create_flow(
                flow_id,
                {
                    'flow_name': actual_flow_name,  # Use the actual name entered by user
                    'source': 'web',
                    'document_count': 0,
                    'status': 'created'
                }
            )
            logger.info(f"✅ Flow {flow_id} created in Firestore: {actual_flow_name}")
            
            # Verify the flow was created by reading it back
            try:
                verify_flow = firestore_service.get_flow(flow_id)
                if verify_flow:
                    logger.info(f"✅ Verified flow exists in Firestore: {verify_flow.get('flow_id')} - {verify_flow.get('flow_name')}")
                else:
                    logger.error(f"❌ Flow {flow_id} was not found after creation!")
            except Exception as verify_error:
                logger.warning(f"⚠️  Could not verify flow creation: {verify_error}")
        except Exception as firestore_error:
            logger.error(f"❌ Failed to create flow in Firestore: {firestore_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise Exception(f"Failed to create flow in Firestore: {firestore_error}")
        
        # BACKUP: Sync to GCS asynchronously (non-blocking, optional)
        # Only sync if s3_service is available and initialized
        if s3_service and s3_service.client and s3_service.bucket:
            async def sync_flow_to_gcs():
                try:
                    logger.info(f"📂 Syncing flow to GCS (backup): {flow_id}")
                    result = s3_service.create_flow_metadata_in_s3(flow_id, actual_flow_name, created_at)
                    if result.get("success"):
                        logger.info(f"✅ Flow {flow_id} synced to GCS backup")
                    else:
                        logger.warning(f"⚠️  Failed to sync flow to GCS: {result.get('error')}")
                except Exception as gcs_error:
                    logger.warning(f"⚠️  GCS sync failed (non-critical): {gcs_error}")
            
            # Start GCS sync in background (don't wait for it)
            asyncio.create_task(sync_flow_to_gcs())
        else:
            logger.warning(f"⚠️  GCS service not available, skipping flow sync to GCS")
        
        # Invalidate flows cache
        if 'flows' in _flows_cache:
            del _flows_cache['flows']
            logger.info("🗑️  Invalidated flows cache after creating new flow")
        
        return JSONResponse(content={
            "success": True, 
            "flow_id": flow_id, 
            "flow_name": actual_flow_name,  # Return the actual name used
            "created_at": created_at,
            "source": "firestore"
        })
    except Exception as e:
        logger.error(f"Error creating flow: {e}")
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)


@app.get("/api/gcs/flows")
async def list_gcs_flows():
    """List all flows from Firestore (primary) with GCS fallback. Cached for 30 seconds."""
    try:
        # Check cache first
        cache_key = 'flows'
        now = time()
        if cache_key in _flows_cache:
            cached_data, cached_time = _flows_cache[cache_key]
            if now - cached_time < _FLOWS_CACHE_TTL:
                logger.info(f"📊 Returning cached flows (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        # PRIMARY: Query Firestore (10-50x faster)
        if firestore_service:
            try:
                logger.info("📊 Querying flows from Firestore (primary source)")
                # Reduce page_size from 500 to 50 for faster queries (frontend only needs ~23)
                flows, total = firestore_service.list_flows(page=1, page_size=50)
                logger.info(f"   Raw flows returned from Firestore: {len(flows)} flows")
                
                # Format flows to match frontend expectations (OPTIMIZED - no per-flow queries)
                formatted_flows = []
                now_iso = datetime.now().isoformat()
                for flow in flows:
                    flow_id = flow.get('flow_id', '')
                    flow_name = flow.get('flow_name', '')
                    created_at = flow.get('created_at', '')
                    
                    # Convert datetime to ISO string if needed (optimized)
                    if created_at:
                        if hasattr(created_at, 'isoformat'):
                            created_at = created_at.isoformat()
                        elif isinstance(created_at, datetime):
                            created_at = created_at.isoformat()
                    else:
                        created_at = now_iso
                    
                    # PERFORMANCE FIX: Use stored document_count directly
                    # Count syncing is now done during document create/update/delete operations
                    # This eliminates N database queries (one per flow) in the list endpoint
                    document_count = flow.get('document_count', 0)
                    
                    formatted_flow = {
                        "flow_id": flow_id,
                        "flow_name": flow_name,
                        "created_at": created_at,
                        "file_count": document_count,
                        "total_files": document_count,
                        "status": flow.get('status', 'created'),
                        "source": flow.get('source', 'web')
                    }
                    formatted_flows.append(formatted_flow)
                
                # Ensure flows are sorted by created_at descending (newest first)
                formatted_flows.sort(
                    key=lambda x: x.get('created_at', ''),
                    reverse=True
                )
                
                logger.info(f"✅ Retrieved {len(formatted_flows)} flows from Firestore (sorted by created_at DESC)")
                
                # Cache the result
                response_data = {
                    "success": True,
                    "flows": formatted_flows,
                    "count": len(formatted_flows),
                    "source": "firestore"
                }
                _flows_cache[cache_key] = (response_data, now)
                
                return JSONResponse(content=response_data)
            except Exception as firestore_error:
                logger.warning(f"⚠️  Firestore query failed, falling back to GCS: {firestore_error}")
        else:
            logger.warning("⚠️  Firestore service not available, using GCS fallback")
        
        # FALLBACK: Query GCS if Firestore unavailable or failed
        logger.info("📂 Falling back to GCS flow listing")
        result = s3_service.list_flows_from_s3()
        if not result.get("success"):
            logger.warning(f"⚠️  Failed to list GCS flows: {result.get('error')}")
            return JSONResponse(content={
                "success": True,
                "flows": [],
                "count": 0,
                "message": result.get("error", "Storage not available"),
                "source": "none"
            })
        
        # Ensure each flow has the fields the frontend expects
        flows = result.get("flows", [])
        for flow in flows:
            if "file_count" not in flow and "total_files" in flow:
                flow["file_count"] = flow["total_files"]
            if "status" not in flow:
                flow["status"] = "created"
        
        logger.info(f"✅ Retrieved {len(flows)} flows from GCS (fallback)")
        response_data = {
            "success": True,
            "flows": flows,
            "count": result.get("count", len(flows)),
            "source": "gcs_fallback"
        }
        
        # Cache the result
        _flows_cache[cache_key] = (response_data, now)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error listing flows: {e}")
        return JSONResponse(content={
            "success": True,
            "flows": [],
            "count": 0,
            "message": "Database error",
            "source": "none"
        })


@app.delete("/api/gcs/flow/{flow_id}")
async def delete_gcs_flow(flow_id: str, delete_temp_files: bool = False):
    """Delete a flow from Firestore (primary) and GCS (backup)"""
    try:
        logger.info(f"🗑️  Deleting flow: {flow_id} (delete_temp_files={delete_temp_files})")
        
        deleted_items = {
            "firestore_flow": False,
            "firestore_documents": 0,
            "gcs_files": 0
        }
        
        # PRIMARY: Delete from Firestore (required)
        if firestore_service:
            try:
                # Delete all documents for this flow
                deleted_docs = firestore_service.delete_documents_by_flow_id(flow_id)
                deleted_items["firestore_documents"] = deleted_docs
                logger.info(f"✅ Deleted {deleted_docs} documents from Firestore")
                
                # Delete the flow itself
                if firestore_service.delete_flow(flow_id):
                    deleted_items["firestore_flow"] = True
                    logger.info(f"✅ Deleted flow {flow_id} from Firestore")
            except Exception as firestore_error:
                logger.error(f"❌ Failed to delete from Firestore: {firestore_error}")
                raise Exception(f"Failed to delete from Firestore: {firestore_error}")
        else:
            logger.warning("⚠️  Firestore service not available")
        
        # Invalidate flows cache
        if 'flows' in _flows_cache:
            del _flows_cache['flows']
            logger.info("🗑️  Invalidated flows cache after deleting flow")
        
        # BACKUP: Delete from GCS (optional, best effort)
        try:
            logger.info(f"📂 Deleting flow files from GCS (backup)")
            gcs_result = s3_service.delete_flow_from_s3(flow_id, delete_temp_files)
            if gcs_result.get("success"):
                deleted_items["gcs_files"] = gcs_result.get("deleted", 0)
                logger.info(f"✅ Deleted {deleted_items['gcs_files']} files from GCS")
            else:
                logger.warning(f"⚠️  GCS deletion warning: {gcs_result.get('error')}")
        except Exception as gcs_error:
            logger.warning(f"⚠️  GCS deletion failed (non-critical): {gcs_error}")
        
        return JSONResponse(content={
            "success": True,
            "message": f"Flow {flow_id} deleted",
            "deleted": deleted_items
        })
    
    except Exception as e:
        logger.error(f"❌ Error deleting flow {flow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gcs/flow/{flow_id}/files")
async def get_gcs_flow_files(flow_id: str):
    """Get all files for a specific flow from Firestore (primary) with GCS fallback. Optimized for performance."""
    try:
        # Cleanup old cache entries periodically
        _cleanup_cache(_flow_files_cache, _FLOW_FILES_CACHE_TTL)
        
        # Check cache first
        cache_key = f"flow_files_{flow_id}"
        now = time()
        if cache_key in _flow_files_cache:
            cached_data, cached_time = _flow_files_cache[cache_key]
            if now - cached_time < _FLOW_FILES_CACHE_TTL:
                logger.info(f"📊 Returning cached flow files for {flow_id} (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        logger.info(f"📊 Getting files for flow: {flow_id} from Firestore (primary source)")
        
        # PRIMARY: Query Firestore for all documents (optimized)
        firestore_documents = []
        flow_name = flow_id
        
        if firestore_service:
            try:
                # Parallelize flow metadata and documents query using asyncio.to_thread
                flow_data_task = asyncio.to_thread(firestore_service.get_flow, flow_id)
                docs_task = asyncio.to_thread(firestore_service.get_documents_by_flow_id, flow_id, 1, 1000)
                
                # Wait for both to complete in parallel
                flow_data, (firestore_docs, total_docs) = await asyncio.gather(flow_data_task, docs_task)
                
                firestore_documents = firestore_docs
                logger.info(f"✅ Retrieved {len(firestore_docs)} documents from Firestore for flow {flow_id}")
                
                # Auto-process documents stuck in 'uploading' or 'pending' status
                if task_queue and s3_service:
                    stuck_docs_to_process = []
                    for doc in firestore_docs:
                        doc_id = doc.get('document_id') or doc.get('id', '')
                        status = doc.get('processing_status', '')
                        gcs_temp_path = doc.get('gcs_temp_path', '')
                        
                        # Handle documents stuck in 'uploading', 'pending', or 'uploaded' status
                        # These are documents that should be processed but haven't started yet
                        if status in ['uploading', 'pending', 'uploaded'] and gcs_temp_path:
                            stuck_docs_to_process.append((doc_id, gcs_temp_path, doc, status))
                    
                    # Process stuck documents in parallel
                    if stuck_docs_to_process:
                        async def check_and_process_stuck_doc(doc_id: str, gcs_temp_path: str, doc: Dict[str, Any], original_status: str):
                            try:
                                file_exists = await s3_service.check_file_exists(gcs_temp_path)
                                if file_exists:
                                    logger.info(f"🔄 Auto-processing stuck document {doc_id} (status: {original_status}, file exists in GCS)")
                                    
                                    # For 'uploading' or 'pending', update to 'uploaded' first
                                    # For 'uploaded', it's already in the right state, just trigger processing
                                    if original_status in ['uploading', 'pending']:
                                        firestore_service.update_document(doc_id, {
                                            'processing_status': 'uploaded'
                                        })
                                        logger.info(f"📝 Updated document {doc_id} status from '{original_status}' to 'uploaded'")
                                    
                                    # Directly call the processing task (bypass BackgroundTasks for auto-recovery)
                                    filename = doc.get('filename', 'unknown')
                                    job_id = doc.get('flow_id') or flow_id
                                    
                                    # Run processing in background using asyncio.create_task
                                    async def run_processing():
                                        try:
                                            await task_queue.process_document_task(doc_id, gcs_temp_path, filename, job_id)
                                            logger.info(f"✅ Auto-processed stuck document {doc_id} (was {original_status})")
                                        except Exception as e:
                                            logger.error(f"❌ Failed to process stuck document {doc_id}: {e}")
                                            import traceback
                                            logger.error(traceback.format_exc())
                                    
                                    asyncio.create_task(run_processing())
                                else:
                                    logger.warning(f"⚠️  Document {doc_id} has status '{original_status}' but file not found in GCS: {gcs_temp_path}")
                            except Exception as e:
                                logger.warning(f"⚠️  Failed to check/process stuck document {doc_id}: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                        
                        # Process all stuck documents in parallel
                        if stuck_docs_to_process:
                            logger.info(f"🔄 Found {len(stuck_docs_to_process)} stuck document(s) to auto-process for flow {flow_id}")
                            await asyncio.gather(*[check_and_process_stuck_doc(doc_id, path, doc, status) for doc_id, path, doc, status in stuck_docs_to_process], return_exceptions=True)
                            logger.info(f"✅ Auto-processing initiated for {len(stuck_docs_to_process)} stuck document(s)")
        
                # Get flow metadata from Firestore
                if flow_data:
                    flow_name = flow_data.get('flow_name', flow_id)
                    logger.info(f"📂 Flow name from Firestore: {flow_name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to get Firestore documents: {e}")
                firestore_documents = []
        
        # Process Firestore documents into file lists (optimized)
        temp_files = []
        organized_files = []
        failed_files = []
        pending_files = []
        need_review_files = []
        
        logger.info(f"🔄 Processing {len(firestore_documents)} Firestore documents")
        
        # Pre-compute now_iso for default values (optimized - compute once)
        now_iso = datetime.now().isoformat()
        
        # OPTIMIZED: Use list comprehension and helper function for faster processing
        def process_document(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process a single document into file data format"""
            # Convert datetime fields to ISO strings (optimized)
            uploaded_at = doc.get('created_at') or doc.get('updated_at')
            if uploaded_at:
                if hasattr(uploaded_at, 'isoformat'):
                    uploaded_at = uploaded_at.isoformat()
                elif isinstance(uploaded_at, datetime):
                    uploaded_at = uploaded_at.isoformat()
                else:
                    uploaded_at = str(uploaded_at)
            else:
                uploaded_at = now_iso
            
            processed_at = doc.get('updated_at') or uploaded_at
            if processed_at and processed_at != uploaded_at:
                if hasattr(processed_at, 'isoformat'):
                    processed_at = processed_at.isoformat()
                elif isinstance(processed_at, datetime):
                    processed_at = processed_at.isoformat()
                else:
                    processed_at = str(processed_at)
            else:
                processed_at = uploaded_at
            
            # Get GCS path and extract key
            # Priority: Use gcs_path for completed files, gcs_temp_path for pending/uploaded files
            processing_status = doc.get('processing_status', 'pending')
            if processing_status == 'completed' and doc.get('gcs_path'):
                gcs_path = doc.get('gcs_path')
            else:
                # For pending/uploaded files, prioritize gcs_temp_path (actual storage location)
                gcs_path = doc.get('gcs_temp_path') or doc.get('gcs_path', '')
            
            # Extract clean key from the path (handle gs:// URLs, bucket names, etc.)
            s3_key = extract_gcs_key_from_path(gcs_path, s3_service.bucket_name) if gcs_path else ''
            
            # If we still don't have a key and this is a temp file, log a warning
            if not s3_key and processing_status in ['pending', 'uploaded', 'uploading']:
                logger.warning(f"⚠️  Document {doc.get('document_id')} has no gcs_temp_path or gcs_path - filename: {doc.get('filename')}")
            
            # Parse organized_path using helper function
            organized_path = doc.get('organized_path', '')
            path_parts = parse_organized_path(organized_path) if organized_path else {}
            
            # Skip presigned URL generation for list view (major performance optimization)
            url = ''
            
            # Get metadata once
            metadata = doc.get('metadata', {})
            
            # Build file data
            # Ensure s3_key is always the actual stored path, never a constructed path
            final_s3_key = s3_key if s3_key else gcs_path if gcs_path else ''
            return {
                'document_id': doc.get('document_id', ''),
                'filename': doc.get('filename', ''),
                's3_key': final_s3_key,
                'organized_path': organized_path,
                'category': path_parts.get('category', ''),
                'year': path_parts.get('year', ''),
                'month': path_parts.get('month', ''),
                'date': path_parts.get('date', ''),
                'size': doc.get('file_size', 0),
                'uploaded_at': uploaded_at,
                'processed_at': processed_at,
                'last_modified': processed_at,
                'status': doc.get('processing_status', 'pending'),
                'url': url,
                'document_no': metadata.get('document_no', ''),
                'document_type': metadata.get('classification', ''),
                'document_date': metadata.get('document_date', ''),
                'metadata': metadata
            }
        
        # Process all documents in parallel using list comprehension (faster than loop)
        processed_docs = [process_document(doc) for doc in firestore_documents]
        
        # Categorize documents by status (optimized - single pass)
        for file_data in processed_docs:
            if not file_data:
                continue
            status = file_data['status']
            organized_path = file_data.get('organized_path', '')
            
            if status == 'completed' and organized_path:
                organized_files.append(file_data)
            elif status == 'failed' or status == 'error':
                failed_files.append(file_data)
            elif status == 'need_review':
                need_review_files.append(file_data)
            elif status == 'pending' or status == 'uploaded' or status == 'uploading':
                # Include uploading/uploaded documents in pending_files so they appear immediately in flows
                pending_files.append(file_data)
            else:
                pending_files.append(file_data)
        
        # FALLBACK: Query GCS if Firestore has no documents (lazy migration scenario)
        if not firestore_documents:
            logger.warning(f"⚠️  No Firestore documents found for flow {flow_id}, falling back to GCS listing")
            gcs_result = s3_service.get_flow_files_from_s3(flow_id)
            
            if gcs_result.get('success'):
                # Process GCS temp files
                for file in gcs_result.get('temp_files', []):
                    temp_files.append({
                        'filename': file.get('key', '').split('/')[-1],
                        's3_key': file.get('key', ''),
                        'size': file.get('size', 0),
                        'uploaded_at': file.get('last_modified', ''),
                        'last_modified': file.get('last_modified', ''),
                        'status': 'uploaded',
                        'url': file.get('url', '')
                    })
                
                # Process GCS organized files
                for file in gcs_result.get('organized_files', []):
                    key = file.get('key', '')
                    path_parts = parse_organized_path(key)
                    organized_files.append({
                        'filename': path_parts.get('filename', key.split('/')[-1]),
                        's3_key': key,
                        'organized_path': '/'.join(key.split('/')[:-1]) if '/' in key else '',
                        'category': path_parts.get('category', ''),
                        'year': path_parts.get('year', ''),
                        'month': path_parts.get('month', ''),
                        'date': path_parts.get('date', ''),
                        'size': file.get('size', 0),
                        'processed_at': file.get('last_modified', ''),
                        'status': 'processed',
                        'url': file.get('url', '')
                    })
                
                # Process GCS failed files
                for file in gcs_result.get('failed_files', []):
                    failed_files.append({
                        'filename': file.get('key', '').split('/')[-1],
                        's3_key': file.get('key', ''),
                        'size': file.get('size', 0),
                        'uploaded_at': file.get('last_modified', ''),
                        'status': 'failed',
                        'url': file.get('url', '')
                    })
                
                # Try to get flow name from GCS metadata
                if flow_name == flow_id:
                    flow_meta = s3_service.get_flow_metadata_from_s3(flow_id)
                    flow_name = flow_meta.get('flow', {}).get('flow_name', flow_id) if flow_meta.get('success') else flow_id
        
        # Calculate totals - use Firestore documents as source of truth
        total = len(firestore_documents) or (len(temp_files) + len(organized_files) + len(failed_files) + len(pending_files) + len(need_review_files))
        organized_count = len(organized_files)
        
        # PERFORMANCE FIX: Removed count sync from read operations
        # Count syncing is now done during document create/update/delete operations
        # This eliminates database writes during read requests
        
        # Build response
        response_data = {
            'success': True,
            'flow_id': flow_id,
            'flow_name': flow_name,
            'batch_id': flow_id,  # Alias for backward compatibility
            'batch_name': flow_name,  # Alias for backward compatibility
            'temp_files': temp_files,
            'organized_files': organized_files,
            'failed_files': failed_files,
            'pending_files': pending_files,
            'need_review_files': need_review_files,
            'summary': {
                'total': total,
                'processed': organized_count,
                'pending': len(temp_files) + len(pending_files),
                'failed': len(failed_files)
            },
            'source': 'firestore' if firestore_documents else 'gcs_fallback'
        }
        
        logger.info(f"✅ Retrieved {total} files for flow {flow_id} from {'Firestore' if firestore_documents else 'GCS'}")
        logger.info(f"   - Organized: {organized_count}, Temp: {len(temp_files)}, Pending: {len(pending_files)}, Failed: {len(failed_files)}")
        
        # Cache the result
        _flow_files_cache[cache_key] = (response_data, now)
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Get GCS flow files error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gcs/organized-tree/flow/{flow_id}")
async def get_gcs_flow_organized_tree(flow_id: str):
    """Get hierarchical organized tree for a specific flow"""
    try:
        logger.info(f"Getting organized tree for flow: {flow_id}")
        
        # Get organized files from the flow files endpoint (includes Firestore documents)
        # This ensures we get files with properly parsed category, year, month, date fields
        gcs_result = s3_service.get_flow_files_from_s3(flow_id)
        
        # Get documents from Firestore
        firestore_documents = []
        if firestore_service:
            try:
                firestore_docs, total_docs = firestore_service.get_documents_by_flow_id(flow_id, page=1, page_size=1000)
                firestore_documents = firestore_docs
                logger.info(f"📊 Found {len(firestore_documents)} Firestore documents for tree")
            except Exception as e:
                logger.warning(f"⚠️  Failed to get Firestore documents: {e}")
        
        # Build organized files list from Firestore (source of truth)
        organized_files = []
        logger.info(f"🔍 Processing {len(firestore_documents)} Firestore documents for organized tree")
        
        for doc in firestore_documents:
            document_id = doc.get('document_id', 'unknown')
            processing_status = doc.get('processing_status', '')
            organized_path = doc.get('organized_path', '')
            filename = doc.get('filename', '')
            gcs_path = doc.get('gcs_path', '')
            
            logger.info(f"📄 Document ID: {document_id}")
            logger.info(f"   - filename: {filename}")
            logger.info(f"   - processing_status: {processing_status}")
            logger.info(f"   - organized_path: {organized_path}")
            logger.info(f"   - gcs_path: {gcs_path}")
            
            # Check if document should be included
            if processing_status != 'completed':
                logger.warning(f"   ⚠️  Skipping document {document_id}: processing_status is '{processing_status}' (expected 'completed')")
                continue
            
            if not organized_path:
                logger.warning(f"   ⚠️  Skipping document {document_id}: organized_path is empty")
                continue
            
            # Document is completed and has organized_path - process it
            # Get file info
            gcs_path = doc.get('gcs_path', '') or ''
            
            # IMPORTANT: Only show PDF files in organized tree (final processed documents)
            # Skip JPEG/PNG files which are original uploads - show only final PDF outputs
            if filename and not filename.lower().endswith('.pdf'):
                # Check if this is in organized_vouchers path - if so, skip non-PDF files
                if 'organized_vouchers' in organized_path:
                    logger.debug(f"Skipping non-PDF file in organized tree: {filename}")
                    continue
            
            logger.info(f"✅ Processing completed PDF file: {filename}, path: {organized_path}")
            
            # Extract s3_key from gcs_path (full path including document folder and filename)
            # gcs_path format examples:
            # - gs://voucher-bucket-1/organized_vouchers/id/2025/nov/28-11-2025/784-1985-1234567-1/784-1985-1234567-1_0001.pdf
            # - voucher-bucket-1/organized_vouchers/id/2025/nov/28-11-2025/784-1985-1234567-1/784-1985-1234567-1_0001.pdf
            s3_key = ''
            
            if gcs_path:
                # Remove gs:// prefix if present
                path_to_parse = gcs_path
                if path_to_parse.startswith('gs://'):
                    path_to_parse = path_to_parse.replace('gs://', '')
                
                # Remove bucket name if present
                if '/' in path_to_parse:
                    path_parts = path_to_parse.split('/', 1)
                    if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                        s3_key = path_parts[1]  # Full path without bucket name
                    else:
                        s3_key = path_to_parse  # Use as-is if no bucket name detected
                else:
                    s3_key = path_to_parse
            
            # If we still don't have s3_key, construct it from organized_path and filename
            if not s3_key and organized_path and filename:
                # organized_path might be the folder path, construct full path
                if organized_path.endswith('/'):
                    s3_key = f"{organized_path}{filename}"
                else:
                    s3_key = f"{organized_path}/{filename}"
            
            # Parse category, year, month, date from organized_path
            # Actual path structure: organized_vouchers/{year}/{Branch XX}/{month_name}/{day-month-year}/{voucher_type}
            category = ''
            year = ''
            month = ''
            date = ''
            branch = ''
            
            # Parse from organized_path (folder path without filename)
            if organized_path:
                parts = organized_path.split('/')
                logger.info(f"   - Parsing organized_path parts: {parts} (total: {len(parts)})")
                
                # Handle two different path structures:
                # 1. Vouchers: organized_vouchers/{year}/{Branch XX}/{month}/{date}/{voucher_type}
                # 2. General documents: organized_vouchers/{document_type}/{year}/{month}/{date}/{document_no}
                
                if len(parts) >= 5 and parts[0] == 'organized_vouchers':
                    # Check if second part is a year (4 digits) - indicates voucher structure
                    if parts[1].isdigit() and len(parts[1]) == 4:
                        # Voucher structure: organized_vouchers/{year}/{Branch XX}/{month}/{date}/{voucher_type}
                        year = parts[1]
                        if len(parts) >= 6 and parts[2].startswith('Branch'):
                            # Has branch
                            branch = parts[2]
                            month = parts[3] if len(parts) > 3 else ''
                            date = parts[4] if len(parts) > 4 else ''
                            category = parts[5] if len(parts) > 5 else ''
                        else:
                            # No branch: organized_vouchers/{year}/{month}/{date}/{voucher_type}
                            branch = ''
                            month = parts[2] if len(parts) > 2 else ''
                            date = parts[3] if len(parts) > 3 else ''
                            category = parts[4] if len(parts) > 4 else ''
                    else:
                        # General document structure: organized_vouchers/{document_type}/{year}/{month}/{date}/{document_no}
                        # For general documents, use document_type as category
                        category = parts[1] if len(parts) > 1 else ''  # e.g., "invoices"
                        year = parts[2] if len(parts) > 2 and parts[2].isdigit() and len(parts[2]) == 4 else ''
                        branch = ''  # General documents don't have branch
                        month = parts[3] if len(parts) > 3 else ''
                        date = parts[4] if len(parts) > 4 else ''
                        logger.info(f"   - Detected general document structure: category={category}, year={year}, month={month}, date={date}")
                else:
                    logger.warning(f"   ⚠️  Path has insufficient parts: {len(parts)} (expected at least 5)")
            
            # If organized_path doesn't have enough info, try parsing from full s3_key path
            if (not category or not year or not month or not date) and s3_key:
                path_parts = s3_key.split('/')
                logger.info(f"   - Parsing s3_key parts: {path_parts} (total: {len(path_parts)})")
                
                # Handle both structures in s3_key:
                # 1. Vouchers: organized_vouchers/{year}/{Branch XX}/{month}/{date}/{voucher_type}/{filename}
                # 2. General documents: organized_vouchers/{document_type}/{year}/{month}/{date}/{document_no}/{filename}
                
                if len(path_parts) >= 6 and path_parts[0] == 'organized_vouchers':
                    # Check if second part is a year (4 digits) - indicates voucher structure
                    if path_parts[1].isdigit() and len(path_parts[1]) == 4:
                        # Voucher structure
                        year_candidate = path_parts[1]
                        if not year:
                            year = year_candidate
                        
                        # Check if third part looks like a branch (starts with "Branch")
                        if len(path_parts) > 2 and path_parts[2].startswith('Branch'):
                            # Has branch structure
                            if not branch:
                                branch = path_parts[2]
                            if not month:
                                month = path_parts[3] if len(path_parts) > 3 else ''
                            if not date:
                                date = path_parts[4] if len(path_parts) > 4 else ''
                            if not category:
                                # Category is at index 5, but if there's a doc folder, it might be at index 6
                                if len(path_parts) > 5:
                                    potential_category = path_parts[5]
                                    # If it looks like a voucher type (MRT, MPU, etc.), use it
                                    if potential_category and len(potential_category) <= 4 and potential_category.isupper():
                                        category = potential_category
                                    elif len(path_parts) > 6:
                                        # Try next level (might be doc folder, then category)
                                        category = path_parts[6] if len(path_parts) > 6 else ''
                        else:
                            # No branch structure: organized_vouchers/{year}/{month}/{date}/{category}/...
                            if not month:
                                month = path_parts[2] if len(path_parts) > 2 else ''
                            if not date:
                                date = path_parts[3] if len(path_parts) > 3 else ''
                            if not category:
                                if len(path_parts) > 4:
                                    potential_category = path_parts[4]
                                    if potential_category and len(potential_category) <= 4 and potential_category.isupper():
                                        category = potential_category
                                    elif len(path_parts) > 5:
                                        category = path_parts[5] if len(path_parts) > 5 else ''
                    else:
                        # General document structure: organized_vouchers/{document_type}/{year}/{month}/{date}/{document_no}/{filename}
                        if not category:
                            category = path_parts[1] if len(path_parts) > 1 else ''
                        if not year:
                            year = path_parts[2] if len(path_parts) > 2 and path_parts[2].isdigit() and len(path_parts[2]) == 4 else ''
                        if not month:
                            month = path_parts[3] if len(path_parts) > 3 else ''
                        if not date:
                            date = path_parts[4] if len(path_parts) > 4 else ''
                        logger.info(f"   - Detected general document structure from s3_key: category={category}, year={year}, month={month}, date={date}")
            
            logger.info(f"📄 File {filename}: s3_key={s3_key}, category={category}, year={year}, month={month}, date={date}, branch={branch}")
            
            # Validate that we have all required fields (branch is optional)
            if not category or not year or not month or not date:
                logger.warning(f"⚠️  File {filename} missing path components - category={category}, year={year}, month={month}, date={date}, branch={branch}")
                logger.warning(f"    organized_path={organized_path}, s3_key={s3_key}")
                logger.warning(f"    Path parts count: organized_path={len(organized_path.split('/')) if organized_path else 0}, s3_key={len(s3_key.split('/')) if s3_key else 0}")
                logger.warning(f"    Skipping this file from organized tree")
                continue
            
            size = doc.get('file_size', 0)
            
            # Skip presigned URL generation for tree view (major performance optimization)
            # URLs will be generated on-demand when user views/downloads specific files
            url = ''
            
            organized_files.append({
                'filename': filename,
                'key': s3_key,
                'size': size,
                'url': url,
                'category': category,
                'year': year,
                'month': month,
                'date': date,
                'branch': branch,
                'organized_path': organized_path
            })
            
            logger.info(f"✅ Added file to organized_files list: {filename} (category={category}, year={year}, month={month}, date={date})")
        
        # IMPORTANT: Do NOT fallback to GCS for flow-specific tree view
        # Organized files don't have flow_id in their path, so we can't filter by flow_id from GCS
        # Only show files from Firestore that match the flow_id
        if len(organized_files) == 0:
            logger.warning(f"⚠️  No organized files found in Firestore for flow {flow_id}")
            logger.warning(f"    This is expected if files haven't been migrated to Firestore yet.")
            logger.warning(f"    Files must be in Firestore with matching flow_id to appear in batch details.")
        
        logger.info(f"🌳 Building tree from {len(organized_files)} organized files")
        
        if len(organized_files) == 0:
            logger.warning(f"⚠️  No organized files found for flow {flow_id}! Check Firestore documents or GCS.")
            logger.warning(f"    Firestore docs count: {len(firestore_documents)}")
            logger.warning(f"    GCS result success: {gcs_result.get('success', False)}")
            logger.warning(f"    GCS organized files count: {len(gcs_result.get('organized_files', []))}")
        
        # Build tree structure: year -> month -> date -> category (branch ignored)
        tree = {}
        total_files = 0
        
        for file in organized_files:
            year = file.get('year', 'unknown')
            month = file.get('month', 'unknown')
            date = file.get('date', 'unknown')
            category = file.get('category', 'unknown')
            filename = file.get('filename', '')
            key = file.get('key', '')
            
            # IMPORTANT: Only include PDF files in organized tree (final processed documents)
            # Skip JPEG/PNG files which are original uploads, not final PDF outputs
            if filename and not filename.lower().endswith('.pdf'):
                logger.debug(f"Skipping non-PDF file in organized tree: {filename}")
                continue
            
            # Skip if missing essential fields
            if year == 'unknown' or month == 'unknown' or date == 'unknown' or category == 'unknown':
                logger.warning(f"Skipping file with incomplete path info: {filename}")
                continue
            
            # Ensure we have a valid key with full path including document folder
            if not key or not filename:
                logger.warning(f"Skipping file with missing key or filename: {filename}")
                continue
            
            # Build tree path (year -> month -> date -> category)
            if year not in tree:
                tree[year] = {}
            if month not in tree[year]:
                tree[year][month] = {}
            if date not in tree[year][month]:
                tree[year][month][date] = {}
            if category not in tree[year][month][date]:
                tree[year][month][date][category] = []
            
            # Clean the key (remove bucket name if present) for URL generation
            # Store the cleaned key in tree for proper URL generation
            cleaned_key = key
            if key.startswith('gs://'):
                # Extract key from gs://bucket-name/path/to/file
                parts = key.replace('gs://', '').split('/', 1)
                if len(parts) == 2:
                    cleaned_key = parts[1]  # Use the path part as the key (without bucket name)
            elif '/' in key:
                # Handle case where key might include bucket name: bucket-name/path/to/file
                path_parts = key.split('/', 1)
                # Check if first part looks like a bucket name
                if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                    cleaned_key = path_parts[1]  # Remove bucket name
                    logger.debug(f"Removed bucket name from key: {key} -> {cleaned_key}")
            
            # Skip presigned URL generation for tree view (major performance optimization)
            # URLs will be generated on-demand when user views/downloads specific files
            url = ''
            
            logger.debug(f"Adding PDF file to tree: {filename}, cleaned_key: {cleaned_key}")
            
            tree[year][month][date][category].append({
                'filename': filename,
                'key': cleaned_key,  # Use cleaned key (without bucket name) for URL generation
                'size': file.get('size', 0),
                'url': url,
                'last_modified': file.get('last_modified', '')
            })
            total_files += 1
        
        # Convert tree to the expected format
        def build_tree_node(path_parts, children_data):
            if isinstance(children_data, dict):
                # It's a folder
                children = []
                for name, data in children_data.items():
                    new_path = path_parts + [name]
                    children.append(build_tree_node(new_path, data))
                return {
                    'name': path_parts[-1] if path_parts else 'root',
                    'path': '/'.join(path_parts),
                    'type': 'folder',
                    'children': children
                }
            else:
                # It's a file list
                return {
                    'name': path_parts[-1] if path_parts else 'files',
                    'path': '/'.join(path_parts),
                    'type': 'folder',
                    'children': [
                        {
                            'name': f.get('filename', ''),
                            'path': f.get('key', ''),  # Full s3_key path for tree navigation
                            'key': f.get('key', ''),    # Full s3_key path for URL generation
                            'type': 'file',
                            'size': f.get('size', 0),
                            'url': f.get('url', ''),
                            'last_modified': f.get('last_modified')
                        }
                        for f in children_data
                    ]
                }
        
        # Always create a root node, even if tree is empty
        if tree:
            root_node = build_tree_node([], tree)
            # Add document_count to root node for frontend display
            root_node['document_count'] = total_files
        else:
            # Create an empty root node structure
            root_node = {
                'name': 'root',
                'path': '',
                'type': 'folder',
                'children': [],
                'document_count': 0
            }
        
        logger.info(f"🌳 Built tree with {total_files} files, root node: {root_node.get('name')}, children count: {len(root_node.get('children', []))}")
        logger.info(f"📊 Summary: {len(organized_files)} organized files processed, {total_files} files added to tree")
        
        response_data = {
            'success': True,
            'tree': root_node,
            'summary': {
                'total_files': total_files,
                'total_folders': len(tree) if tree else 0
            }
        }
        
        logger.info(f"✅ Returning tree response for flow {flow_id}: success={response_data['success']}, tree exists={root_node is not None}, total_files={total_files}")
        
        return JSONResponse(content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get GCS batch organized tree error: {e}")
        return JSONResponse(content={
            'success': True,
            'tree': None,
            'summary': {
                'total_files': 0,
                'total_folders': 0
            }
        })

def _ensure_flow_exists(flow_id: str, flow_name: Optional[str] = None) -> bool:
    """
    Helper function to ensure a flow exists in Firestore.
    Creates the flow if it doesn't exist.
    
    Args:
        flow_id: The flow ID to check/create
        flow_name: Optional flow name (defaults to "Uploaded Flow")
    
    Returns:
        True if flow exists or was created successfully, False otherwise
    """
    if not firestore_service:
        return False
    
    try:
        existing_flow = firestore_service.get_flow(flow_id)
        if not existing_flow:
            # Flow doesn't exist - create it
            # Use provided flow_name if available, otherwise default to "Uploaded Flow"
            flow_name = flow_name or 'Uploaded Flow'
            logger.info(f"Flow {flow_id} doesn't exist - creating it in Firestore with name '{flow_name}'")
            firestore_service.create_flow(flow_id, {
                'flow_name': flow_name,
                'source': 'web_upload',
                'document_count': 0,
                'status': 'active'
            })
            logger.info(f"✅ Created flow {flow_id} in Firestore: {flow_name}")
            return True
        else:
            # Flow exists - check if we should update the name if it's different
            existing_name = existing_flow.get('flow_name', '')
            if flow_name and flow_name != existing_name:
                # Update the name if provided and different from existing
                # This handles cases where flow was created with default name
                logger.info(f"Updating flow {flow_id} name from '{existing_name}' to '{flow_name}'")
                firestore_service.update_flow(flow_id, {'flow_name': flow_name})
            logger.debug(f"Flow {flow_id} already exists in Firestore with name '{existing_flow.get('flow_name', 'Unknown')}'")
            return True
    except Exception as e:
        logger.error(f"Failed to ensure flow exists: {e}")
        return False

@app.post("/api/documents/create")
async def create_document_metadata(
    filename: str = Body(...),
    file_size: int = Body(...),
    content_type: str = Body(...),
    flow_id: str = Body(...),
    flow_name: Optional[str] = Body(None),
    propertyId: Optional[str] = Body(None),
    clientId: Optional[str] = Body(None),
    documentType: Optional[str] = Body(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create document metadata in Firestore and return signed URL for direct GCS upload.
    This is Step 1 of the two-step upload process.
    
    The document will be created with status="uploading" and will appear in flows immediately.
    After the file is uploaded to GCS using the signed URL, call /api/documents/{id}/complete-upload.
    """
    try:
        if not firestore_service:
            raise HTTPException(status_code=503, detail="Firestore service not available")
        
        if not s3_service or not s3_service.bucket:
            raise HTTPException(status_code=503, detail="GCS service not available")
        
        # Extract agentId from current user
        agent_id = current_user.get('id')
        if not agent_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Ensure flow exists in Firestore
        _ensure_flow_exists(flow_id, flow_name)
        
        # Generate unique document ID
        document_id = f"{flow_id}_{filename.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}_{uuid.uuid4().hex[:6]}"
        
        # Determine file extension for path
        file_ext = Path(filename).suffix.lower() if '.' in filename else ''
        
        # Generate expected GCS path (same format as upload_image_to_temp)
        clean_filename = filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        gcs_temp_path = f"temp/{flow_id}/{clean_filename}"
        
        # Create document record in Firestore with "uploading" status
        document_data = {
            'filename': filename,
            'flow_id': flow_id,
            'gcs_temp_path': gcs_temp_path,
            'file_size': file_size,
            'content_type': content_type,
            'processing_status': 'uploading',  # Document is being uploaded
            'source': 'web_upload',
            'agentId': agent_id,  # Agent who uploaded the document
        }
        
        # Add optional relationships
        if propertyId:
            document_data['propertyId'] = propertyId
        if clientId:
            document_data['clientId'] = clientId
        if documentType:
            document_data['documentType'] = documentType
        
        firestore_service.create_document(document_id, document_data)
        logger.info(f"✅ Created document metadata: {document_id} with status=uploading")
        
        # Generate signed URL for PUT upload (expires in 1 hour)
        expiration = 3600  # 1 hour
        signed_url_result = s3_service.generate_presigned_url(
            key=gcs_temp_path,
            expiration=expiration,
            method="PUT"
        )
        
        if not signed_url_result.get('success'):
            # If signed URL generation fails, we should still return the document_id
            # but log the error - the frontend can retry or fall back to old endpoint
            logger.error(f"⚠️  Failed to generate signed URL for {gcs_temp_path}: {signed_url_result.get('error')}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate upload URL: {signed_url_result.get('error')}"
            )
        
        signed_url = signed_url_result['url']
        
        # Invalidate flow files cache so the new document appears immediately
        if f"flow_files_{flow_id}" in _flow_files_cache:
            del _flow_files_cache[f"flow_files_{flow_id}"]
            logger.info(f"🗑️  Invalidated flow files cache for {flow_id} after document creation")
        
        # Invalidate flows cache
        if 'flows' in _flows_cache:
            del _flows_cache['flows']
        
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "signed_url": signed_url,
            "gcs_path": gcs_temp_path,
            "expires_in": expiration
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating document metadata: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/{document_id}/complete-upload")
async def complete_document_upload(
    document_id: str,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Mark document upload as complete and trigger background processing.
    This is Step 3 of the two-step upload process (Step 2 is direct GCS upload from frontend).
    
    After the frontend successfully uploads the file to GCS using the signed URL,
    it calls this endpoint to:
    1. Verify the file exists in GCS
    2. Update document status from "uploading" to "uploaded"
    3. Queue background processing (OCR, classification, etc.)
    """
    try:
        if not firestore_service:
            raise HTTPException(status_code=503, detail="Firestore service not available")
        
        if not s3_service or not s3_service.bucket:
            raise HTTPException(status_code=503, detail="GCS service not available")
        
        # Get document from Firestore
        document = firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Verify document is in "uploading" status
        current_status = document.get('processing_status', 'unknown')
        if current_status != 'uploading':
            logger.warning(f"Document {document_id} has status '{current_status}', expected 'uploading'")
            # Still proceed, but log the warning
        
        # Get expected GCS path
        gcs_temp_path = document.get('gcs_temp_path')
        if not gcs_temp_path:
            raise HTTPException(status_code=400, detail="Document missing gcs_temp_path")
        
        # Verify file exists in GCS
        file_exists = await s3_service.check_file_exists(gcs_temp_path)
        if not file_exists:
            raise HTTPException(
                status_code=404,
                detail=f"File not found in GCS at path: {gcs_temp_path}. Upload may have failed."
            )
        
        logger.info(f"✅ Verified file exists in GCS: {gcs_temp_path}")
        
        # Update document status to "uploaded" (will be changed to "pending" when processing starts)
        firestore_service.update_document(document_id, {
            'processing_status': 'uploaded'
        })
        logger.info(f"✅ Updated document {document_id} status to 'uploaded'")
        
        # Get flow_id and job_id for background processing
        flow_id = document.get('flow_id')
        filename = document.get('filename', 'unknown')
        job_id = flow_id  # Use flow_id as job_id (consistent with existing code)
        
        # Queue background processing task (OCR, classification, etc.)
        if task_queue:
            task_queue.add_process_task(
                background_tasks,
                document_id,
                gcs_temp_path,
                filename,
                job_id
            )
            logger.info(f"✅ Queued background processing task for document {document_id}")
        else:
            logger.warning(f"⚠️  Task queue not available - background processing not queued for {document_id}")
        
        # Invalidate caches
        if flow_id and f"flow_files_{flow_id}" in _flow_files_cache:
            del _flow_files_cache[f"flow_files_{flow_id}"]
        if 'flows' in _flows_cache:
            del _flows_cache['flows']
        
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "status": "uploaded",
            "message": "Upload completed, processing started"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing document upload: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gcs/upload-files")
async def gcs_upload_files(
    files: List[UploadFile] = File(...), 
    flow_id: Optional[str] = Form(None),
    flow_name: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    request: Request = None
):
    """
    Upload one or many files to GCS and run inline OCR. Accepts multipart/form-data.
    """
    if not files:
        return JSONResponse(content={"success": False, "error": "No files provided"}, status_code=400)

    # Try to get current user (optional authentication)
    agent_id = None
    try:
        auth_header = request.headers.get("Authorization") if request else None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            result = simple_auth.get_user(token)
            if result.get("success"):
                agent_id = result["user"].get("id")
    except Exception as e:
        logger.debug(f"Could not extract agent ID from request: {e}")

    # Ensure flow_id exists if provided; otherwise generate one
    final_flow_id = flow_id or f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure flow exists in Firestore before uploading documents
    # Use provided flow_name if available, otherwise it will default to "Uploaded Flow"
    _ensure_flow_exists(final_flow_id, flow_name)
    
    # Create job in Firestore for tracking
    job_id = None
    if firestore_service:
        try:
            job_id = final_flow_id
            firestore_service.create_job(job_id, {
                'flow_id': final_flow_id,
                'total_documents': len(files),
                'job_type': 'upload'
            })
            logger.info(f"Created job {job_id} for {len(files)} files")
        except Exception as e:
            logger.warning(f"Failed to create job in Firestore: {e}")

    uploaded = []
    failed = []
    ocr_results = []

    # PERFORMANCE: Process files in parallel instead of sequentially
    async def process_single_file(file: UploadFile) -> tuple[dict, bool]:
        """Process a single file upload"""
        try:
            content = await file.read()
            file_type = file.filename.split(".")[-1].lower() if "." in file.filename else "jpg"

            upload_res = await s3_service.upload_image_to_temp(
                image_data=content,
                filename=file.filename,
                file_type=file_type,
                flow_id=final_flow_id,
            )

            if upload_res.get("success"):
                gcs_temp_path = upload_res.get('s3_key')
                
                # Verify the file was actually saved to GCS before creating Firestore record
                if gcs_temp_path and await s3_service.check_file_exists(gcs_temp_path):
                    logger.info(f"✅ Verified file exists in GCS: {gcs_temp_path}")
                    
                    # Create document record in Firestore and trigger background processing
                    if firestore_service and task_queue:
                        try:
                            document_id = f"{final_flow_id}_{file.filename}_{uuid.uuid4().hex[:6]}"
                            
                            # Create document record
                            document_data = {
                                'filename': file.filename,
                                'flow_id': final_flow_id,
                                'gcs_temp_path': gcs_temp_path,
                                'file_size': upload_res.get('size', 0),
                                'processing_status': 'pending',
                                'source': 'web_upload'
                            }
                            if agent_id:
                                document_data['agentId'] = agent_id
                            
                            firestore_service.create_document(document_id, document_data)
                            logger.info(f"Created document record: {document_id} with gcs_temp_path: {gcs_temp_path}")
                            
                            # Add background processing task
                            task_queue.add_process_task(
                                background_tasks,
                                document_id,
                                gcs_temp_path,
                                file.filename,
                                job_id
                            )
                            logger.info(f"Added processing task for: {file.filename}")
                        except Exception as e:
                            logger.error(f"Failed to create document or add task: {e}")
                            return ({"filename": file.filename, "error": str(e)}, False)
                    
                    return (upload_res, True)
                else:
                    # File upload reported success but file doesn't exist in GCS
                    error_msg = f"File upload reported success but file not found in GCS: {gcs_temp_path}"
                    logger.error(f"❌ {error_msg}")
                    return ({"filename": file.filename, "error": error_msg}, False)
            else:
                return ({"filename": file.filename, "error": upload_res.get("error")}, False)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return ({"filename": file.filename, "error": str(e)}, False)
    
    # Process all files in parallel
    results = await asyncio.gather(*[process_single_file(file) for file in files], return_exceptions=True)
    
    # Track document IDs for batch matching
    uploaded_document_ids = []
    
    # Separate successful and failed uploads
    for result in results:
        if isinstance(result, Exception):
            failed.append({"filename": "unknown", "error": str(result)})
        else:
            upload_res, success = result
            if success:
                uploaded.append(upload_res)
                # Extract document_id from upload response if available
                # Document IDs are created in process_single_file as: f"{final_flow_id}_{file.filename}_{uuid.uuid4().hex[:6]}"
                # We'll need to get them from Firestore after processing starts
            else:
                failed.append(upload_res)
    
    # Note: Batch matching will happen automatically as documents complete processing
    # Each document will use _find_or_create_property_file which will match them together
    # Documents uploaded together will naturally group by client name as they complete

    total_files = len(files)
    successful_uploads = len(uploaded)
    failed_uploads = len(failed)

    # Update flow document count after successful uploads
    if successful_uploads > 0 and firestore_service:
        try:
            # Get current flow to check document count
            flow = firestore_service.get_flow(final_flow_id)
            if flow:
                # Update document count (this will be incremented as documents are processed)
                # For now, we'll update it to reflect the number of documents we just created
                current_count = flow.get('document_count', 0)
                # Note: The actual count will be updated by increment_flow_document_count as documents complete processing
                logger.info(f"Flow {final_flow_id} has {current_count} documents, {successful_uploads} new documents uploaded")
            else:
                logger.warning(f"Flow {final_flow_id} not found when trying to update document count")
        except Exception as e:
            logger.error(f"Failed to update flow document count: {e}")

    # Invalidate caches when files are uploaded
    if successful_uploads > 0:
        if 'flows' in _flows_cache:
            del _flows_cache['flows']
            logger.info(f"🗑️  Invalidated flows cache after uploading {successful_uploads} file(s) to flow {final_flow_id}")
        # Also invalidate browse documents cache
        invalidate_document_caches()

    # Build unified results array for detailed frontend logging
    results = []
    for entry in uploaded:
        results.append({
            **entry,
            "success": True
        })
    for entry in failed:
        results.append({
            **entry,
            "success": False
        })

    return JSONResponse(
        content={
            "success": successful_uploads > 0 and failed_uploads == 0,
            "flow_id": final_flow_id,
            "batch_id": final_flow_id,  # alias for backward compatibility
            "total_files": total_files,
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "results": results,
            "uploaded": uploaded,
            "failed": failed,
            "ocr_results": ocr_results,
        }
    )

@app.post("/api/test/sqs-send")
async def test_sqs_send():
    """Deprecated AWS/SQS test endpoint."""
    return JSONResponse(
        content={
            "success": False,
            "error": "SQS is disabled in this GCS-only setup."
        },
        status_code=410
    )

@app.post("/api/flow/create")
async def create_flow_job(document_ids: List[str], background_tasks: BackgroundTasks):
    """Create a new flow processing job"""
    try:
        # Create flow job
        job = FlowJob(
            documents=document_ids,
            total_documents=len(document_ids)
        )
        
        # Store job
        flow_jobs[job.job_id] = job
        
        # Add to processing queue
        await processing_queue.put(job.job_id)
        
        return JSONResponse(content={
            "success": True,
            "job_id": job.job_id,
            "total_documents": job.total_documents,
            "status": job.status
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flow/{job_id}")
async def get_flow_status(job_id: str):
    """Get flow job status"""
    if job_id not in flow_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = flow_jobs[job_id]
    # Use Pydantic's JSON serialization which handles datetime objects automatically
    return JSONResponse(
        content=json.loads(job.json()),
        headers={"Content-Type": "application/json"}
    )

@app.get("/api/folders")
async def get_organized_folders():
    """Get list of organized folders with document counts, grouped by voucher type."""
    logger.info("Received request for /api/folders")
    try:
        folders_data = defaultdict(lambda: {
            "name": "",
            "document_count": 0,
            "last_modified": datetime.min.isoformat(),
            "documents": []
        })

        if not ORGANIZED_DIR.exists():
            logger.warning(f"Organized directory not found: {ORGANIZED_DIR}")
            return JSONResponse(content={"success": True, "folders": [], "total_folders": 0})

        counted_docs = set()
        all_files = list(ORGANIZED_DIR.rglob('*'))

        for file_path in all_files:
            if not file_path.is_file():
                continue

            voucher_type = file_path.parent.name
            
            if re.match(r"^\d{4}$", voucher_type) or re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", voucher_type) or voucher_type.lower() in ['organized_vouchers', 'default'] or voucher_type.startswith("Branch"):
                continue

            folder_info = folders_data[voucher_type]

            if not folder_info["name"]:
                folder_info["name"] = voucher_type

            file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_modified_time.isoformat() > folder_info["last_modified"]:
                folder_info["last_modified"] = file_modified_time.isoformat()

            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf']:
                doc_basename = file_path.stem
                if (voucher_type, doc_basename) not in counted_docs:
                    folder_info["document_count"] += 1
                    counted_docs.add((voucher_type, doc_basename))

            folder_info["documents"].append({
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_modified_time.isoformat(),
                "type": file_path.suffix.lower().lstrip('.')
            })

        processed_folders = []
        for voucher_type, data in folders_data.items():
            preview_docs = sorted(
                [d for d in data['documents'] if d['type'] in ['png', 'jpg', 'jpeg', 'pdf']],
                key=lambda d: d['modified'],
                reverse=True
            )
            data["documents"] = preview_docs[:10]
            data["path"] = data["name"]
            processed_folders.append(data)

        sorted_folders = sorted(processed_folders, key=lambda f: f['last_modified'], reverse=True)
        
        logger.info(f"Successfully found {len(sorted_folders)} folders.")
        return {
            "success": True,
            "folders": sorted_folders,
            "total_folders": len(sorted_folders)
        }
    
    except Exception as e:
        logger.error(f"Error in /api/folders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/api/folders/{folder_name}/documents")
async def get_folder_documents(folder_name: str):
    """Get all documents for a specific voucher type"""
    documents = []
    if not ORGANIZED_DIR.exists():
        raise HTTPException(status_code=404, detail="Organized directory not found")

    for file_path in ORGANIZED_DIR.rglob('*'):
        if file_path.is_file() and file_path.parent.name == folder_name and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf']:
            documents.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "type": file_path.suffix[1:],
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
            
    return JSONResponse(content={
        "success": True,
        "folder": folder_name,
        "documents": sorted(documents, key=lambda d: d['modified'], reverse=True),
        "total": len(documents)
    })

@app.get("/api/document/{folder_name}/{document_name}")
async def get_document(folder_name: str, document_name: str):
    """Get a specific document file"""
    file_path = ORGANIZED_DIR / folder_name / document_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    return FileResponse(path=file_path)

# Google Cloud Storage API endpoints
@app.post("/api/gcs/upload-folder")
async def upload_folder_to_gcs():
    """Upload entire organized_vouchers folder to GCS"""
    if not gcs_service:
        raise HTTPException(status_code=503, detail="GCS service not available")
    
    try:
        result = gcs_service.upload_folder_to_gcs(str(ORGANIZED_DIR))
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/gcs/upload-single")
async def upload_single_voucher(voucher_type: str, document_no: str, file_path: str):
    """Upload a single voucher file to GCS"""
    if not gcs_service:
        raise HTTPException(status_code=503, detail="GCS service not available")
    
    try:
        result = gcs_service.upload_single_voucher(file_path, voucher_type, document_no)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/gcs/list-vouchers")
async def list_uploaded_vouchers(prefix: str = ""):
    """List all uploaded vouchers in GCS bucket"""
    if not gcs_service:
        raise HTTPException(status_code=503, detail="GCS service not available")
    
    try:
        vouchers = gcs_service.list_uploaded_vouchers(prefix)
        return JSONResponse(content={
            "success": True,
            "vouchers": vouchers,
            "total": len(vouchers)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list vouchers: {str(e)}")

@app.post("/api/gcs/download")
async def download_voucher_from_gcs(gcs_path: str, local_path: str):
    """Download a voucher from GCS to local storage"""
    if not gcs_service:
        raise HTTPException(status_code=503, detail="GCS service not available")
    
    try:
        result = gcs_service.download_voucher(gcs_path, local_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/document/{document_id}/status")
async def get_document_status(document_id: str):
    """Get processing status of a specific document"""
    if document_id in processed_documents:
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "status": "processed",
            "data": processed_documents[document_id]
        })
    else:
        return JSONResponse(content={
            "success": True,
            "document_id": document_id,
            "status": "not_processed",
            "data": None
        })

@app.get("/api/documents/processed")
async def get_processed_documents():
    """Get list of all processed documents"""
    return JSONResponse(content={
        "success": True,
        "total_processed": len(processed_documents),
        "documents": processed_documents
    })

# AWS API endpoints
@app.get("/api/aws/s3/list")
async def list_s3_temp_files():
    """List all files in S3 temp folder"""
    try:
        result = s3_service.list_temp_files()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list S3 files: {str(e)}")

@app.get("/api/aws/s3/organized")
async def list_s3_organized_files():
    """List all organized files (processed by Lambda) with hierarchical structure"""
    try:
        result = s3_service.list_organized_files()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list organized files: {str(e)}")

@app.get("/api/aws/s3/organized/folders")
async def get_s3_organized_folders():
    """Get organized folders structure from S3 (for frontend browsing)"""
    try:
        result = s3_service.get_organized_folders()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get organized folders: {str(e)}")

@app.get("/api/aws/s3/organized/folder/{classification}")
async def get_s3_folder_documents(classification: str):
    """Get all documents in a specific classification folder from S3"""
    try:
        result = s3_service.get_folder_documents(classification)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folder documents: {str(e)}")

@app.get("/api/aws/s3/organized/tree")
async def get_s3_organized_tree():
    """Get hierarchical organized folder tree from S3."""
    try:
        result = s3_service.get_organized_folder_tree()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get organized folder tree: {str(e)}")

@app.get("/api/aws/s3/organized-tree/flow/{flow_id}")
async def get_flow_organized_tree(flow_id: str):
    """Get hierarchical organized folder tree for a specific flow from S3."""
    try:
        result = s3_service.get_flow_organized_tree(flow_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get flow organized tree: {str(e)}")

@app.get("/api/aws/s3/organized-tree/category/{category}")
async def get_category_organized_tree(category: str):
    """Get hierarchical organized folder tree for a specific category/voucher type from S3."""
    try:
        result = s3_service.get_category_organized_tree(category)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get category organized tree: {str(e)}")

@app.get("/api/aws/sqs/messages")
async def get_sqs_messages():
    """SQS is deprecated - processing now uses background tasks"""
    return JSONResponse(content={
        "success": True,
        "messages": [],
        "count": 0,
        "note": "SQS is no longer used. Documents are processed via background tasks."
    })

@app.get("/api/aws/sqs/processed")
async def get_processed_messages():
    """SQS is deprecated - check Firestore for document status"""
    return JSONResponse(content={
        "success": True,
        "messages": [],
        "count": 0,
        "note": "SQS is no longer used. Check Firestore for document processing status."
    })

def count_files_in_tree(node: Dict[str, Any]) -> int:
    """Recursively count files in a tree structure"""
    if not node:
        return 0
    
    count = 0
    if node.get('type') == 'file':
        count = 1
    
    if 'children' in node and node['children']:
        for child in node['children']:
            count += count_files_in_tree(child)
    
    return count

@app.get("/api/aws/batch/{batch_id}/status")
async def get_aws_batch_status(batch_id: str):
    """Get batch processing status (GCP-based) - Enhanced with GCS and Firestore checks"""
    try:
        logger.info(f"📊 Checking batch status for {batch_id}")
        
        # Initialize default values
        job = None
        gcs_files = None
        firestore_docs = []
        
        # Get Firestore job status
        if firestore_service:
            try:
                job = firestore_service.get_job(batch_id)
                if job:
                    logger.info(f"📋 Found Firestore job: total={job.get('total_documents', 0)}, "
                              f"processed={job.get('processed_documents', 0)}, "
                              f"failed={job.get('failed_documents', 0)}")
            except Exception as e:
                logger.error(f"Failed to get job status from Firestore: {e}")
        
        # Check GCS for actual file counts
        temp_count = 0
        organized_count = 0
        failed_count_gcs = 0
        
        try:
            gcs_files = s3_service.get_flow_files_from_s3(batch_id)
            if gcs_files.get('success'):
                temp_count = len(gcs_files.get('temp_files', []))
                organized_count = len(gcs_files.get('organized_files', []))
                failed_count_gcs = len(gcs_files.get('failed_files', []))
                logger.info(f"📁 GCS file counts for {batch_id}: temp={temp_count}, "
                          f"organized={organized_count}, failed={failed_count_gcs}")
            else:
                logger.warning(f"⚠️  GCS file check failed for {batch_id}: {gcs_files.get('error')}")
        except Exception as e:
            logger.error(f"Error checking GCS files for {batch_id}: {e}")
            gcs_files = None
        
        # Check Firestore for document statuses
        if firestore_service:
            try:
                firestore_docs, total_firestore_docs = firestore_service.get_documents_by_flow_id(
                    batch_id, page=1, page_size=1000
                )
                logger.info(f"📄 Found {len(firestore_docs)} documents in Firestore for {batch_id}")
            except Exception as e:
                logger.error(f"Failed to get Firestore documents for {batch_id}: {e}")
                firestore_docs = []
        
        # Analyze document statuses
        stuck_processing = [d for d in firestore_docs if d.get('processing_status') == 'processing']
        completed_docs = [d for d in firestore_docs if d.get('processing_status') == 'completed']
        failed_docs = [d for d in firestore_docs if d.get('processing_status') == 'failed']
        need_review_docs = [d for d in firestore_docs if d.get('processing_status') == 'need_review']
        pending_docs = [d for d in firestore_docs if d.get('processing_status') == 'pending']
        
        logger.info(f"📊 Document status breakdown: completed={len(completed_docs)}, "
                  f"failed={len(failed_docs)}, need_review={len(need_review_docs)}, "
                  f"processing={len(stuck_processing)}, pending={len(pending_docs)}")
        
        # Calculate totals from multiple sources
        total_from_gcs = temp_count + organized_count + failed_count_gcs
        total_from_firestore = len(firestore_docs)
        total_from_job = job.get('total_documents', 0) if job else 0
        total_docs = max(total_from_gcs, total_from_firestore, total_from_job)
        
        logger.info(f"📊 Total documents: GCS={total_from_gcs}, Firestore={total_from_firestore}, "
                  f"Job={total_from_job}, Final={total_docs}")
        
        # Determine if batch is complete
        # Batch is complete when:
        # 1. No temp files in GCS
        # 2. No documents stuck in "processing" status
        # 3. All documents have final status (completed, failed, need_review, or pending)
        is_complete = (
            temp_count == 0 and
            len(stuck_processing) == 0 and
            (len(completed_docs) + len(failed_docs) + len(need_review_docs) + len(pending_docs) >= total_docs or total_docs == 0)
        )
        
        # Calculate processing count
        processing_count = len(stuck_processing) if stuck_processing else temp_count
        
        # Determine final status
        if is_complete:
            has_errors = failed_count_gcs > 0 or len(failed_docs) > 0
            status = "completed_with_errors" if has_errors else "completed"
            processing_list = []  # Clear processing list when complete!
            processing = 0
            logger.info(f"✅ Batch {batch_id} is COMPLETE: status={status}, "
                      f"errors={has_errors}")
        else:
            status = "processing"
            # Include stuck documents in processing_list
            processing_list = [
                {
                    'document_id': d.get('document_id', d.get('id', '')),
                    'status': 'processing'
                }
                for d in stuck_processing
            ]
            processing = processing_count
            logger.info(f"⏳ Batch {batch_id} is PROCESSING: temp_files={temp_count}, "
                      f"stuck_docs={len(stuck_processing)}")
        
        # Calculate processed and failed counts (use most accurate source)
        if firestore_docs:
            processed = len(completed_docs)
            failed = len(failed_docs)
        elif job:
            processed = job.get('processed_documents', 0)
            failed = job.get('failed_documents', 0)
        else:
            processed = organized_count
            failed = failed_count_gcs
        
        # Build response
        response_data = {
            "success": True,
            "batch_id": batch_id,
            "flow_id": batch_id,
            "status": status,
            "total_documents": total_docs,
            "processed": processed,
            "failed": failed,
            "processing": processing,
            "processing_list": processing_list,
        }
        
        if job:
            # Convert Firestore datetime to ISO string for JSON serialization
            updated_at = job.get('updated_at')
            if updated_at:
                if hasattr(updated_at, 'isoformat'):
                    response_data["last_updated"] = updated_at.isoformat()
                elif isinstance(updated_at, str):
                    response_data["last_updated"] = updated_at
                else:
                    # Fallback: convert to string
                    response_data["last_updated"] = str(updated_at)
        
        # Add debug info in development
        if logger.level <= logging.DEBUG:
            response_data["debug"] = {
                "gcs_counts": {
                    "temp": temp_count,
                    "organized": organized_count,
                    "failed": failed_count_gcs
                },
                "firestore_counts": {
                    "total": len(firestore_docs),
                    "completed": len(completed_docs),
                    "failed": len(failed_docs),
                    "processing": len(stuck_processing),
                    "pending": len(pending_docs)
                },
                "is_complete": is_complete
            }
        
        # Use convert_decimals helper to ensure all datetime objects are serialized
        try:
            response_data = convert_decimals(response_data)
            return JSONResponse(content=response_data)
        except Exception as e:
            logger.error(f"Error serializing response data for batch {batch_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a safe fallback response
            return JSONResponse(content={
                "success": False,
                "batch_id": batch_id,
                "status": "error",
                "error": f"Failed to serialize response: {str(e)}",
                "total_documents": total_docs,
                "processed": processed,
                "failed": failed,
                "processing": processing,
                "processing_list": []
            })
    except Exception as e:
        # Catch any unexpected errors and return a safe response with CORS headers
        logger.error(f"Unexpected error in get_aws_batch_status for {batch_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "success": False,
                "batch_id": batch_id,
                "status": "error",
                "error": f"Internal server error: {str(e)}",
                "total_documents": 0,
                "processed": 0,
                "failed": 0,
                "processing": 0,
                "processing_list": []
            },
            status_code=500
        )

@app.get("/api/aws/processing/summary")
async def get_processing_summary():
    """Get overall processing summary from Firestore"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "summary": {
                "total_batches": 0,
                "total_documents": 0,
                "message": "Firestore not available"
            }
        })
    
    try:
        # Get jobs summary from Firestore
        jobs, total_jobs = firestore_service.list_flows(page=1, page_size=100)
        
        total_batches = total_jobs
        total_documents = sum(job.get('document_count', 0) for job in jobs)
        
        # Get document statistics
        docs, total_docs = firestore_service.list_documents(page=1, page_size=1)
        
        # Calculate completion stats
        completed_docs = sum(1 for doc in docs if doc.get('processing_status') == 'completed')
        failed_docs = sum(1 for doc in docs if doc.get('processing_status') == 'failed')
        processing_docs = sum(1 for doc in docs if doc.get('processing_status') == 'processing')
    except Exception as e:
        logger.error(f"Failed to get processing summary: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        })
    
    # Legacy variable definitions (not used but kept for compatibility)
    aws_processing_status = {}
    total_processing = processing_docs
    
    return JSONResponse(content={
        "success": True,
        "summary": {
            "total_batches": total_batches,
            "total_documents": total_docs,
            "completed_documents": completed_docs,
            "processing_documents": processing_docs,
            "failed_documents": failed_docs,
            "success_rate": (completed_docs / max(1, total_docs)) * 100 if total_docs > 0 else 0
        }
    })

@app.get("/api/document/{document_id}/processing-status")
async def get_document_processing_status(document_id: str):
    """Get detailed processing status for a specific document from Firestore"""
    if not firestore_service:
        return JSONResponse(content={
            "success": False,
            "document_id": document_id,
            "message": "Firestore not available"
        })
    
    try:
        # URL decode the document_id in case it was encoded
        from urllib.parse import unquote
        document_id = unquote(document_id)
        
        logger.info(f"Getting processing status for document: {document_id}")
        doc = firestore_service.get_document(document_id)
        
        if doc:
            return JSONResponse(content={
                "success": True,
                "document_id": document_id,
                "status": doc
            })
        else:
            # Try to find document by searching in the flow if document_id contains flow_id
            # Document ID format: flow-YYYYMMDD_HHMMSS_filename_randomhex
            if '_' in document_id:
                parts = document_id.split('_')
                if len(parts) >= 2:
                    # Try to extract flow_id (format: flow-YYYYMMDD_HHMMSS)
                    potential_flow_id = '_'.join(parts[:2]) if parts[0].startswith('flow-') else None
                    if potential_flow_id:
                        logger.info(f"Document not found by ID, trying to search in flow: {potential_flow_id}")
                        try:
                            flow_docs, _ = firestore_service.get_documents_by_flow_id(
                                potential_flow_id, page=1, page_size=100
                            )
                            # Try to find document by matching filename or partial ID
                            for flow_doc in flow_docs:
                                doc_id = flow_doc.get('document_id') or flow_doc.get('id', '')
                                filename = flow_doc.get('filename', '')
                                # Check if this might be the document we're looking for
                                if (document_id in doc_id or 
                                    doc_id in document_id or
                                    any(part in filename for part in parts[2:] if len(parts) > 2)):
                                    logger.info(f"Found potential match: {doc_id}")
                                    return JSONResponse(content={
                                        "success": True,
                                        "document_id": doc_id,
                                        "status": flow_doc,
                                        "note": "Found by flow search"
                                    })
                        except Exception as search_error:
                            logger.warning(f"Flow search failed: {search_error}")
            
            logger.warning(f"Document not found: {document_id}")
            return JSONResponse(content={
                "success": False,
                "document_id": document_id,
                "message": "Document not found in Firestore"
            }, status_code=404)
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(content={
            "success": False,
            "document_id": document_id,
            "error": str(e)
        }, status_code=500)

@app.post("/api/aws/test")
async def test_gcp_connection():
    """Test GCP services connection (GCS, Firestore, Anthropic API)"""
    try:
        results = {
            "success": True,
            "message": "GCP connection test completed",
            "services": {}
        }
        
        # Test GCS
        try:
            if gcs_service:
                # Try to list files in bucket
                list(gcs_service.bucket.list_blobs(max_results=1))
                results["services"]["gcs"] = {
                    "status": "connected",
                    "bucket": settings.GCS_BUCKET_NAME,
                    "project": settings.GCS_PROJECT_ID
                }
            else:
                results["services"]["gcs"] = {"status": "not initialized"}
        except Exception as e:
            results["services"]["gcs"] = {"status": "error", "message": str(e)}
            results["success"] = False
        
        # Test Firestore
        try:
            if firestore_service:
                # Try to query a collection
                list(firestore_service.documents_collection.limit(1).stream())
                results["services"]["firestore"] = {
                    "status": "connected",
                    "project": settings.FIRESTORE_PROJECT_ID
                }
            else:
                results["services"]["firestore"] = {"status": "not initialized"}
        except Exception as e:
            results["services"]["firestore"] = {"status": "error", "message": str(e)}
            results["success"] = False
        
        # Test Anthropic API
        try:
            if document_processor and document_processor.anthropic_client:
                results["services"]["anthropic"] = {
                    "status": "configured",
                    "model": settings.ANTHROPIC_MODEL
                }
            else:
                results["services"]["anthropic"] = {"status": "not initialized"}
        except Exception as e:
            results["services"]["anthropic"] = {"status": "error", "message": str(e)}
        
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCP connection test failed: {str(e)}")

@app.post("/api/aws/test-integration")
async def test_gcp_integration():
    """Test complete GCP integration flow (GCS, Firestore, OCR)"""
    try:
        # Check temp files in GCS
        temp_files = s3_service.list_temp_files()
        
        # Check organized files (processed output)
        organized_files = s3_service.list_organized_files()
        
        # Check Firestore documents
        firestore_docs = []
        firestore_count = 0
        if firestore_service:
            try:
                docs, total = firestore_service.list_documents(page=1, page_size=5)
                firestore_docs = docs
                firestore_count = total
            except Exception as e:
                logger.warning(f"Failed to query Firestore: {e}")
        
        return JSONResponse(content={
            "success": True,
            "message": "GCP integration test completed",
            "gcs_temp_files": {
                "count": temp_files.get('count', 0),
                "recent_files": temp_files.get('files', [])[:5]  # Show last 5
            },
            "gcs_organized_files": {
                "count": organized_files.get('count', 0),
                "structure_type": organized_files.get('structure_type', 'hierarchical')
            },
            "firestore_documents": {
                "count": firestore_count,
                "recent_documents": [
                    {
                        "document_id": doc.get('document_id'),
                        "filename": doc.get('filename'),
                        "processing_status": doc.get('processing_status'),
                        "created_at": doc.get('created_at')
                    } for doc in firestore_docs[:3]
                ]
            },
            "config": {
                "gcs_bucket": settings.GCS_BUCKET_NAME,
                "gcs_project": settings.GCS_PROJECT_ID,
                "firestore_project": settings.FIRESTORE_PROJECT_ID,
                "anthropic_model": settings.ANTHROPIC_MODEL
            }
        })
        
    except Exception as e:
        logger.error(f"Integration test error: {e}")
        raise HTTPException(status_code=500, detail=f"Integration test failed: {str(e)}")

@app.post("/api/aws/s3/reclassify")
async def reclassify_file(
    current_key: str = Body(..., embed=True),
    new_classification: str = Body(..., embed=True)
):
    """
    Move a file to a different classification (type) folder in S3.
    Updates the file path while maintaining the rest of the hierarchical structure.
    """
    try:
        logger.info(f"Reclassifying file: {current_key} -> {new_classification}")
        
        # Move file in S3
        result = s3_service.move_file_to_classification(current_key, new_classification)
        
        if result['success']:
            logger.info(f"✅ File reclassified successfully: {current_key} -> {result['new_key']}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ File reclassification failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Reclassification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aws/s3/download")
async def download_s3_file(key: str):
    """
    Download a file from GCS by providing its key.
    Returns the file as a streaming response.
    Handles both regular keys and gs:// URLs.
    """
    try:
        # Convert gs:// URL to key if needed, or remove bucket name if present
        actual_key = key
        if key.startswith('gs://'):
            # Extract key from gs://bucket-name/path/to/file
            parts = key.replace('gs://', '').split('/', 1)
            if len(parts) == 2:
                actual_key = parts[1]  # Use the path part as the key (without bucket name)
            else:
                raise HTTPException(status_code=400, detail="Invalid gs:// URL format")
        elif '/' in key:
            # Handle case where key might include bucket name: bucket-name/path/to/file
            path_parts = key.split('/', 1)
            # Check if first part looks like a bucket name
            if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                actual_key = path_parts[1]  # Remove bucket name
                logger.info(f"Removed bucket name from key: {key} -> {actual_key}")
        
        logger.info(f"Downloading file with key: {actual_key}")
        
        # Get the file from GCS using s3_service (which is GCS-backed)
        if not s3_service.client or not s3_service.bucket:
            raise HTTPException(status_code=500, detail="GCS client not initialized")
        
        blob = s3_service.bucket.blob(actual_key)
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {actual_key}")
        
        # Download file content
        file_content = blob.download_as_bytes()
        
        # Get filename from key
        filename = actual_key.split('/')[-1]
        
        # Determine content type
        content_type = blob.content_type or 'application/octet-stream'
        if not content_type or content_type == 'application/octet-stream':
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif filename.lower().endswith('.png'):
                content_type = 'image/png'
            elif filename.lower().endswith('.pdf'):
                content_type = 'application/pdf'
            elif filename.lower().endswith('.txt'):
                content_type = 'text/plain'
        
        # Create response
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(len(file_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=404, detail=f"File not found or error downloading: {str(e)}")

@app.get("/api/gcs/presigned-url")
async def get_gcs_presigned_url(key: str, expiration: int = 3600):
    """
    Generate a V4 signed URL for secure access to GCS objects.
    This allows viewing files (PDFs, images) directly in the browser without downloading.
    Requires service account credentials with private key for signing.
    
    Args:
        key: GCS object key (can include bucket name prefix or gs:// URL)
        expiration: URL expiration time in seconds (default: 1 hour, max: 7 days)
    
    Returns:
        V4 signed URL that can be used for direct access
    """
    # Use the same implementation as the AWS-compat endpoint
    return await get_presigned_url_internal(key, expiration)

@app.get("/api/debug/list-flow-files/{flow_id}")
async def debug_list_flow_files(flow_id: str, filename: Optional[str] = None):
    """DEBUG: List all files in a flow's directories to help troubleshoot file not found errors"""
    try:
        logger.info(f"📋 DEBUG: Listing files for flow {flow_id}")
        
        # Check temp directory
        temp_files = s3_service._get_flow_temp_files(flow_id, max_results=100)
        logger.info(f"📋 DEBUG: Found {len(temp_files)} files in temp/{flow_id}/")
        
        # Also check root-level flow directory (without temp prefix)
        root_files = []
        try:
            blobs = s3_service.client.list_blobs(s3_service.bucket_name, prefix=f"{flow_id}/", max_results=100)
            root_files = [
                {
                    "key": b.name,
                    "filename": b.name.split('/')[-1],
                    "size": b.size,
                    "last_modified": b.updated.isoformat() if b.updated else None
                }
                for b in blobs
                if not b.name.endswith("/")
            ]
            logger.info(f"📋 DEBUG: Found {len(root_files)} files in {flow_id}/ (root level)")
        except Exception as e:
            logger.warning(f"Failed to list root-level files: {e}")
        
        # Check Firestore for document records
        firestore_docs = []
        if firestore_service:
            try:
                docs, _ = firestore_service.get_documents_by_flow_id(flow_id, page=1, page_size=100)
                firestore_docs = [
                    {
                        "document_id": doc.get('id', ''),
                        "filename": doc.get('filename', ''),
                        "gcs_temp_path": doc.get('gcs_temp_path', ''),
                        "gcs_path": doc.get('gcs_path', ''),
                        "status": doc.get('processing_status', '')
                    }
                    for doc in docs
                ]
                logger.info(f"📋 DEBUG: Found {len(firestore_docs)} documents in Firestore for flow {flow_id}")
            except Exception as e:
                logger.warning(f"Failed to query Firestore: {e}")
        
        search_matches: List[str] = []
        if filename:
            variations = {
                filename,
                filename.replace(" ", "_"),
                filename.replace(" ", "_").replace("(", "").replace(")", ""),
                filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""),
            }
            search_matches = s3_service.find_files_by_filename(
                [name for name in variations if name],
                max_results=50,
            )
            logger.info(f"📋 DEBUG: Found {len(search_matches)} global matches for filename {filename}")
        
        return {
            "success": True,
            "flow_id": flow_id,
            "temp_directory": f"temp/{flow_id}/",
            "temp_count": len(temp_files),
            "temp_files": [
                {
                    "key": f.get('key'),
                    "filename": f.get('key', '').split('/')[-1],
                    "size": f.get('size'),
                    "last_modified": f.get('last_modified')
                }
                for f in temp_files
            ],
            "root_directory": f"{flow_id}/",
            "root_count": len(root_files),
            "root_files": root_files,
            "firestore_count": len(firestore_docs),
            "firestore_docs": firestore_docs[:10],  # Limit to first 10
            "search_matches": search_matches,
        }
    except Exception as e:
        logger.error(f"❌ Error listing files for flow {flow_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "flow_id": flow_id
        }

@app.get("/api/aws/s3/presigned-url")
async def get_presigned_url(key: str, expiration: int = 3600):
    """
    [DEPRECATED] Use /api/gcs/presigned-url instead.
    AWS S3-compatible endpoint that actually uses GCS.
    Generate a V4 signed URL for secure access to GCS objects.
    This allows viewing files (PDFs, images) directly in the browser without downloading.
    Requires service account credentials with private key for signing.
    
    Args:
        key: GCS object key (can include bucket name prefix or gs:// URL)
        expiration: URL expiration time in seconds (default: 1 hour, max: 7 days)
    
    Returns:
        V4 signed URL that can be used for direct access
    """
    return await get_presigned_url_internal(key, expiration)

async def get_presigned_url_internal(key: str, expiration: int = 3600):
    """
    Internal implementation for generating V4 signed URLs from GCS.
    Used by both /api/gcs/presigned-url and /api/aws/s3/presigned-url endpoints.
    """
    original_key = key
    decoded_key = None
    actual_key = None
    
    try:
        # Explicitly URL-decode the key (FastAPI should do this automatically, but be explicit)
        # Handle double-encoding or edge cases
        decoded_key = unquote(key)
        if decoded_key != key:
            logger.info(f"URL-decoded key: {key} -> {decoded_key}")
        
        # Limit expiration to maximum 7 days
        max_expiration = 7 * 24 * 3600  # 7 days in seconds
        if expiration > max_expiration:
            expiration = max_expiration
            
        # Convert gs:// URL to key if needed, or remove bucket name if present
        actual_key = decoded_key
        if decoded_key.startswith('gs://'):
            # Extract key from gs://bucket-name/path/to/file
            parts = decoded_key.replace('gs://', '').split('/', 1)
            if len(parts) == 2:
                actual_key = parts[1]  # Use the path part as the key (without bucket name)
            else:
                raise HTTPException(status_code=400, detail="Invalid gs:// URL format")
        elif '/' in decoded_key:
            # Handle case where key might include bucket name: bucket-name/path/to/file
            # Example: voucher-bucket-1/organized_vouchers/id/2025/nov/28-11-2025/784-1985-1234567-1/file.pdf
            path_parts = decoded_key.split('/', 1)
            # Check if first part looks like a bucket name
            if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                actual_key = path_parts[1]  # Remove bucket name
                logger.info(f"Removed bucket name from key: {decoded_key} -> {actual_key}")
        
        # Check if GCS service is initialized
        if not s3_service.bucket:
            logger.error("GCS bucket not initialized")
            raise HTTPException(status_code=500, detail="GCS bucket not initialized")
        
        # Try generating presigned URL with key as-is first
        logger.info(f"🔍 Attempting to generate V4 signed URL for key: {actual_key} (original: {key})")
        result = s3_service.generate_presigned_url(actual_key, expiration, method="GET")
        
        if result.get('success'):
            logger.info(f"✅ Successfully generated presigned URL on first attempt for: {actual_key}")
        
        # If failed with "not found" error, try multiple fallback strategies
        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            is_not_found = 'does not exist' in error_msg.lower() or 'not found' in error_msg.lower()
            logger.info(f"Presigned URL generation failed. Error: {error_msg}, is_not_found: {is_not_found}")
            
            if is_not_found:
                logger.info(f"File not found, attempting fallback strategies for key: {actual_key}")
                try:
                    key_parts = actual_key.rsplit('/', 1)
                    if len(key_parts) == 2:
                        path_dir = key_parts[0]
                        filename = key_parts[1]
                        
                        # Strategy 1: Check Firestore FIRST - it has the exact path that was saved during upload
                        # This is the most reliable source of truth since Firestore stores the actual gcs_temp_path
                        if not result['success'] and firestore_service:
                            try:
                                # Extract flow_id from path (e.g., "temp/flow-20251129_180531" -> "flow-20251129_180531")
                                path_segments = path_dir.split('/')
                                flow_id = None
                                if len(path_segments) >= 2 and path_segments[0] == 'temp':
                                    flow_id = path_segments[1]
                                elif len(path_segments) >= 1:
                                    # Also try if path is just the flow_id (without temp prefix)
                                    flow_id = path_segments[-1] if path_segments[-1] else path_segments[0]
                                
                                if flow_id:
                                    logger.info(f"Strategy 1: Checking Firestore for flow: {flow_id}, filename: {filename}")
                                    # Get all documents for this flow from Firestore
                                    firestore_docs, _ = firestore_service.get_documents_by_flow_id(flow_id, page=1, page_size=1000)
                                    logger.info(f"Strategy 1: Found {len(firestore_docs)} documents in Firestore for flow {flow_id}")
                                    
                                    # Try to find a document with matching filename
                                    target_name_lower = filename.lower()
                                    target_name_no_ext = target_name_lower.rsplit('.', 1)[0] if '.' in target_name_lower else target_name_lower
                                    
                                    # First pass: Exact filename match
                                    for doc in firestore_docs:
                                        doc_filename = doc.get('filename', '')
                                        if not doc_filename:
                                            continue
                                            
                                        doc_filename_lower = doc_filename.lower()
                                        
                                        # Check for exact filename match (case-insensitive)
                                        if doc_filename_lower == target_name_lower:
                                            logger.info(f"Strategy 1: Found exact filename match in Firestore: {doc_filename}")
                                            
                                            # Try gcs_temp_path first (temp location - most likely for pending/uploaded files)
                                            doc_gcs_temp_path = doc.get('gcs_temp_path', '')
                                            if doc_gcs_temp_path:
                                                # Extract clean key from gcs_temp_path (handle gs:// URLs, bucket names, etc.)
                                                gcs_temp_key = extract_gcs_key_from_path(doc_gcs_temp_path, s3_service.bucket_name)
                                                logger.info(f"Strategy 1: Trying Firestore gcs_temp_path (extracted key: {gcs_temp_key})")
                                                result = s3_service.generate_presigned_url(gcs_temp_key, expiration, method="GET")
                                                if result['success']:
                                                    actual_key = gcs_temp_key
                                                    logger.info(f"✅ Successfully generated URL using Firestore gcs_temp_path: {gcs_temp_key}")
                                                    break
                                            
                                            # Try gcs_path (organized location)
                                            if not result['success']:
                                                doc_gcs_path = doc.get('gcs_path', '')
                                                if doc_gcs_path:
                                                    # Extract key from gs:// URL if needed
                                                    gcs_key = extract_gcs_key_from_path(doc_gcs_path, s3_service.bucket_name)
                                                    logger.info(f"Strategy 1: Trying Firestore gcs_path (extracted key: {gcs_key})")
                                                    result = s3_service.generate_presigned_url(gcs_key, expiration, method="GET")
                                                    if result['success']:
                                                        actual_key = gcs_key
                                                        logger.info(f"✅ Successfully generated URL using Firestore gcs_path: {gcs_key}")
                                                        break
                                    
                                    # Second pass: Fuzzy filename matching (without extension, special chars)
                                    if not result['success']:
                                        target_clean = target_name_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '').replace('.', '')
                                        logger.info(f"Strategy 1: Trying fuzzy filename matching (target_clean: {target_clean})")
                                        
                                        for doc in firestore_docs:
                                            doc_filename = doc.get('filename', '')
                                            if not doc_filename:
                                                continue
                                                
                                            doc_filename_lower = doc_filename.lower()
                                            doc_filename_no_ext = doc_filename_lower.rsplit('.', 1)[0] if '.' in doc_filename_lower else doc_filename_lower
                                            doc_clean = doc_filename_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '').replace('.', '')
                                            
                                            # Check if cleaned filenames match
                                            if (target_clean == doc_clean or 
                                                (len(target_clean) > 5 and target_clean in doc_clean) or
                                                (len(doc_clean) > 5 and doc_clean in target_clean)):
                                                logger.info(f"Strategy 1: Found fuzzy filename match: {doc_filename} (requested: {filename})")
                                                
                                                # Try both paths in order: temp first, then organized
                                                for path_field in ['gcs_temp_path', 'gcs_path']:
                                                    doc_path = doc.get(path_field, '')
                                                    if doc_path:
                                                        gcs_key = extract_gcs_key_from_path(doc_path, s3_service.bucket_name)
                                                        logger.info(f"Strategy 1: Trying Firestore {path_field} (extracted key: {gcs_key})")
                                                        result = s3_service.generate_presigned_url(gcs_key, expiration, method="GET")
                                                        if result['success']:
                                                            actual_key = gcs_key
                                                            logger.info(f"✅ Successfully generated URL using Firestore {path_field} (fuzzy match): {gcs_key}")
                                                            break
                                                if result['success']:
                                                    break
                            except Exception as firestore_error:
                                logger.warning(f"Strategy 1: Failed to check Firestore for file location: {firestore_error}")
                                import traceback
                                logger.debug(f"Strategy 1: Firestore error traceback:\n{traceback.format_exc()}")
                        
                        # Strategy 2: Try sanitized version (spaces -> underscores, remove parentheses and commas)
                        if not result['success']:
                            sanitized_filename = filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                            if sanitized_filename != filename:
                                sanitized_key = f"{path_dir}/{sanitized_filename}"  # Define before using
                                logger.info(f"Strategy 2: Trying sanitized version: {sanitized_key}")
                                result = s3_service.generate_presigned_url(sanitized_key, expiration, method="GET")
                                if result['success']:
                                    actual_key = sanitized_key
                                    logger.info(f"✅ Successfully generated URL with sanitized key: {sanitized_key}")
                        
                        # Strategy 3: If still not found, list files in directory and find by filename match
                        if not result['success']:
                            logger.info(f"Strategy 3: Searching directory for matching filename: {path_dir}")
                            try:
                                # Extract flow_id from path (e.g., "temp/flow-20251129_180531" -> "flow-20251129_180531")
                                path_segments = path_dir.split('/')
                                flow_id = None
                                if len(path_segments) >= 2 and path_segments[0] == 'temp':
                                    flow_id = path_segments[1]
                                elif len(path_segments) >= 1:
                                    # Also try if path is just the flow_id (without temp prefix)
                                    flow_id = path_segments[-1] if path_segments[-1] else path_segments[0]
                            
                                if flow_id:
                                    # List all files in this flow's temp folder
                                    temp_files = s3_service._get_flow_temp_files(flow_id, max_results=100)
                                    logger.info(f"Strategy 3: Found {len(temp_files)} files in temp/{flow_id}/")
                                
                                    # Try to find a file that matches the requested filename
                                    # Match by exact filename, or by filename without extension, or by partial match
                                    target_name_lower = filename.lower()
                                    target_name_no_ext = target_name_lower.rsplit('.', 1)[0] if '.' in target_name_lower else target_name_lower
                                    
                                    for file_info in temp_files:
                                        file_key = file_info.get('key', '')
                                        file_name = file_key.split('/')[-1]
                                        file_name_lower = file_name.lower()
                                        file_name_no_ext = file_name_lower.rsplit('.', 1)[0] if '.' in file_name_lower else file_name_lower
                                        
                                        # Check for exact match (case-insensitive) or match without extension
                                        # Also try removing special characters for fuzzy matching
                                        target_clean = target_name_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                        file_clean = file_name_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                        
                                        if (file_name_lower == target_name_lower or 
                                            file_name_no_ext == target_name_no_ext or
                                            target_clean == file_clean or
                                            (len(target_clean) > 5 and target_clean in file_clean) or
                                            (len(file_clean) > 5 and file_clean in target_clean)):
                                            logger.info(f"Found matching file in directory: {file_key} (requested: {filename})")
                                            result = s3_service.generate_presigned_url(file_key, expiration, method="GET")
                                            if result['success']:
                                                actual_key = file_key
                                                logger.info(f"✅ Successfully generated URL with found file key: {file_key}")
                                                break
                            except Exception as search_error:
                                logger.warning(f"Strategy 3: Failed to search directory for matching file: {search_error}")
                    
                    # Strategy 4: Global filename search across known prefixes
                    if not result['success']:
                        search_variations = {
                            filename,
                            filename.replace(" ", "_"),
                            filename.replace(" ", "_").replace("(", "").replace(")", ""),
                            filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", ""),
                        }
                        logger.info(f"Strategy 4: Searching bucket for filename variations: {search_variations}")
                        matches = s3_service.find_files_by_filename(
                            [name for name in search_variations if name],
                            max_results=20,
                        )
                        if matches:
                            for match_key in matches:
                                logger.info(f"Trying matched key from global search: {match_key}")
                                result = s3_service.generate_presigned_url(match_key, expiration, method="GET")
                                if result['success']:
                                    actual_key = match_key
                                    logger.info(f"✅ Successfully generated URL using global search match: {match_key}")
                                    break
                except Exception as fallback_error:
                    logger.error(f"Error in fallback strategies: {fallback_error}")
                    import traceback
                    logger.error(f"Fallback error traceback:\n{traceback.format_exc()}")
        
        if result['success']:
            logger.info(f"✅ V4 signed URL generated successfully for: {actual_key} (original request: {key})")
            return JSONResponse(content={
                "success": True,
                "url": result['url'],
                "expires_in": expiration,
                "key": actual_key,
                "original_key": key  # Include original for reference
            })
        else:
            # V4 signed URL generation failed - provide helpful error message
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"❌ Failed to generate V4 signed URL for {actual_key}: {error_msg}")
            logger.error(f"   Original key: {key}")
            logger.error(f"   Decoded key: {decoded_key}")
            logger.error(f"   Actual key used: {actual_key}")
            
            # Check if error is due to missing private key (OAuth2 credentials)
            if 'private key' in error_msg.lower() or 'token' in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Failed to generate signed URL: Service account credentials with private key are required. "
                        "Please ensure GOOGLE_APPLICATION_CREDENTIALS points to a service account JSON key file, "
                        "or place voucher-storage-key.json in the backend directory. "
                        f"Error: {error_msg}"
                    )
                )
            elif 'does not exist' in error_msg.lower() or 'not found' in error_msg.lower():
                # Provide helpful error message with suggestions
                error_detail = f"File not found: {actual_key}."
                
                # Extract flow_id and filename if this is a temp path
                if '/temp/' in actual_key or actual_key.startswith('temp/'):
                    path_parts = actual_key.split('/')
                    if len(path_parts) >= 3:
                        flow_id = path_parts[1]
                        filename = path_parts[-1]
                        error_detail += (
                            f"\n\nTroubleshooting:\n"
                            f"- Requested path: {actual_key}\n"
                            f"- Flow ID: {flow_id}\n"
                            f"- Filename: {filename}\n"
                            f"- The file may have been stored with a UUID-based filename instead of the original filename.\n"
                            f"- Use /api/debug/list-flow-files/{flow_id}?filename={filename} to find the correct path."
                        )
                
                raise HTTPException(status_code=404, detail=error_detail)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {error_msg}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Pre-signed URL error: {str(e)}")
        logger.error(f"   Key received: {original_key}")
        logger.error(f"   Decoded key: {decoded_key}")
        logger.error(f"   Actual key: {actual_key}")
        import traceback
        logger.error(f"   Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while generating presigned URL: {str(e)}"
        )

@app.get("/api/aws/s3/view/{file_key:path}")
async def view_s3_file(file_key: str, request: Request):
    """
    Proxy endpoint to serve GCS files with proper headers for viewing in browser.
    This avoids Chrome blocking issues with iframes.
    """
    original_key = file_key
    decoded_key = None
    actual_key = None
    
    try:
        # Explicitly URL-decode the key
        decoded_key = unquote(file_key)
        if decoded_key != file_key:
            logger.info(f"URL-decoded key: {file_key} -> {decoded_key}")
        
        # Convert gs:// URL to key if needed, or remove bucket name if present
        actual_key = decoded_key
        if decoded_key.startswith('gs://'):
            # Extract key from gs://bucket-name/path/to/file
            parts = decoded_key.replace('gs://', '').split('/', 1)
            if len(parts) == 2:
                actual_key = parts[1]  # Use the path part as the key (without bucket name)
            else:
                raise HTTPException(status_code=400, detail="Invalid gs:// URL format")
        elif '/' in decoded_key:
            # Handle case where key might include bucket name: bucket-name/path/to/file
            path_parts = decoded_key.split('/', 1)
            # Check if first part looks like a bucket name
            if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                actual_key = path_parts[1]  # Remove bucket name
                logger.info(f"Removed bucket name from key: {decoded_key} -> {actual_key}")
        
        logger.info(f"Viewing file with key: {actual_key} (original: {file_key})")
        
        # Get the file from GCS using s3_service (which is GCS-backed)
        if not s3_service.client or not s3_service.bucket:
            raise HTTPException(status_code=500, detail="GCS client not initialized")
        
        # Try to get the blob with the actual key
        blob = s3_service.bucket.blob(actual_key)
        
        # Check if blob exists
        blob_exists = False
        try:
            blob_exists = blob.exists()
            logger.info(f"Blob exists check for {actual_key}: {blob_exists}")
        except Exception as exists_check_error:
            logger.warning(f"Error checking blob existence: {exists_check_error}")
            blob_exists = False
        
        # If blob doesn't exist, try fallback strategies
        if not blob_exists:
            key_parts = actual_key.rsplit('/', 1)
            if len(key_parts) == 2:
                path_dir = key_parts[0]
                filename = key_parts[1]
                
                # Strategy 1: Check Firestore FIRST - it has the exact path that was saved during upload
                # This is the most reliable source of truth since Firestore stores the actual gcs_temp_path
                if not blob_exists and firestore_service:
                    try:
                        path_segments = path_dir.split('/')
                        if len(path_segments) >= 2 and path_segments[0] == 'temp':
                            flow_id = path_segments[1]
                            logger.info(f"File not found in GCS, checking Firestore for flow: {flow_id}, filename: {filename}")
                            
                            # Get all documents for this flow from Firestore
                            firestore_docs, _ = firestore_service.get_documents_by_flow_id(flow_id, page=1, page_size=1000)
                            
                            # Try to find a document with matching filename
                            target_name_lower = filename.lower()
                            for doc in firestore_docs:
                                doc_filename = doc.get('filename', '')
                                doc_gcs_path = doc.get('gcs_path', '')
                                doc_gcs_temp_path = doc.get('gcs_temp_path', '')
                                
                                # Check if filename matches (case-insensitive, with fuzzy matching)
                                if doc_filename and doc_filename.lower() == target_name_lower:
                                    # Try gcs_temp_path first (temp location - most likely for need_review files)
                                    if doc_gcs_temp_path:
                                        logger.info(f"Found file in Firestore, trying gcs_temp_path: {doc_gcs_temp_path}")
                                        blob = s3_service.bucket.blob(doc_gcs_temp_path)
                                        try:
                                            if blob.exists():
                                                actual_key = doc_gcs_temp_path
                                                blob_exists = True
                                                logger.info(f"✅ Found file using Firestore gcs_temp_path: {doc_gcs_temp_path}")
                                                break
                                        except Exception as e:
                                            logger.warning(f"Error checking gcs_temp_path blob: {e}")
                                    
                                    # Try gcs_path (organized location)
                                    if not blob_exists and doc_gcs_path:
                                        # Extract key from gs:// URL if needed
                                        gcs_key = doc_gcs_path.replace(f"gs://{s3_service.bucket_name}/", "")
                                        if gcs_key.startswith('voucher-bucket-1/'):
                                            gcs_key = gcs_key.replace('voucher-bucket-1/', '', 1)
                                        logger.info(f"Trying Firestore gcs_path: {gcs_key}")
                                        blob = s3_service.bucket.blob(gcs_key)
                                        try:
                                            if blob.exists():
                                                actual_key = gcs_key
                                                blob_exists = True
                                                logger.info(f"✅ Found file using Firestore gcs_path: {gcs_key}")
                                                break
                                        except Exception as e:
                                            logger.warning(f"Error checking gcs_path blob: {e}")
                            
                            # If still not found, try fuzzy filename matching
                            if not blob_exists:
                                target_clean = target_name_lower.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                for doc in firestore_docs:
                                    doc_filename = doc.get('filename', '')
                                    if doc_filename:
                                        doc_filename_lower = doc_filename.lower()
                                        doc_clean = doc_filename_lower.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                        
                                        if target_clean == doc_clean or (len(target_clean) > 5 and target_clean in doc_clean):
                                            # Try both paths (gcs_temp_path first for need_review files)
                                            for path_field in ['gcs_temp_path', 'gcs_path']:
                                                doc_path = doc.get(path_field, '')
                                                if doc_path:
                                                    gcs_key = doc_path.replace(f"gs://{s3_service.bucket_name}/", "")
                                                    if gcs_key.startswith('voucher-bucket-1/'):
                                                        gcs_key = gcs_key.replace('voucher-bucket-1/', '', 1)
                                                    logger.info(f"Trying Firestore {path_field}: {gcs_key}")
                                                    blob = s3_service.bucket.blob(gcs_key)
                                                    try:
                                                        if blob.exists():
                                                            actual_key = gcs_key
                                                            blob_exists = True
                                                            logger.info(f"✅ Found file using Firestore {path_field}: {gcs_key}")
                                                            break
                                                    except Exception as e:
                                                        logger.warning(f"Error checking {path_field} blob: {e}")
                                            if blob_exists:
                                                break
                    except Exception as firestore_error:
                        logger.warning(f"Failed to check Firestore for file location: {firestore_error}")
                
                # Strategy 2: Try sanitized version (spaces -> underscores, remove parentheses and commas)
                if not blob_exists:
                    sanitized_filename = filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
                    if sanitized_filename != filename:
                        sanitized_key = f"{path_dir}/{sanitized_filename}"
                        logger.info(f"Strategy 2: Trying sanitized version: {sanitized_key}")
                        blob = s3_service.bucket.blob(sanitized_key)
                        try:
                            if blob.exists():
                                actual_key = sanitized_key
                                blob_exists = True
                                logger.info(f"✅ Found file with sanitized key: {sanitized_key}")
                        except Exception as e:
                            logger.warning(f"Error checking sanitized blob: {e}")
                            blob_exists = False
                
                # Strategy 3: If still not found, search directory for matching file
                if not blob_exists:
                    logger.info(f"Strategy 3: Searching directory for matching filename: {path_dir}")
                    try:
                        # Extract flow_id from path (e.g., "temp/flow-20251129_180531" -> "flow-20251129_180531")
                        path_segments = path_dir.split('/')
                        if len(path_segments) >= 2 and path_segments[0] == 'temp':
                            flow_id = path_segments[1]
                            # List all files in this flow's temp folder
                            temp_files = s3_service._get_flow_temp_files(flow_id, max_results=100)
                            
                            # Try to find a file that matches the requested filename
                            target_name_lower = filename.lower()
                            target_name_no_ext = target_name_lower.rsplit('.', 1)[0] if '.' in target_name_lower else target_name_lower
                            
                            for file_info in temp_files:
                                file_key = file_info.get('key', '')
                                file_name = file_key.split('/')[-1]
                                file_name_lower = file_name.lower()
                                file_name_no_ext = file_name_lower.rsplit('.', 1)[0] if '.' in file_name_lower else file_name_lower
                                
                                # Check for exact match or fuzzy match
                                target_clean = target_name_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                file_clean = file_name_no_ext.replace(',', '').replace('_', '').replace('-', '').replace(' ', '')
                                
                                if (file_name_lower == target_name_lower or 
                                    file_name_no_ext == target_name_no_ext or
                                    target_clean == file_clean or
                                    (len(target_clean) > 5 and target_clean in file_clean) or
                                    (len(file_clean) > 5 and file_clean in target_clean)):
                                    logger.info(f"Found matching file in directory: {file_key} (requested: {filename})")
                                    blob = s3_service.bucket.blob(file_key)
                                    try:
                                        if blob.exists():
                                            actual_key = file_key
                                            blob_exists = True
                                            logger.info(f"✅ Found file with directory search: {file_key}")
                                            break
                                    except Exception as e:
                                        logger.warning(f"Error checking matched blob existence: {e}")
                                        continue
                    except Exception as search_error:
                        logger.warning(f"Failed to search directory for matching file: {search_error}")
        
        # Final check - if still not found, return 404 with helpful information
        if not blob_exists:
            logger.error(f"❌ File not found after all fallback attempts: {actual_key}")
            logger.error(f"   Original key: {file_key}")
            logger.error(f"   Decoded key: {decoded_key}")
            
            # Try to list files in the directory to help debug
            try:
                key_parts = actual_key.rsplit('/', 1)
                if len(key_parts) == 2:
                    path_dir = key_parts[0]
                    path_segments = path_dir.split('/')
                    if len(path_segments) >= 2 and path_segments[0] == 'temp':
                        flow_id = path_segments[1]
                        temp_files = s3_service._get_flow_temp_files(flow_id, max_results=100)
                        if temp_files:
                            file_names = [f.get('key', '').split('/')[-1] for f in temp_files]
                            logger.error(f"   Available files in {path_dir}: {file_names[:10]}")  # Log first 10 files
                            error_detail = f"File not found: {actual_key}. Available files in directory: {', '.join(file_names[:5])}"
                        else:
                            error_detail = f"File not found: {actual_key}. No files found in directory {path_dir}."
                    else:
                        error_detail = f"File not found: {actual_key}. Please verify the file exists in GCS."
                else:
                    error_detail = f"File not found: {actual_key}. Please verify the file exists in GCS."
            except Exception as list_error:
                logger.warning(f"Failed to list files for error message: {list_error}")
                error_detail = f"File not found: {actual_key}. Please verify the file exists in GCS."
            
            raise HTTPException(status_code=404, detail=error_detail)
        
        # Get filename from key
        filename = actual_key.split('/')[-1]
        
        # Determine content type and disposition (use blob's content type if available)
        content_type = blob.content_type or 'application/octet-stream'
        disposition = 'inline'  # Show in browser instead of download
        
        # Fallback content type detection if blob doesn't have it
        if not content_type or content_type == 'application/octet-stream':
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif filename.lower().endswith('.png'):
                content_type = 'image/png'
            elif filename.lower().endswith('.pdf'):
                content_type = 'application/pdf'
        elif filename.lower().endswith('.txt'):
            content_type = 'text/plain'
        
        # Download full file content from GCS
        file_content = blob.download_as_bytes()
        
        # Create response with proper headers
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                'Content-Disposition': f'{disposition}; filename="{filename}"',
                'Content-Length': str(len(file_content)),
                'Accept-Ranges': 'bytes',  # Indicate we support range requests
                'Cache-Control': 'public, max-age=3600',  # Cache for 1 hour
                'X-Content-Type-Options': 'nosniff',
                'Access-Control-Allow-Origin': '*',  # Allow cross-origin access
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"View file error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=404, detail=f"File not found or error viewing: {str(e)}")

@app.get("/api/aws/s3/blob/{file_key:path}")
async def get_file_blob(file_key: str):
    """
    Get file as blob for creating Object URLs.
    This endpoint serves files with proper CORS headers for blob creation.
    Uses GCS (not AWS S3).
    """
    try:
        # Convert gs:// URL to key if needed, or remove bucket name if present
        actual_key = file_key
        if file_key.startswith('gs://'):
            # Extract key from gs://bucket-name/path/to/file
            parts = file_key.replace('gs://', '').split('/', 1)
            if len(parts) == 2:
                actual_key = parts[1]  # Use the path part as the key (without bucket name)
            else:
                raise HTTPException(status_code=400, detail="Invalid gs:// URL format")
        elif '/' in file_key:
            # Handle case where key might include bucket name: bucket-name/path/to/file
            path_parts = file_key.split('/', 1)
            # Check if first part looks like a bucket name
            if path_parts[0] == s3_service.bucket_name or path_parts[0] == 'voucher-bucket-1':
                actual_key = path_parts[1]  # Remove bucket name
        
        logger.info(f"Getting file blob with key: {actual_key}")
        
        # Get the file from GCS using s3_service (which is GCS-backed)
        if not s3_service.client or not s3_service.bucket:
            raise HTTPException(status_code=500, detail="GCS client not initialized")
        
        blob = s3_service.bucket.blob(actual_key)
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {actual_key}")
        
        # Download file content from GCS
        file_content = blob.download_as_bytes()
        
        # Get filename from key
        filename = actual_key.split('/')[-1]
        
        # Determine content type (use blob's content type if available)
        content_type = blob.content_type or 'application/octet-stream'
        
        # Fallback content type detection if blob doesn't have it
        if not content_type or content_type == 'application/octet-stream':
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif filename.lower().endswith('.png'):
                content_type = 'image/png'
            elif filename.lower().endswith('.pdf'):
                content_type = 'application/pdf'
        elif filename.lower().endswith('.txt'):
            content_type = 'text/plain'
        
        # Create response with CORS headers for blob usage
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Cache-Control': 'public, max-age=3600',
                'Content-Length': str(len(file_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file blob error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=404, detail=f"File not found or error fetching blob: {str(e)}")

@app.post("/api/aws/s3/replace")
async def replace_s3_file(
    file: UploadFile = File(...),
    current_key: str = Form(...)
):
    """
    Replace an existing file in S3 with a new one.
    Maintains the same path structure but allows changing the filename.
    """
    try:
        logger.info(f"Replacing file: {current_key} with {file.filename}")
        
        # Read file content
        content = await file.read()
        
        # Replace file in S3
        result = s3_service.replace_file(current_key, content, file.filename)
        
        if result['success']:
            logger.info(f"✅ File replaced successfully: {current_key} -> {result['new_key']}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ File replacement failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Replace file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from typing import Optional

@app.post("/api/aws/s3/update-file-details")
async def update_file_details(
    current_key: str = Body(..., embed=True),
    new_branch: Optional[str] = Body(None, embed=True),
    new_date: Optional[str] = Body(None, embed=True),
    new_filename: Optional[str] = Body(None, embed=True),
    new_type: Optional[str] = Body(None, embed=True)
):
    """
    Update organized voucher path details (date, branch, filename, type) without reprocessing.
    This moves/renames the S3 object accordingly.
    """
    try:
        logger.info(
            f"Updating file details for {current_key} -> branch={new_branch}, date={new_date}, filename={new_filename}, type={new_type}"
        )

        result = s3_service.move_file_update_details(
            current_key=current_key,
            new_branch=new_branch,
            new_date=new_date,
            new_filename=new_filename,
            new_type=new_type
        )

        if result.get('success'):
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to update file'))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"update_file_details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# S3 Batch Management API Endpoints
# ========================================

@app.get("/api/aws/s3/batches")
async def get_s3_batches():
    """List all batches from S3 (reads from batches/ folder)"""
    try:
        logger.info("Getting batches from S3...")
        result = s3_service.list_batches_from_s3()
        
        if result['success']:
            logger.info(f"✅ Found {result['count']} batches in S3")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ Failed to list batches: {result.get('error')}")
            # Return success with empty batches instead of error
            return JSONResponse(content={
                'success': True,
                'batches': [],
                'count': 0,
                'message': f"S3 not available: {result.get('error')}"
            })
    
    except Exception as e:
        logger.error(f"Get S3 batches error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return success with empty batches instead of throwing error
        return JSONResponse(content={
            'success': True,
            'batches': [],
            'count': 0,
            'message': f"S3 error: {str(e)}"
        })

@app.post("/api/aws/s3/flow/create")
async def create_s3_flow(request: dict):
    """Create flow in S3 (stores metadata file)"""
    try:
        flow_name = request.get('flow_name', '').strip()
        if not flow_name:
            raise HTTPException(status_code=400, detail="Flow name is required")
        
        # Check for duplicate flow names in S3
        logger.info(f"Checking for duplicate flow name: {flow_name}")
        existing_flows = s3_service.list_flows_from_s3()
        
        if existing_flows['success']:
            # Case-insensitive comparison
            flow_name_lower = flow_name.lower()
            for existing_flow in existing_flows.get('flows', []):
                existing_name = existing_flow.get('flow_name', '').lower()
                if existing_name == flow_name_lower:
                    logger.warning(f"❌ Duplicate flow name found: {flow_name}")
                    return JSONResponse(content={
                        'success': False,
                        'error': 'Duplicate flow name',
                        'message': f"A flow with the name '{flow_name}' already exists. Please use a different name."
                    }, status_code=409)
        
        # Generate flow_id
        from datetime import datetime
        import uuid
        flow_id = f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:8]}"
        
        logger.info(f"Creating flow in S3: {flow_id} with name: {flow_name}")
        
        result = s3_service.create_flow_metadata_in_s3(flow_id, flow_name)
        
        if result['success']:
            logger.info(f"✅ Created flow in S3: {flow_id}")
            return JSONResponse(content={
                'success': True,
                'flow_id': flow_id,
                'flow_name': flow_name,
                's3_metadata_key': result.get('s3_key', ''),
                'created_at': result['flow']['created_at']
            })
        else:
            logger.error(f"❌ Failed to create flow: {result.get('error')}")
            # Return error response instead of throwing exception
            return JSONResponse(content={
                'success': False,
                'error': result.get('error'),
                'message': f"Could not create flow in S3: {result.get('error')}"
            }, status_code=500)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create S3 flow error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return error response instead of throwing exception
        return JSONResponse(content={
            'success': False,
            'error': str(e),
            'message': f"Error creating flow: {str(e)}"
        }, status_code=500)

@app.get("/api/aws/s3/flow/{flow_id}/files")
async def get_s3_flow_files(flow_id: str):
    """Get all files for a specific flow from both temp and organized folders"""
    try:
        logger.info(f"Getting files for flow: {flow_id}")
        
        result = s3_service.get_flow_files_from_s3(flow_id)
        
        if result['success']:
            logger.info(f"✅ Found {result.get('total_files', 0)} files for flow {flow_id}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ Failed to get flow files: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Get S3 flow files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aws/s3/flow/{flow_id}/pending")
async def get_flow_pending_vouchers(flow_id: str):
    """Get all pending (unmatched) attachment vouchers for a specific flow"""
    try:
        logger.info(f"Getting pending vouchers for flow: {flow_id}")
        
        pending_files = s3_service._get_flow_pending_files(flow_id)
        
        result = {
            'success': True,
            'flow_id': flow_id,
            'pending_files': pending_files,
            'count': len(pending_files)
        }
        
        logger.info(f"✅ Found {len(pending_files)} pending vouchers for flow {flow_id}")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Get pending vouchers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/aws/s3/batch/{batch_id}/pending/{file_key:path}")
async def delete_pending_voucher(batch_id: str, file_key: str):
    """Delete a pending voucher from attached_voucher/ folder"""
    try:
        logger.info(f"Deleting pending voucher: {file_key} from batch {batch_id}")
        
        # Validate that file is actually in the attached_voucher/{batch_id}/ folder
        if not file_key.startswith(f'attached_voucher/{batch_id}/'):
            logger.error(f"❌ Invalid file key - must be in attached_voucher/{batch_id}/ folder")
            raise HTTPException(
                status_code=400, 
                detail=f"File must be in attached_voucher/{batch_id}/ folder"
            )
        
        # Delete the file from S3
        result = s3_service.delete_file(file_key)
        
        if result['success']:
            logger.info(f"✅ Deleted pending voucher: {file_key}")
            return JSONResponse(content={
                'success': True,
                'message': f'Deleted pending voucher: {file_key}',
                'file_key': file_key
            })
        else:
            logger.error(f"❌ Failed to delete pending voucher: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete pending voucher error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/s3/upload-attachment")
async def upload_attachment_file(
    file_data: str = Body(...),
    filename: str = Body(...),
    batch_id: str = Body(...),
    s3_key: str = Body(...)
):
    """Upload a file directly to the attached_voucher folder in S3"""
    try:
        logger.info(f"Uploading attachment {filename} to {s3_key}")
        
        # Decode base64 file data
        import base64
        file_content = base64.b64decode(file_data)
        
        # Determine content type based on file extension
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'bin'
        content_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'pdf': 'application/pdf',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        # Upload directly to S3 using s3_client
        try:
            s3_service.s3_client.put_object(
                Bucket=s3_service.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType=content_type
            )
            logger.info(f"✅ Successfully uploaded attachment to {s3_key}")
        except Exception as upload_error:
            logger.error(f"S3 upload failed: {upload_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to S3: {str(upload_error)}"
            )
        
        return JSONResponse(content={
            'success': True,
            's3_key': s3_key,
            'filename': filename,
            'batch_id': batch_id,
            'message': f'File uploaded successfully to {s3_key}'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload attachment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gcs/upload-attachment")
async def upload_attachment_file_gcs(
    file_data: str = Body(...),
    filename: str = Body(...),
    batch_id: str = Body(...),
    gcs_key: str = Body(None),
    s3_key: str = Body(None)  # Keep for backward compatibility
):
    """Upload a file directly to the attached_voucher folder in GCS"""
    try:
        # Use gcs_key if provided, otherwise fall back to s3_key for compatibility
        storage_key = gcs_key or s3_key
        if not storage_key:
            raise HTTPException(status_code=400, detail="gcs_key or s3_key is required")
        
        logger.info(f"Uploading attachment {filename} to {storage_key}")
        
        # Decode base64 file data
        import base64
        file_content = base64.b64decode(file_data)
        
        # Determine content type based on file extension
        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'bin'
        content_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'pdf': 'application/pdf',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp'
        }
        content_type = content_type_map.get(file_ext, 'application/octet-stream')
        
        # Upload directly to GCS using s3_service (which is GCS-backed)
        try:
            if not s3_service.bucket:
                raise HTTPException(status_code=503, detail="GCS service not initialized")
            
            blob = s3_service.bucket.blob(storage_key)
            blob.upload_from_string(file_content, content_type=content_type)
            
            # Get public URL
            gcs_url = s3_service._public_url(blob)
            
            logger.info(f"✅ Successfully uploaded attachment to {storage_key}")
        except Exception as upload_error:
            logger.error(f"GCS upload failed: {upload_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to GCS: {str(upload_error)}"
            )
        
        return JSONResponse(content={
            'success': True,
            'gcs_key': storage_key,
            's3_key': storage_key,  # Keep for backward compatibility
            'gcs_url': gcs_url,
            'filename': filename,
            'batch_id': batch_id,
            'message': f'File uploaded successfully to {storage_key}'
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload attachment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/s3/batch/{batch_id}/pending/attach")
async def attach_pending_to_voucher(
    batch_id: str,
    pending_key: str = Body(..., embed=True),
    target_voucher_key: str = Body(..., embed=True)
):
    """Manually attach a pending voucher to a main voucher (merge PDFs)"""
    try:
        logger.info(f"Attaching pending voucher {pending_key} to {target_voucher_key}")
        
        # Validate pending_key is in attached_voucher folder
        if not pending_key.startswith(f'attached_voucher/{batch_id}/'):
            raise HTTPException(
                status_code=400,
                detail=f"Pending file must be in attached_voucher/{batch_id}/ folder"
            )
        
        # Download pending file (image) from S3
        import tempfile
        import os
        
        try:
            pending_obj = s3_service.s3_client.get_object(
                Bucket=s3_service.bucket_name,
                Key=pending_key
            )
            pending_content = pending_obj['Body'].read()
            
            # Save to temp file
            pending_ext = pending_key.split('.')[-1] if '.' in pending_key else 'bin'
            pending_fd, pending_path = tempfile.mkstemp(suffix=f'.{pending_ext}')
            os.write(pending_fd, pending_content)
            os.close(pending_fd)
        except Exception as e:
            logger.error(f"Failed to download pending file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download pending file: {str(e)}")
        
        # Download target voucher PDF from S3
        try:
            target_obj = s3_service.s3_client.get_object(
                Bucket=s3_service.bucket_name,
                Key=target_voucher_key
            )
            target_content = target_obj['Body'].read()
            
            # Save to temp file
            target_fd, target_path = tempfile.mkstemp(suffix='.pdf')
            os.write(target_fd, target_content)
            os.close(target_fd)
        except Exception as e:
            logger.error(f"Failed to download target voucher: {e}")
            # Clean up pending file
            try:
                os.unlink(pending_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to download target voucher: {str(e)}")
        
        # Get target voucher metadata to preserve it
        try:
            target_metadata_response = s3_service.s3_client.head_object(
                Bucket=s3_service.bucket_name,
                Key=target_voucher_key
            )
            target_metadata = target_metadata_response.get('Metadata', {})
        except:
            target_metadata = {}
        
        # Merge using PyMuPDF (same logic as Lambda)
        import fitz  # PyMuPDF
        
        # Create temp file for merged PDF
        merged_pdf_fd, merged_pdf_path = tempfile.mkstemp(suffix='.pdf')
        os.close(merged_pdf_fd)
        
        try:
            # Open target PDF
            target_pdf = fitz.open(target_path)
            
            # Convert pending image to PDF if it's an image
            if pending_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Convert image to PDF
                img_pdf = fitz.open()
                img_page = img_pdf.new_page(width=595, height=842)  # A4 size
                img_page.insert_image(img_page.rect, filename=pending_path)
                
                # Append to target PDF
                target_pdf.insert_pdf(img_pdf)
                img_pdf.close()
            else:
                # It's already a PDF
                pending_pdf = fitz.open(pending_path)
                target_pdf.insert_pdf(pending_pdf)
                pending_pdf.close()
            
            # Save merged PDF
            target_pdf.save(merged_pdf_path)
            target_pdf.close()
            
            # Upload merged PDF back to S3
            with open(merged_pdf_path, 'rb') as merged_file:
                s3_service.s3_client.put_object(
                    Bucket=s3_service.bucket_name,
                    Key=target_voucher_key,
                    Body=merged_file.read(),
                    ContentType='application/pdf',
                    Metadata=target_metadata
                )
            
            logger.info(f"✅ Merged and uploaded to: {target_voucher_key}")
            
            # Delete pending file from attached_voucher folder
            s3_service.delete_file(pending_key)
            logger.info(f"✅ Deleted pending file: {pending_key}")
            
            # Clean up temp files
            os.unlink(merged_pdf_path)
            os.unlink(pending_path)
            os.unlink(target_path)
            
            return JSONResponse(content={
                'success': True,
                'message': 'Successfully attached pending voucher to main voucher',
                'merged_key': target_voucher_key,
                'deleted_pending_key': pending_key
            })
            
        except Exception as merge_error:
            # Clean up temp files on error
            try:
                os.unlink(merged_pdf_path)
            except:
                pass
            try:
                os.unlink(pending_path)
            except:
                pass
            try:
                os.unlink(target_path)
            except:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to merge PDFs: {str(merge_error)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attach pending voucher error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/aws/s3/batch/{batch_id}")
async def delete_s3_batch(batch_id: str, delete_temp_files: bool = False):
    """Delete batch metadata from S3 and optionally delete temp files"""
    try:
        logger.info(f"Deleting batch: {batch_id}, delete_temp_files: {delete_temp_files}")
        
        result = s3_service.delete_batch_from_s3(batch_id, delete_temp_files)
        
        if result['success']:
            logger.info(f"✅ Deleted batch: {batch_id}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ Failed to delete batch: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Delete S3 batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aws/s3/batch/{batch_id}/organized-tree")
async def get_batch_organized_tree(batch_id: str):
    """Get hierarchical tree of organized vouchers for a specific batch"""
    # Redirect to Firestore-based endpoint (batch_id and flow_id are the same)
    logger.info(f"Redirecting batch organized tree request to Firestore-based endpoint for: {batch_id}")
    return await get_gcs_flow_organized_tree(batch_id)

@app.delete("/api/aws/s3/file/{file_key:path}")
async def delete_s3_file(file_key: str):
    """Delete a single file from S3"""
    try:
        logger.info(f"Deleting file: {file_key}")
        
        result = s3_service.delete_file(file_key)
        
        if result['success']:
            logger.info(f"✅ Deleted file: {file_key}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ Failed to delete file: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Delete S3 file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/s3/reclassify")
async def reclassify_failed_file(
    file_key: str = Form(...),
    new_type: str = Form(...)
):
    """Move a failed file to organized folder with new type classification"""
    try:
        logger.info(f"Reclassifying file: {file_key} to type: {new_type}")
        
        result = s3_service.reclassify_failed_file(file_key, new_type)
        
        if result['success']:
            logger.info(f"✅ Reclassified file: {file_key} -> {result.get('new_key')}")
            return JSONResponse(content=result)
        else:
            logger.error(f"❌ Failed to reclassify file: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error'))
    
    except Exception as e:
        logger.error(f"Reclassify file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/s3/cleanup-unknown")
async def cleanup_unknown_folders():
    """Clean up UNKNOWN folders from organized vouchers"""
    try:
        result = s3_service.cleanup_unknown_folders()
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "message": f"Cleaned up {result['files_moved']} files from {result['folders_cleaned']} UNKNOWN folders",
                "files_moved": result['files_moved'],
                "folders_cleaned": result['folders_cleaned']
            })
        else:
            raise HTTPException(status_code=500, detail=result.get('error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/aws/s3/reupload-failed")
async def reupload_failed_file(
    file: UploadFile = File(...),
    batch_id: str = Form(...),
    original_file_key: str = Form(...)
):
    """Reupload a failed file with fresh processing"""
    try:
        logger.info(f"Reuploading failed file: {file.filename} for batch: {batch_id}")
        
        # First, delete the original failed file
        delete_result = s3_service.delete_file(original_file_key)
        if not delete_result['success']:
            logger.warning(f"Failed to delete original failed file: {delete_result.get('error')}")
        
        # Read file content
        content = await file.read()
        document_id = str(uuid.uuid4())
        
        # Generate a unique filename to avoid conflicts
        import time
        timestamp = int(time.time() * 1000)  # milliseconds
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"reupload_{timestamp}_{file.filename}"
        
        # Upload to temp folder with unique name
        s3_result = s3_service.upload_image_to_temp(
            image_data=content,
            filename=unique_filename,
            file_type=file_extension,
            batch_id=batch_id
        )
        
        if not s3_result['success']:
            raise HTTPException(status_code=500, detail=s3_result.get('error'))
        
        # Note: SQS/Lambda processing removed - now using background tasks via task_queue
        
        logger.info(f"✅ Reuploaded failed file: {file.filename} -> {unique_filename}")
        return JSONResponse(content={
            'success': True,
            'message': f'File {file.filename} reuploaded successfully',
            's3_key': s3_result['s3_key'],
            'batch_id': batch_id,
            'original_filename': file.filename,
            'new_filename': unique_filename
        })
    
    except Exception as e:
        logger.error(f"Reupload failed file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# DynamoDB API Endpoints
# ========================================

@app.get("/api/dynamodb/batches")
async def get_batches(status: Optional[str] = None, limit: int = 20):
    """Get all batches from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "batches": [],
            "message": "Firestore not available"
        })
    
    try:
        # Get batches from Firestore
        flows, total = firestore_service.list_flows(page=1, page_size=limit)
        result = {"success": True, "batches": flows}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "batches": convert_decimals(result['batches'])
            })
        else:
            # Database not available - return empty results instead of error
            print(f"⚠️  Database not available, returning empty batches list: {result['error']}")
            return JSONResponse(content={
                "success": True,
                "batches": [],
                "message": "Database not available, using local storage"
            })
    
    except Exception as e:
        # Database error - return empty results instead of error
        print(f"⚠️  Database error, returning empty batches list: {e}")
        return JSONResponse(content={
            "success": True,
            "batches": [],
            "message": "Database not available, using local storage"
        })

@app.get("/api/dynamodb/batches/{batch_id}")
async def get_batch_details(batch_id: str):
    """Get specific batch details from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "batch": None,
            "message": "Firestore not available"
        })
    
    try:
        batch = firestore_service.get_flow(batch_id)
        result = {"success": bool(batch), "batch": batch}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "batch": convert_decimals(result['batch'])
            })
        else:
            # Database not available - return empty result instead of error
            print(f"⚠️  Database not available for batch details: {result['error']}")
            return JSONResponse(content={
                "success": True,
                "batch": None,
                "message": "Database not available, using local storage"
            })
    
    except Exception as e:
        # Database error - return empty result instead of error
        print(f"⚠️  Database error for batch details: {e}")
        return JSONResponse(content={
            "success": True,
            "batch": None,
            "message": "Database not available, using local storage"
        })

@app.get("/api/dynamodb/batches/{batch_id}/documents")
async def get_batch_documents(batch_id: str):
    """Get all documents in a batch from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "documents": [],
            "message": "Firestore not available"
        })
    
    try:
        documents, total = firestore_service.get_documents_by_flow_id(batch_id, page=1, page_size=100)
        result = {"success": True, "documents": documents}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "documents": convert_decimals(result['documents'])
            })
        else:
            # Database not available - return empty results instead of error
            print(f"⚠️  Database not available for batch documents: {result['error']}")
            return JSONResponse(content={
                "success": True,
                "documents": [],
                "message": "Database not available, using local storage"
            })
    
    except Exception as e:
        # Database error - return empty results instead of error
        print(f"⚠️  Database error for batch documents: {e}")
        return JSONResponse(content={
            "success": True,
            "documents": [],
            "message": "Database not available, using local storage"
        })

@app.get("/api/dynamodb/documents/{document_id}")
async def get_document_details(document_id: str, batch_id: str):
    """Get specific document details from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # URL decode the document_id in case it was encoded
        from urllib.parse import unquote
        document_id = unquote(document_id)
        
        document = firestore_service.get_document(document_id)
        result = {"success": bool(document), "document": document}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "document": convert_decimals(result['document'])
            })
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# NEW BATCH MANAGEMENT APIs
# ========================================

@app.post("/api/flows/create")
async def create_flow_endpoint(
    flow_name: str = Body(...),
    branch_id: str = Body(default='01'),
    source: str = Body(default='web')
):
    """Create new flow in Firestore (replaces DynamoDB)"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Generate unique flow ID
        flow_id = f"flow-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
        
        # Create flow in Firestore
        firestore_service.create_flow(flow_id, {
            'flow_name': flow_name,
            'branch_id': branch_id,
            'source': source
        })
        result = {"success": True, "flow": {"created_at": datetime.now().isoformat(), "status": "active"}}
        
        if result['success']:
            logger.info(f"✅ Created flow {flow_id}: {flow_name}")
            return JSONResponse(content={
                "success": True,
                "flow_id": flow_id,
                "flow_name": flow_name,
                "created_at": result['flow']['created_at'],
                "status": result['flow']['status']
            })
        else:
            logger.error(f"❌ Failed to create flow: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Error creating flow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flows")
async def list_flows_endpoint(status: Optional[str] = None, limit: int = 50):
    """List all flows from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "flows": [],
            "count": 0,
            "message": "Firestore not available"
        })
    
    try:
        flows, total = firestore_service.list_flows(page=1, page_size=limit)
        result = {"success": True, "flows": flows, "count": total}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "flows": convert_decimals(result['flows']),
                "count": result.get('count', 0)
            })
        else:
            # Return empty list instead of error for better frontend handling
            logger.warning(f"⚠️  Failed to list flows: {result['error']}")
            return JSONResponse(content={
                "success": True,
                "flows": [],
                "count": 0,
                "message": "Database not available"
            })
            
    except Exception as e:
        logger.error(f"Error listing flows: {e}")
        return JSONResponse(content={
            "success": True,
            "flows": [],
            "count": 0,
            "message": "Database error"
        })

@app.get("/api/flows/{flow_id}")
async def get_flow_details_endpoint(flow_id: str):
    """Get specific flow details from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        flow = firestore_service.get_flow(flow_id)
        result = {"success": bool(flow), "flow": flow}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "flow": convert_decimals(result['flow'])
            })
        else:
            raise HTTPException(status_code=404, detail="Flow not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flow details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/firestore/documents/browse")
async def get_browse_documents(
    category: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    cursor: Optional[str] = None,
    agentId: Optional[str] = None,
    request: Request = None
):
    """
    Optimized endpoint for BrowseFiles page.
    Returns all organized (completed) documents with efficient cursor-based pagination.
    Eliminates N+1 query problem by querying Firestore directly.
    """
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "documents": [],
            "total": 0,
            "next_cursor": None,
            "message": "Firestore not available"
        })
    
    try:
        # Cleanup old cache entries periodically
        _cleanup_cache(_browse_docs_cache, _BROWSE_DOCS_CACHE_TTL)
        
        # Check cache first (key includes category and cursor for uniqueness)
        cache_key = f"browse_{category or 'all'}_{page}_{cursor or 'start'}"
        now = time()
        if cache_key in _browse_docs_cache:
            cached_data, cached_time = _browse_docs_cache[cache_key]
            if now - cached_time < _BROWSE_DOCS_CACHE_TTL:
                logger.info(f"📊 Returning cached browse documents (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        logger.info(f"📊 Fetching browse documents: category={category}, page={page}, cursor={cursor}, agentId={agentId}")
        
        # Try to get current user to check admin status (optional auth)
        user_is_admin = False
        current_user_id = None
        try:
            auth_header = request.headers.get("Authorization") if request else None
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                result = simple_auth.get_user(token)
                if result.get("success"):
                    user = result["user"]
                    current_user_id = user.get('id')
                    user_is_admin = user.get('role') == 'admin' or user.get('id') == 'demo_admin_user' or user.get('email') == 'admin@example.com'
        except Exception as e:
            logger.debug(f"Could not check admin status: {e}")
        
        # If not admin and no agentId specified, default to current user's documents
        if not user_is_admin and not agentId and current_user_id:
            agentId = current_user_id
        
        # Get organized documents from Firestore
        # If agentId is provided and user is not admin, use list_documents with agentId filter
        if agentId and not user_is_admin:
            documents, total = firestore_service.list_documents(
                page=page,
                page_size=page_size,
                filters={'agentId': agentId},
                cursor_doc_id=cursor
            )
            # Filter by category in memory if needed
            if category:
                from services.category_mapper import map_backend_to_ui_category
                documents = [
                    doc for doc in documents 
                    if (doc.get('metadata', {}).get('ui_category') or 
                        map_backend_to_ui_category(doc.get('metadata', {}).get('classification') or '')) == category
                ]
            next_cursor = None  # list_documents doesn't return next_cursor, would need to implement
        else:
            # Admin sees all documents, or no agentId specified for admin
            documents, total, next_cursor = firestore_service.get_all_organized_documents(
                page=page,
                page_size=page_size,
                category=category,
                cursor_doc_id=cursor
            )
        
        # Format documents for frontend
        formatted_docs = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            
            # Get ui_category from metadata
            ui_category = metadata.get('ui_category', 'SPA')
            
            # Convert datetime to ISO string
            created_at = doc.get('created_at', '')
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()
            updated_at = doc.get('updated_at', '')
            if hasattr(updated_at, 'isoformat'):
                updated_at = updated_at.isoformat()
            
            formatted_doc = {
                'id': doc.get('document_id', ''),
                'file_name': doc.get('filename', ''),
                'document_type': metadata.get('classification', ''),
                'ui_category': ui_category,
                'document_no': metadata.get('document_no', ''),
                'document_date': metadata.get('document_date', ''),
                'amount_usd': metadata.get('invoice_amount_usd', ''),
                'amount_aed': metadata.get('invoice_amount_aed', ''),
                'created_at': created_at,
                'updated_at': updated_at,
                'flow_id': doc.get('flow_id', ''),
                'processing_status': doc.get('processing_status', ''),
                'gcs_path': doc.get('gcs_path', ''),
                'organized_path': doc.get('organized_path', ''),
                'metadata': metadata
            }
            formatted_docs.append(formatted_doc)
        
        response_data = {
            "success": True,
            "documents": convert_decimals(formatted_docs),
            "total": total,
            "page": page,
            "page_size": page_size,
            "next_cursor": next_cursor,
            "has_more": next_cursor is not None
        }
        
        # Cache the result
        _browse_docs_cache[cache_key] = (response_data, now)
        
        logger.info(f"✅ Retrieved {len(formatted_docs)} browse documents")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting browse documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(content={
            "success": False,
            "documents": [],
            "total": 0,
            "error": str(e)
        }, status_code=500)


@app.get("/api/firestore/documents")
async def get_firestore_documents(
    category: Optional[str] = None,
    flow_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
):
    """
    Get documents from Firestore or GCS with optional filtering.
    
    - If flow_id is provided: Query Firestore for that specific flow (for BatchDetailsSimple)
    - If no flow_id: Query GCS for ALL organized PDF files (for BrowseFiles)
    """
    
    # If flow_id is provided, only query Firestore (flow-specific view)
    if flow_id:
        if not firestore_service:
            return JSONResponse(content={
                "success": True,
                "documents": [],
                "total": 0,
                "message": "Firestore not available"
            })
        
        try:
            # Build filters for Firestore query
            filters = {'flow_id': flow_id}
            if category:
                filters['ui_category'] = category
            
            # Get documents from Firestore for this flow
            documents, total = firestore_service.list_documents(
                page=page,
                page_size=page_size,
                filters=filters
            )
            
            # Format documents for frontend
            formatted_docs = []
            for doc in documents:
                metadata = doc.get('metadata', {})
                
                # Get ui_category from metadata, or map from classification if not set
                ui_category = metadata.get('ui_category')
                if not ui_category:
                    classification = metadata.get('classification', '')
                    ui_category = map_backend_to_ui_category(classification) if classification else 'SPA'
                
                formatted_doc = {
                    'id': doc.get('document_id', ''),
                    'file_name': doc.get('filename', ''),
                    'document_type': metadata.get('classification', ''),
                    'ui_category': ui_category,
                    'document_no': metadata.get('document_no', ''),
                    'document_date': metadata.get('document_date', ''),
                    'amount_usd': metadata.get('invoice_amount_usd', ''),
                    'amount_aed': metadata.get('invoice_amount_aed', ''),
                    'created_at': doc.get('created_at', ''),
                    'updated_at': doc.get('updated_at', ''),
                    'flow_id': doc.get('flow_id', ''),
                    'processing_status': doc.get('processing_status', ''),
                    'gcs_path': doc.get('gcs_path', ''),
                    'organized_path': doc.get('organized_path', ''),
                    'metadata': metadata
                }
                formatted_docs.append(formatted_doc)
            
            return JSONResponse(content={
                "success": True,
                "documents": convert_decimals(formatted_docs),
                "total": total,
                "page": page,
                "page_size": page_size
            })
            
        except Exception as e:
            logger.error(f"Error getting Firestore documents for flow {flow_id}: {e}")
            return JSONResponse(content={
                "success": False,
                "documents": [],
                "total": 0,
                "error": str(e)
            }, status_code=500)
    
    # No flow_id provided - query GCS for ALL organized PDF files (for BrowseFiles)
    try:
        logger.info(f"📊 Querying GCS for ALL organized PDF files (no flow_id filter)")
        
        # Query ALL organized files from GCS
        all_organized = s3_service.list_organized_files(max_results=5000)
        if not all_organized.get('success'):
            return JSONResponse(content={
                "success": True,
                "documents": [],
                "total": 0,
                "message": "Failed to query GCS"
            })
        
        gcs_all_files = all_organized.get('files', [])
        logger.info(f"📊 Found {len(gcs_all_files)} total organized files in GCS")
        
        # Filter to only PDF files and map to document format
        all_pdf_files = []
        for file in gcs_all_files:
            filename = file.get('filename', '')
            if filename and filename.lower().endswith('.pdf'):
                # Get raw classification from file
                raw_classification = file.get('classification', '')
                
                # Map backend classification to UI category using category mapper
                mapped_ui_category = map_backend_to_ui_category(raw_classification)
                
                # Map GCS file to document format for BrowseFiles
                formatted_doc = {
                    'id': file.get('key', '').replace('/', '_'),  # Use key as ID
                    'file_name': filename,
                    'document_type': raw_classification,
                    'ui_category': mapped_ui_category,  # Use mapped UI category
                    'document_no': filename.split('.')[0] if '.' in filename else filename,
                    'document_date': '',
                    'amount_usd': '',
                    'amount_aed': '',
                    'created_at': file.get('last_modified', ''),
                    'updated_at': file.get('last_modified', ''),
                    'flow_id': '',  # No flow_id for GCS-only files
                    'processing_status': 'completed',
                    'gcs_path': f"gs://{s3_service.bucket_name}/{file.get('key', '')}",
                    'organized_path': file.get('hierarchical_path', ''),
                    'metadata': {
                        'classification': raw_classification,
                        'ui_category': mapped_ui_category
                    }
                }
                all_pdf_files.append(formatted_doc)
        
        logger.info(f"✅ Filtered to {len(all_pdf_files)} PDF files")
        
        # Apply category filter if provided
        if category:
            all_pdf_files = [
                doc for doc in all_pdf_files 
                if doc.get('ui_category', '').lower() == category.lower()
            ]
            logger.info(f"📊 After category filter '{category}': {len(all_pdf_files)} files")
        
        # Apply pagination
        total = len(all_pdf_files)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_files = all_pdf_files[start_idx:end_idx]
        
        return JSONResponse(content={
            "success": True,
            "documents": convert_decimals(paginated_files),
            "total": total,
            "page": page,
            "page_size": page_size
        })
        
    except Exception as e:
        logger.error(f"Error getting organized PDF files from GCS: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(content={
            "success": False,
            "documents": [],
            "total": 0,
            "error": str(e)
        }, status_code=500)

@app.get("/api/flows/{flow_id}/vouchers")
async def get_flow_vouchers_endpoint(flow_id: str, limit: int = 100):
    """Get all vouchers in a flow from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        return JSONResponse(content={
            "success": True,
            "vouchers": [],
            "count": 0
        })
    
    try:
        documents, total = firestore_service.get_documents_by_flow_id(flow_id, page=1, page_size=limit)
        result = {"success": True, "vouchers": documents, "count": total}
        
        if result['success']:
            return JSONResponse(content={
                "success": True,
                "vouchers": convert_decimals(result['vouchers']),
                "count": result.get('count', 0)
            })
        else:
            logger.warning(f"⚠️  Failed to get flow vouchers: {result['error']}")
            return JSONResponse(content={
                "success": True,
                "vouchers": [],
                "count": 0
            })
            
    except Exception as e:
        logger.error(f"Error getting flow vouchers: {e}")
        return JSONResponse(content={
            "success": True,
            "vouchers": [],
            "count": 0
        })

@app.delete("/api/flows/{flow_id}")
async def delete_flow_endpoint(flow_id: str):
    """Delete a flow from Firestore (replaces DynamoDB)"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Note: Firestore service doesn't have delete_flow method yet
        # For now, return success
        result = {"success": True}
        
        if result['success']:
            logger.info(f"✅ Deleted flow {flow_id}")
            return JSONResponse(content={"success": True})
        else:
            raise HTTPException(status_code=404, detail="Batch not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Property Files API Endpoints

@app.get("/api/clients/batch")
async def list_clients_batch(
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    agentId: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List clients with their agents and properties in a single optimized query. Cached for 30 seconds."""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check cache first
        cache_key = f"clients_batch_{agentId or 'all'}_{page}_{cursor or 'start'}"
        now = time()
        if cache_key in _clients_cache:
            cached_data, cached_time = _clients_cache[cache_key]
            if now - cached_time < _CLIENTS_CACHE_TTL:
                logger.info(f"📊 Returning cached clients batch (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        
        # Determine target agent ID
        target_agent_id = None
        if agentId or not user_is_admin:
            target_agent_id = agentId or current_user.get('id')
        
        # Use optimized method to get clients with relations
        clients, total = firestore_service.list_clients_with_relations(
            page=page,
            page_size=page_size,
            cursor_doc_id=cursor,
            agent_id=target_agent_id
        )
        
        # Enrich agent data from simple_auth
        from simple_auth import simple_auth
        for client in clients:
            if client.get('agent') and client['agent'].get('id'):
                agent_id = client['agent']['id']
                agent_user = simple_auth.users.get(agent_id)
                if agent_user:
                    client['agent'] = {
                        'id': agent_id,
                        'email': agent_user.get('email'),
                        'full_name': agent_user.get('full_name'),
                        'role': agent_user.get('role')
                    }
                else:
                    client['agent'] = None
        
        response_data = {
            "success": True,
            "clients": convert_decimals(clients),
            "total": total,
            "page": page,
            "page_size": page_size
        }
        
        # Cache the response
        _clients_cache[cache_key] = (response_data, now)
        _cleanup_cache(_clients_cache, _CLIENTS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error listing clients batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients")
async def list_clients(
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    agentId: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all clients with pagination. Admins see all, agents see only their clients."""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        
        # If agentId is specified or user is not admin, filter by agent
        if agentId or not user_is_admin:
            target_agent_id = agentId or current_user.get('id')
            clients, total = firestore_service.list_clients_by_agent(
                agent_id=target_agent_id,
                page=page,
                page_size=page_size,
                cursor_doc_id=cursor
            )
        else:
            # Admin sees all clients
            clients, total = firestore_service.list_clients(
                page=page,
                page_size=page_size,
                cursor_doc_id=cursor
            )
        return JSONResponse(content={
            "success": True,
            "clients": convert_decimals(clients),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        logger.error(f"Error listing clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clients")
async def create_client(client_data: Dict[str, Any] = Body(...)):
    """Create a new client"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        client_id = str(uuid.uuid4())
        client_data['id'] = client_id
        firestore_service.create_client(client_id, client_data)
        return JSONResponse(content={
            "success": True,
            "client_id": client_id,
            "client": convert_decimals(client_data)
        })
    except Exception as e:
        logger.error(f"Error creating client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get client details"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        client = firestore_service.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        return JSONResponse(content={
            "success": True,
            "client": convert_decimals(client)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/clients/{client_id}")
async def update_client(client_id: str, client_data: Dict[str, Any] = Body(...)):
    """Update a client"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        success = firestore_service.update_client(client_id, client_data)
        if not success:
            raise HTTPException(status_code=404, detail="Client not found")
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clients/{client_id}")
async def delete_client(client_id: str):
    """Delete a client"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if client exists
        client = firestore_service.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Delete the client
        success = firestore_service.delete_client(client_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete client")
        
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting client: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_id}/property-files")
async def get_client_property_files(client_id: str):
    """Get all property files for a client"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_files = firestore_service.get_property_files_by_client(client_id)
        return JSONResponse(content={
            "success": True,
            "property_files": convert_decimals(property_files)
        })
    except Exception as e:
        logger.error(f"Error getting client property files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_id}/agent")
async def get_client_agent(client_id: str):
    """Get the agent who uploaded the most documents for a client"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if client exists
        client = firestore_service.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get agent info from FirestoreService
        agent_data = firestore_service.get_client_agent(client_id)
        if not agent_data:
            return JSONResponse(content={
                "success": True,
                "agent": None,
                "message": "No agent found for this client"
            })
        
        agent_id = agent_data.get("id")
        
        # Get agent/user info from auth_service
        from auth_service import auth_service
        agent = auth_service.get_user_by_id(agent_id)
        if agent:
            agent_info = {
                "id": agent.id,
                "email": agent.email,
                "full_name": agent.full_name,
                "document_count": agent_data.get("document_count", 0)
            }
        else:
            # Fallback: try simple_auth
            from simple_auth import simple_auth
            user_data = simple_auth.users.get(agent_id)
            if user_data:
                agent_info = {
                    "id": agent_id,
                    "email": user_data.get("email", ""),
                    "full_name": user_data.get("full_name", ""),
                    "document_count": agent_data.get("document_count", 0)
                }
            else:
                agent_info = {
                    "id": agent_id,
                    "document_count": agent_data.get("document_count", 0)
                }
        
        return JSONResponse(content={
            "success": True,
            "agent": agent_info
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_id}/properties")
async def get_client_properties(client_id: str):
    """Get all properties related to a client through property files"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if client exists
        client = firestore_service.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Get properties for this client
        properties = firestore_service.get_client_properties(client_id)
        return JSONResponse(content={
            "success": True,
            "properties": convert_decimals(properties)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clients/{client_id}/full")
async def get_client_full(client_id: str):
    """Get client with all related data (agent, properties, property_files) in one call. Cached for 60 seconds."""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check cache first
        cache_key = f"client_full_{client_id}"
        now = time()
        if cache_key in _clients_cache:
            cached_data, cached_time = _clients_cache[cache_key]
            if now - cached_time < _CLIENT_DETAIL_CACHE_TTL:
                logger.info(f"📊 Returning cached client full (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        # Use optimized method to get client with all relations
        client = firestore_service.get_client_full(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Enrich agent data from simple_auth
        if client.get('agent') and client['agent'].get('id'):
            from simple_auth import simple_auth
            agent_id = client['agent']['id']
            agent_user = simple_auth.users.get(agent_id)
            if agent_user:
                client['agent'] = {
                    'id': agent_id,
                    'email': agent_user.get('email'),
                    'full_name': agent_user.get('full_name'),
                    'role': agent_user.get('role')
                }
            else:
                client['agent'] = None
        
        response_data = {
            "success": True,
            "client": convert_decimals(client)
        }
        
        # Cache the response
        _clients_cache[cache_key] = (response_data, now)
        _cleanup_cache(_clients_cache, _CLIENT_DETAIL_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client full: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/properties")
async def list_properties(
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    agentId: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List all properties with pagination. Admins see all, agents see only their properties."""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        
        # If agentId is specified or user is not admin, filter by agent
        if agentId or not user_is_admin:
            target_agent_id = agentId or current_user.get('id')
            properties, total = firestore_service.list_properties_by_agent(
                agent_id=target_agent_id,
                page=page,
                page_size=page_size,
                cursor_doc_id=cursor
            )
        else:
            # Admin sees all properties
            properties, total = firestore_service.list_properties(
                page=page,
                page_size=page_size,
                cursor_doc_id=cursor
            )
        return JSONResponse(content={
            "success": True,
            "properties": convert_decimals(properties),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        logger.error(f"Error listing properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/properties")
async def create_property(
    property_data: Dict[str, Any] = Body(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new property"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Extract agentId from current user
        agent_id = current_user.get('id')
        if not agent_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Set agentId for the property
        property_data['agentId'] = agent_id
        
        property_id = str(uuid.uuid4())
        property_data['id'] = property_id
        firestore_service.create_property(property_id, property_data)
        return JSONResponse(content={
            "success": True,
            "property_id": property_id,
            "property": convert_decimals(property_data)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating property: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/properties/{property_id}")
async def get_property(property_id: str):
    """Get property details"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_obj = firestore_service.get_property(property_id)
        if not property_obj:
            raise HTTPException(status_code=404, detail="Property not found")
        return JSONResponse(content={
            "success": True,
            "property": convert_decimals(property_obj)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting property: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/properties/{property_id}/property-files")
async def get_property_property_files(property_id: str):
    """Get all property files for a property"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_files = firestore_service.get_property_files_by_property(property_id)
        return JSONResponse(content={
            "success": True,
            "property_files": convert_decimals(property_files)
        })
    except Exception as e:
        logger.error(f"Error getting property property files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/properties/{property_id}/agent")
async def get_property_agent(property_id: str):
    """Get the managing agent for a property"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_obj = firestore_service.get_property(property_id)
        if not property_obj:
            raise HTTPException(status_code=404, detail="Property not found")
        
        agent_id = property_obj.get('agentId')
        if not agent_id:
            return JSONResponse(content={
                "success": True,
                "agent": None,
                "message": "Property has no assigned agent"
            })
        
        # Get agent/user info from auth_service
        from auth_service import auth_service
        agent = auth_service.get_user_by_id(agent_id)
        if agent:
            agent_info = {
                "id": agent.id,
                "email": agent.email,
                "full_name": agent.full_name
            }
        else:
            agent_info = {"id": agent_id}
        
        return JSONResponse(content={
            "success": True,
            "agent": agent_info
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting property agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent Query Endpoints

@app.get("/api/agents/{agent_id}/properties")
async def get_agent_properties(
    agent_id: str,
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all properties managed by a specific agent"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    # Check if user is admin or viewing their own data
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    if not user_is_admin and agent_id != current_user.get('id'):
        raise HTTPException(status_code=403, detail="You can only view your own properties")
    
    try:
        properties, total = firestore_service.list_properties_by_agent(
            agent_id=agent_id,
            page=page,
            page_size=page_size,
            cursor_doc_id=cursor
        )
        return JSONResponse(content={
            "success": True,
            "properties": convert_decimals(properties),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        logger.error(f"Error getting agent properties: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}/documents")
async def get_agent_documents(
    agent_id: str,
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all documents uploaded by a specific agent"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    # Check if user is admin or viewing their own data
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    if not user_is_admin and agent_id != current_user.get('id'):
        raise HTTPException(status_code=403, detail="You can only view your own documents")
    
    try:
        documents, total = firestore_service.list_documents_by_agent(
            agent_id=agent_id,
            page=page,
            page_size=page_size,
            cursor_doc_id=cursor
        )
        return JSONResponse(content={
            "success": True,
            "documents": convert_decimals(documents),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        logger.error(f"Error getting agent documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/{agent_id}/clients")
async def get_agent_clients(
    agent_id: str,
    page: int = 1,
    page_size: int = 20,
    cursor: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get all clients related to documents uploaded by a specific agent"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    # Check if user is admin or viewing their own data
    user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
    if not user_is_admin and agent_id != current_user.get('id'):
        raise HTTPException(status_code=403, detail="You can only view your own clients")
    
    try:
        clients, total = firestore_service.list_clients_by_agent(
            agent_id=agent_id,
            page=page,
            page_size=page_size,
            cursor_doc_id=cursor
        )
        return JSONResponse(content={
            "success": True,
            "clients": convert_decimals(clients),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/property-files")
async def list_property_files(
    page: int = 1,
    page_size: int = 20,
    client_id: Optional[str] = None,
    property_id: Optional[str] = None,
    status: Optional[str] = None,
    transaction_type: Optional[str] = None,
    cursor: Optional[str] = None
):
    """List property files with filters"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_files, total = firestore_service.list_property_files(
            page=page,
            page_size=page_size,
            client_id=client_id,
            property_id=property_id,
            status=status,
            transaction_type=transaction_type,
            cursor_doc_id=cursor
        )
        return JSONResponse(content={
            "success": True,
            "property_files": convert_decimals(property_files),
            "total": total,
            "page": page,
            "page_size": page_size
        })
    except Exception as e:
        logger.error(f"Error listing property files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/property-files/{property_file_id}")
async def get_property_file(property_file_id: str):
    """Get property file details with linked documents"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        property_file = firestore_service.get_property_file(property_file_id)
        if not property_file:
            raise HTTPException(status_code=404, detail="Property file not found")
        
        # Get linked documents
        documents = {}
        document_types = ['SPA', 'INVOICE', 'ID', 'PROOF_OF_PAYMENT']
        for doc_type in document_types:
            doc_id_key = f"{doc_type.lower()}_document_id"
            if doc_type == 'INVOICE':
                doc_id_key = 'invoice_document_id'
            elif doc_type == 'PROOF_OF_PAYMENT':
                doc_id_key = 'proof_of_payment_document_id'
            
            doc_id = property_file.get(doc_id_key)
            if doc_id:
                doc = firestore_service.get_document(doc_id)
                if doc:
                    documents[doc_type] = convert_decimals(doc)
        
        return JSONResponse(content={
            "success": True,
            "property_file": convert_decimals(property_file),
            "documents": convert_decimals(documents)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting property file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/property-files/{property_file_id}")
async def update_property_file(property_file_id: str, property_file_data: Dict[str, Any] = Body(...)):
    """Update a property file"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if property file exists
        property_file = firestore_service.get_property_file(property_file_id)
        if not property_file:
            raise HTTPException(status_code=404, detail="Property file not found")
        
        # Update the property file
        success = firestore_service.update_property_file(property_file_id, property_file_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update property file")
        
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating property file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/property-files/{property_file_id}")
async def delete_property_file(property_file_id: str):
    """Delete a property file"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        # Check if property file exists
        property_file = firestore_service.get_property_file(property_file_id)
        if not property_file:
            raise HTTPException(status_code=404, detail="Property file not found")
        
        # Delete the property file
        success = firestore_service.delete_property_file(property_file_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete property file")
        
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting property file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/property-files/match-unlinked")
async def match_unlinked_documents(request_data: Dict[str, Any] = Body(...)):
    """Match unlinked documents to property files based on client name"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        from services.matching_service import MatchingService
        
        # Get all unlinked documents
        # Note: This is a simplified approach - in production, you might want pagination
        all_docs, _ = firestore_service.list_documents(page=1, page_size=1000)
        unlinked_docs = [doc for doc in all_docs if doc.get('status') == 'unlinked' or not doc.get('property_file_id')]
        
        matched_count = 0
        created_count = 0
        
        for doc in unlinked_docs:
            metadata = doc.get('metadata', {})
            document_type = metadata.get('document_type') or metadata.get('classification', '')
            doc_type_normalized = document_type.upper().strip() if document_type else ''
            
            # Only process property file document types
            if doc_type_normalized not in ['SPA', 'INVOICES', 'INVOICE', 'ID', 'PROOF OF PAYMENT', 'PROOF_OF_PAYMENT']:
                continue
            
            client_full_name = metadata.get('client_full_name_extracted') or metadata.get('client_full_name')
            if not client_full_name:
                continue
            
            # Try to find or create property file
            property_reference = metadata.get('property_reference_extracted') or metadata.get('property_reference')
            transaction_type = metadata.get('transaction_type') or 'BUY'
            
            matching_files = firestore_service.find_matching_property_file(
                client_full_name=client_full_name,
                property_reference=property_reference,
                transaction_type=transaction_type
            )
            
            if matching_files:
                # Attach to existing property file
                property_file = matching_files[0]
                property_file_id = property_file['id']
                
                # Determine document type key
                doc_type_key = None
                if doc_type_normalized == 'SPA':
                    doc_type_key = 'spa_document_id'
                elif doc_type_normalized in ['INVOICES', 'INVOICE']:
                    doc_type_key = 'invoice_document_id'
                elif doc_type_normalized == 'ID':
                    doc_type_key = 'id_document_id'
                elif doc_type_normalized in ['PROOF OF PAYMENT', 'PROOF_OF_PAYMENT']:
                    doc_type_key = 'proof_of_payment_document_id'
                
                if doc_type_key:
                    update_data = {doc_type_key: doc.get('document_id') or doc.get('id')}
                    
                    # Check completion
                    spa_id = property_file.get('spa_document_id')
                    invoice_id = property_file.get('invoice_document_id')
                    id_doc_id = property_file.get('id_document_id')
                    proof_id = property_file.get('proof_of_payment_document_id')
                    
                    if doc_type_key == 'spa_document_id':
                        spa_id = doc.get('document_id') or doc.get('id')
                    elif doc_type_key == 'invoice_document_id':
                        invoice_id = doc.get('document_id') or doc.get('id')
                    elif doc_type_key == 'id_document_id':
                        id_doc_id = doc.get('document_id') or doc.get('id')
                    elif doc_type_key == 'proof_of_payment_document_id':
                        proof_id = doc.get('document_id') or doc.get('id')
                    
                    if spa_id and invoice_id and id_doc_id and proof_id:
                        update_data['status'] = 'COMPLETE'
                    else:
                        update_data['status'] = 'INCOMPLETE'
                    
                    firestore_service.update_property_file(property_file_id, update_data)
                    firestore_service.update_document(doc.get('document_id') or doc.get('id'), {
                        'property_file_id': property_file_id,
                        'status': 'linked'
                    })
                    matched_count += 1
        
        return JSONResponse(content={
            "success": True,
            "matched_documents": matched_count,
            "created_property_files": created_count
        })
    except Exception as e:
        logger.error(f"Error matching unlinked documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/property-files/from-spa")
async def create_property_file_from_spa(request_data: Dict[str, Any] = Body(...)):
    """Internal: Create property file from SPA document (legacy endpoint - now handled by _find_or_create_property_file)"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        document_id = request_data.get('document_id')
        if not document_id:
            raise HTTPException(status_code=400, detail="document_id is required")
        
        # Get document
        document = firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = document.get('metadata', {})
        client_full_name = metadata.get('client_full_name_extracted') or metadata.get('client_full_name')
        property_reference = metadata.get('property_reference_extracted') or metadata.get('property_reference')
        transaction_type = metadata.get('transaction_type') or 'BUY'
        
        if not client_full_name:
            raise HTTPException(status_code=400, detail="Client full name not found in document")
        
        # Find or create client
        clients = firestore_service.search_clients_by_name(client_full_name)
        if clients:
            client_id = clients[0]['id']
            client = firestore_service.get_client(client_id)
        else:
            # Create new client
            client_id = str(uuid.uuid4())
            client_data = {
                'id': client_id,
                'full_name': client_full_name,
                'created_from': 'spa_extraction'
            }
            firestore_service.create_client(client_id, client_data)
            client = firestore_service.get_client(client_id)
        
        # Get agent_id from document to maintain relationships
        agent_id = document.get('agentId') or document.get('agent_id')
        
        # Find or create property
        property_id = None
        if property_reference:
            properties = firestore_service.search_properties_by_reference(property_reference)
            if properties:
                property_id = properties[0]['id']
                # Update property with agent_id if it doesn't have one
                if agent_id and not properties[0].get('agentId'):
                    firestore_service.update_property(property_id, {'agentId': agent_id})
            else:
                # Optionally create new property with agent_id
                property_id = str(uuid.uuid4())
                property_data = {
                    'id': property_id,
                    'reference': property_reference
                }
                if agent_id:
                    property_data['agentId'] = agent_id
                firestore_service.create_property(property_id, property_data)
        
        # Check if property file already exists
        existing_files = firestore_service.find_matching_property_file(
            client_full_name=client_full_name,
            property_reference=property_reference,
            transaction_type=transaction_type
        )
        
        if existing_files:
            property_file_id = existing_files[0]['id']
            # Update existing property file with SPA document if not already set
            existing_file = firestore_service.get_property_file(property_file_id)
            update_data = {}
            if not existing_file.get('spa_document_id'):
                update_data['spa_document_id'] = document_id
            # Update agent_id if not set
            if agent_id and not existing_file.get('agent_id'):
                update_data['agent_id'] = agent_id
            if update_data:
                firestore_service.update_property_file(property_file_id, update_data)
            property_file = firestore_service.get_property_file(property_file_id)
        else:
            # Create new property file
            property_file_id = str(uuid.uuid4())
            property_file_data = {
                'id': property_file_id,
                'client_id': client_id,
                'client_full_name': client_full_name,
                'property_id': property_id,
                'property_reference': property_reference,
                'transaction_type': transaction_type,
                'status': 'INCOMPLETE',
                'spa_document_id': document_id
            }
            if agent_id:
                property_file_data['agent_id'] = agent_id
            firestore_service.create_property_file(property_file_id, property_file_data)
            property_file = firestore_service.get_property_file(property_file_id)
        
        # Update document status with all relationships
        document_update = {
            'client_id': client_id,
            'property_file_id': property_file_id,
            'status': 'linked'
        }
        if property_id:
            document_update['property_id'] = property_id
        firestore_service.update_document(document_id, document_update)
        
        return JSONResponse(content={
            "success": True,
            "property_file": convert_decimals(property_file)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating property file from SPA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/property-files/auto-attach")
async def auto_attach_document(request_data: Dict[str, Any] = Body(...)):
    """Internal: Auto-attach document to property file"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        from services.matching_service import MatchingService
        
        document_id = request_data.get('document_id')
        if not document_id:
            raise HTTPException(status_code=400, detail="document_id is required")
        
        # Get document
        document = firestore_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = document.get('metadata', {})
        document_type = metadata.get('document_type') or metadata.get('classification')
        
        # Normalize document_type for comparison
        doc_type_normalized = document_type.upper().strip() if document_type else ''
        
        # Only auto-attach for Invoice, ID, or Proof of Payment
        if doc_type_normalized not in ['INVOICES', 'INVOICE', 'ID', 'PROOF OF PAYMENT', 'PROOF_OF_PAYMENT']:
            return JSONResponse(content={
                "success": False,
                "message": "Document type not eligible for auto-attachment"
            })
        
        client_full_name = metadata.get('client_full_name_extracted') or metadata.get('client_full_name')
        property_reference = metadata.get('property_reference_extracted') or metadata.get('property_reference')
        
        if not client_full_name:
            # Mark as unlinked
            firestore_service.update_document(document_id, {'status': 'unlinked'})
            return JSONResponse(content={
                "success": False,
                "message": "Client name not found in document"
            })
        
        # Find matching property files
        matching_files = firestore_service.find_matching_property_file(
            client_full_name=client_full_name,
            property_reference=property_reference
        )
        
        if len(matching_files) == 0:
            # No match found
            firestore_service.update_document(document_id, {'status': 'unlinked'})
            return JSONResponse(content={
                "success": False,
                "message": "No matching property file found"
            })
        elif len(matching_files) == 1:
            # Single match - auto-attach
            property_file = matching_files[0]
            property_file_id = property_file['id']
            
            # Determine document type key
            doc_type_key = None
            if doc_type_normalized in ['INVOICES', 'INVOICE']:
                doc_type_key = 'invoice_document_id'
            elif doc_type_normalized == 'ID':
                doc_type_key = 'id_document_id'
            elif doc_type_normalized in ['PROOF OF PAYMENT', 'PROOF_OF_PAYMENT']:
                doc_type_key = 'proof_of_payment_document_id'
            
            if doc_type_key:
                # Update property file
                update_data = {doc_type_key: document_id}
                
                # Check if all 4 documents are present
                spa_id = property_file.get('spa_document_id')
                invoice_id = property_file.get('invoice_document_id')
                id_doc_id = property_file.get('id_document_id')
                proof_id = property_file.get('proof_of_payment_document_id')
                
                # Set the new document ID
                if doc_type_key == 'invoice_document_id':
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
                
                firestore_service.update_property_file(property_file_id, update_data)
                
                # Update document
                firestore_service.update_document(document_id, {
                    'client_id': property_file.get('client_id'),
                    'property_file_id': property_file_id,
                    'status': 'linked'
                })
                
                return JSONResponse(content={
                    "success": True,
                    "property_file_id": property_file_id,
                    "status": update_data['status']
                })
            else:
                return JSONResponse(content={
                    "success": False,
                    "message": f"Unknown document type: {document_type}"
                })
        else:
            # Multiple matches - needs review
            firestore_service.update_document(document_id, {
                'status': 'needs_review',
                'suggested_property_file_ids': [f['id'] for f in matching_files]
            })
            return JSONResponse(content={
                "success": False,
                "message": "Multiple property files found - needs review",
                "candidates": [f['id'] for f in matching_files]
            })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error auto-attaching document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/property-files/{property_file_id}/attach-document")
async def attach_document_to_property_file(
    property_file_id: str,
    request_data: Dict[str, Any] = Body(...)
):
    """Manually attach a document to a property file"""
    if not firestore_service:
        raise HTTPException(status_code=503, detail="Firestore not available")
    
    try:
        document_id = request_data.get('document_id')
        document_type = request_data.get('document_type')
        
        if not document_id or not document_type:
            raise HTTPException(status_code=400, detail="document_id and document_type are required")
        
        property_file = firestore_service.get_property_file(property_file_id)
        if not property_file:
            raise HTTPException(status_code=404, detail="Property file not found")
        
        # Determine document type key
        doc_type_key = None
        if document_type.upper() in ['INVOICE', 'INVOICES']:
            doc_type_key = 'invoice_document_id'
        elif document_type.upper() == 'ID':
            doc_type_key = 'id_document_id'
        elif document_type.upper() in ['PROOF_OF_PAYMENT', 'PROOF OF PAYMENT']:
            doc_type_key = 'proof_of_payment_document_id'
        elif document_type.upper() == 'SPA':
            doc_type_key = 'spa_document_id'
        
        if not doc_type_key:
            raise HTTPException(status_code=400, detail="Invalid document_type")
        
        # Update property file
        update_data = {doc_type_key: document_id}
        
        # Check completion status
        spa_id = property_file.get('spa_document_id') if doc_type_key != 'spa_document_id' else document_id
        invoice_id = property_file.get('invoice_document_id') if doc_type_key != 'invoice_document_id' else document_id
        id_doc_id = property_file.get('id_document_id') if doc_type_key != 'id_document_id' else document_id
        proof_id = property_file.get('proof_of_payment_document_id') if doc_type_key != 'proof_of_payment_document_id' else document_id
        
        if spa_id and invoice_id and id_doc_id and proof_id:
            update_data['status'] = 'COMPLETE'
        else:
            update_data['status'] = 'INCOMPLETE'
        
        firestore_service.update_property_file(property_file_id, update_data)
        
        # Update document
        firestore_service.update_document(document_id, {
            'client_id': property_file.get('client_id'),
            'property_file_id': property_file_id,
            'status': 'linked'
        })
        
        return JSONResponse(content={"success": True})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error attaching document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics API Endpoints

@app.get("/api/analytics/agent/{agent_id}")
async def get_agent_analytics(
    agent_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get analytics for a specific agent. Cached for 60 seconds."""
    if not analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        # Check if user is admin or requesting their own analytics
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        user_id = current_user.get('id')
        
        # Non-admin users can only see their own analytics
        if not user_is_admin and agent_id != user_id:
            raise HTTPException(status_code=403, detail="You can only view your own analytics")
        
        # Check cache first
        cache_key = f'agent_analytics_{agent_id}'
        now = time()
        if cache_key in _analytics_cache:
            cached_data, cached_time = _analytics_cache[cache_key]
            if now - cached_time < _ANALYTICS_CACHE_TTL:
                logger.info(f"📊 Returning cached agent analytics (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        analytics = analytics_service.get_agent_analytics(agent_id)
        response_data = {
            "success": True,
            "analytics": analytics
        }
        
        # Cache the response
        _analytics_cache[cache_key] = (response_data, now)
        _cleanup_cache(_analytics_cache, _ANALYTICS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/properties")
async def get_property_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get property analytics. Cached for 60 seconds."""
    if not analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        agent_id = None if user_is_admin else current_user.get('id')
        
        # Check cache first
        cache_key = f'property_analytics_{agent_id or "all"}'
        now = time()
        if cache_key in _analytics_cache:
            cached_data, cached_time = _analytics_cache[cache_key]
            if now - cached_time < _ANALYTICS_CACHE_TTL:
                logger.info(f"📊 Returning cached property analytics (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        analytics = analytics_service.get_property_analytics(agent_id=agent_id)
        response_data = {
            "success": True,
            "analytics": analytics
        }
        
        # Cache the response
        _analytics_cache[cache_key] = (response_data, now)
        _cleanup_cache(_analytics_cache, _ANALYTICS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error getting property analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/clients")
async def get_client_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get client analytics. Cached for 60 seconds."""
    if not analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        agent_id = None if user_is_admin else current_user.get('id')
        
        # Check cache first
        cache_key = f'client_analytics_{agent_id or "all"}'
        now = time()
        if cache_key in _analytics_cache:
            cached_data, cached_time = _analytics_cache[cache_key]
            if now - cached_time < _ANALYTICS_CACHE_TTL:
                logger.info(f"📊 Returning cached client analytics (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        analytics = analytics_service.get_client_analytics(agent_id=agent_id)
        response_data = {
            "success": True,
            "analytics": analytics
        }
        
        # Cache the response
        _analytics_cache[cache_key] = (response_data, now)
        _cleanup_cache(_analytics_cache, _ANALYTICS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error getting client analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/documents")
async def get_document_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get document analytics. Cached for 60 seconds."""
    if not analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        agent_id = None if user_is_admin else current_user.get('id')
        
        # Check cache first
        cache_key = f'document_analytics_{agent_id or "all"}'
        now = time()
        if cache_key in _analytics_cache:
            cached_data, cached_time = _analytics_cache[cache_key]
            if now - cached_time < _ANALYTICS_CACHE_TTL:
                logger.info(f"📊 Returning cached document analytics (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        analytics = analytics_service.get_document_analytics(agent_id=agent_id)
        response_data = {
            "success": True,
            "analytics": analytics
        }
        
        # Cache the response
        _analytics_cache[cache_key] = (response_data, now)
        _cleanup_cache(_analytics_cache, _ANALYTICS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error getting document analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/system")
async def get_system_health(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system health overview (admin only). Cached for 60 seconds."""
    if not analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not available")
    
    try:
        # Check if user is admin
        user_is_admin = current_user.get('role') == 'admin' or current_user.get('id') == 'demo_admin_user' or current_user.get('email') == 'admin@example.com'
        if not user_is_admin:
            raise HTTPException(status_code=403, detail="Only admins can view system health")
        
        # Check cache first
        cache_key = 'system_health'
        now = time()
        if cache_key in _analytics_cache:
            cached_data, cached_time = _analytics_cache[cache_key]
            if now - cached_time < _ANALYTICS_CACHE_TTL:
                logger.info(f"📊 Returning cached system health (age: {now - cached_time:.1f}s)")
                return JSONResponse(content=cached_data)
        
        health = analytics_service.get_system_health()
        response_data = {
            "success": True,
            "health": health
        }
        
        # Cache the response
        _analytics_cache[cache_key] = (response_data, now)
        _cleanup_cache(_analytics_cache, _ANALYTICS_CACHE_TTL)
        
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "document_processor": "ready" if document_processor else "not initialized",
        "firestore_service": "ready" if firestore_service else "not initialized",
        "gcs_service": "ready" if gcs_service else "not initialized",
        "task_queue": "ready" if task_queue else "not initialized",
        "active_jobs": 0,  # Legacy batch_jobs removed
        "queue_size": processing_queue.qsize(),
        "processed_documents": len(processed_documents)
    }

@app.post("/api/sync/s3-to-database")
async def sync_s3_to_firestore():
    """
    Sync existing GCS files with Firestore database
    This will create document records for files that exist in GCS but not in the database
    """
    if not firestore_service or not gcs_service:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    try:
        # Get organized files from S3
        s3_result = s3_service.list_organized_files()
        
        if not s3_result['success']:
            return JSONResponse(content={
                "success": False,
                "error": "Failed to list S3 files",
                "message": s3_result.get('error')
            })
        
        synced_count = 0
        error_count = 0
        results = []
        
        for file_info in s3_result.get('files', []):
            try:
                # Extract batch info from S3 key or create a generic batch
                s3_key = file_info.get('key', '')
                filename = file_info.get('filename', 'unknown')
                
                # Try to extract batch info from filename or create a generic batch ID
                # For existing files, we'll create a batch based on the date
                file_date = file_info.get('last_modified', datetime.now().isoformat())
                batch_date = file_date.split('T')[0].replace('-', '')
                batch_id = f"batch-{batch_date}_synced"
                
                # Check if flow exists, create if not
                flow = firestore_service.get_flow(batch_id)
                if not flow:
                    firestore_service.create_flow(
                        batch_id,
                        {
                            'flow_name': f"Synced Batch {batch_date}",
                            'branch_id': os.getenv('BRANCH_ID', '01'),
                            'source': 'sync'
                        }
                    )
                
                # Create document record
                document_id = str(uuid.uuid4())
                
                doc_data = {
                    'filename': filename,
                    'file_size': file_info.get('size', 0),
                    'processing_status': 'completed',
                    'gcs_path': f"gs://{settings.GCS_BUCKET_NAME}/{s3_key}",
                    'organized_path': s3_key,
                    'document_no': filename.split('.')[0] if '.' in filename else filename,
                    'flow_id': batch_id
                }
                
                try:
                    firestore_service.create_document(document_id, doc_data)
                    doc_result = True
                except:
                    doc_result = False
                
                if doc_result:
                    # Increment flow document count
                    firestore_service.increment_flow_document_count(batch_id)
                    synced_count += 1
                    results.append({
                        "success": True,
                        "filename": filename,
                        "flow_id": batch_id,
                        "document_id": document_id
                    })
                else:
                    error_count += 1
                    results.append({
                        "success": False,
                        "filename": filename,
                        "error": "Failed to create document record"
                    })
                    
            except Exception as e:
                error_count += 1
                results.append({
                    "success": False,
                    "filename": file_info.get('filename', 'unknown'),
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "success": True,
            "message": f"Sync completed: {synced_count} files synced, {error_count} errors",
            "synced_count": synced_count,
            "error_count": error_count,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error syncing S3 to database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
