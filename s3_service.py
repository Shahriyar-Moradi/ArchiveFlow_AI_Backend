"""
GCS-backed drop-in replacement for the previous S3 service.
Implements the key methods the backend calls today, using Google Cloud Storage.
Some rarely-used methods return a not-implemented response so callers can handle gracefully.
"""

import json
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import time

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# Default prefixes match the old S3 layout to minimize downstream changes.
TEMP_PREFIX = "temp"
FLOW_META_PREFIX = "flows"
ORG_PREFIX = "organized_vouchers"


def _check_key_file_exists() -> Tuple[bool, Optional[str]]:
    """
    Check if service account key file exists at expected locations.
    
    Returns:
        Tuple of (exists: bool, path: Optional[str])
    """
    # Check GOOGLE_APPLICATION_CREDENTIALS environment variable
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and os.path.exists(key_path):
        return True, key_path
    
    # Check local fallback location
    fallback_key = os.path.join(os.path.dirname(__file__), "voucher-storage-key.json")
    if os.path.exists(fallback_key):
        return True, fallback_key
    
    return False, None


def _credentials_support_signing(credentials) -> bool:
    """
    Check if credentials have a private key for signing URLs.
    
    Args:
        credentials: Google auth credentials object
        
    Returns:
        True if credentials support signing (have private key), False otherwise
    """
    if credentials is None:
        return False
    
    # Service account credentials have a private_key attribute
    if hasattr(credentials, 'private_key'):
        private_key = credentials.private_key
        # Check if private_key exists and is not empty
        if private_key and isinstance(private_key, str) and len(private_key.strip()) > 0:
            return True
    
    # Check if it's a service account credentials object
    if hasattr(credentials, 'service_account_email'):
        # Service account credentials loaded from JSON should have private_key
        if hasattr(credentials, 'private_key'):
            private_key = credentials.private_key
            if private_key and isinstance(private_key, str) and len(private_key.strip()) > 0:
                return True
    
    # Check the class name to see if it's a service account credentials type
    class_name = credentials.__class__.__name__ if credentials else ""
    if 'ServiceAccount' in class_name:
        # If it's a service account credentials class, it should have private_key
        if hasattr(credentials, 'private_key'):
            private_key = credentials.private_key
            if private_key and isinstance(private_key, str) and len(private_key.strip()) > 0:
                return True
    
    return False


def _get_gcs_client():
    """
    Build a GCS client with service account credentials for signed URL generation.
    Follows production best practices from GCP examples:
    1. Try GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Fallback to local service account key file
    3. Last resort: application-default credentials (works on Cloud Run with IAM)
    
    Returns:
        Tuple of (storage.Client instance, supports_signing: bool, key_file_path: Optional[str])
    """
    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCS_PROJECT_ID") or "rocasoft"
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    key_file_used = None
    supports_signing = False

    try:
        # Priority 1: Use GOOGLE_APPLICATION_CREDENTIALS if set (production pattern)
        if key_path and os.path.exists(key_path):
            # Verify JSON file contains private_key before loading
            try:
                import json
                with open(key_path, 'r') as f:
                    key_data = json.load(f)
                    if not key_data.get('private_key'):
                        logger.warning(f"⚠️  Key file {key_path} does not contain private_key field")
                        supports_signing = False
                    else:
                        supports_signing = True
            except Exception as e:
                logger.warning(f"⚠️  Could not verify key file structure: {e}, assuming valid")
                supports_signing = True  # Trust the file if we can't verify
            
            # Use from_service_account_json for cleaner initialization
            client = storage.Client.from_service_account_json(
                key_path,
                project=project_id
            )
            key_file_used = key_path
            # If we successfully loaded from JSON and verified it has private_key, it supports signing
            logger.info(f"✅ GCS client initialized with service account key: {key_path} (signing: {supports_signing})")
            return client, supports_signing, key_file_used

        # Priority 2: Try local service account key file
        fallback_key = os.path.join(os.path.dirname(__file__), "voucher-storage-key.json")
        if os.path.exists(fallback_key):
            # Verify JSON file contains private_key before loading
            try:
                import json
                with open(fallback_key, 'r') as f:
                    key_data = json.load(f)
                    if not key_data.get('private_key'):
                        logger.warning(f"⚠️  Key file {fallback_key} does not contain private_key field")
                        supports_signing = False
                    else:
                        supports_signing = True
            except Exception as e:
                logger.warning(f"⚠️  Could not verify key file structure: {e}, assuming valid")
                supports_signing = True  # Trust the file if we can't verify
            
            # Use from_service_account_json for cleaner initialization
            client = storage.Client.from_service_account_json(
                fallback_key,
                project=project_id
            )
            key_file_used = fallback_key
            # If we successfully loaded from JSON and verified it has private_key, it supports signing
            logger.info(f"✅ GCS client initialized with service account key: {fallback_key} (signing: {supports_signing})")
            return client, supports_signing, key_file_used

        # Priority 3: Application-default credentials (works on Cloud Run with IAM)
        # WARNING: These credentials typically don't have a private key for signing
        logger.warning(
            "⚠️  No service account key file found. Falling back to application-default credentials. "
            "Signed URL generation may fail. Expected locations:\n"
            f"  - GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
            f"  - {fallback_key}"
        )
        client = storage.Client(project=project_id)
        # Check if credentials support signing
        if hasattr(client, '_credentials'):
            supports_signing = _credentials_support_signing(client._credentials)
        if not supports_signing:
            logger.warning(
                "⚠️  Application-default credentials do not support signed URL generation. "
                "Please configure service account credentials with a private key."
            )
        return client, supports_signing, key_file_used
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize GCS client: {e}")
        # Return None to allow graceful degradation
        return None, False, None


class S3Service:
    """
    GCS implementation that mirrors the public interface of the old S3Service.
    """

    def __init__(self):
        self.bucket_name = (
            os.getenv("GCS_BUCKET_NAME") or os.getenv("GCS_BUCKET") or "voucher-bucket-1"
        )
        client_result = _get_gcs_client()
        if isinstance(client_result, tuple) and len(client_result) == 3:
            self.client, self._supports_signing, self._key_file_path = client_result
        else:
            # Backward compatibility if function returns just client
            self.client = client_result
            self._supports_signing = False
            self._key_file_path = None
        
        self.bucket = self.client.bucket(self.bucket_name) if self.client else None
        
        # Check if key file exists but wasn't used
        if not self._supports_signing:
            key_exists, key_path = _check_key_file_exists()
            if not key_exists:
                logger.warning(
                    "⚠️  Service account key file not found. Signed URL generation will fail.\n"
                    "To fix this:\n"
                    "1. Download the service account key from GCP Console:\n"
                    "   - Go to IAM & Admin > Service Accounts\n"
                    "   - Find: voucher-storage-sa@rocasoft.iam.gserviceaccount.com\n"
                    "   - Click on the service account > Keys tab\n"
                    "   - Click 'Add Key' > 'Create new key' > JSON\n"
                    "   - Save as: voucher-storage-key.json\n"
                    "2. Place the file in the backend directory:\n"
                    f"   {os.path.join(os.path.dirname(__file__), 'voucher-storage-key.json')}\n"
                    "   OR set GOOGLE_APPLICATION_CREDENTIALS environment variable to the file path"
                )
        
        # Simple in-memory cache for flow metadata (key: flow_id, value: (data, timestamp))
        self._flow_metadata_cache: Dict[str, tuple] = {}
        self._cache_ttl = 300  # 5 minutes TTL for metadata cache

    # ------------------------------------------------------------------ helpers
    def _blob(self, key: str):
        return self.bucket.blob(key)

    def _now(self) -> datetime:
        return datetime.utcnow()

    # ---------------------------------------------------------------- uploads
    async def upload_image_to_temp(
        self, image_data: bytes, filename: str, file_type: str, flow_id: str = None
    ) -> Dict[str, Any]:
        """
        Upload a file to GCS temp/{flow_id}/..., keeping the original contract fields.
        Now async to prevent blocking the event loop.
        """
        if not self.client or not self.bucket:
            return {"success": False, "error": "GCS client not initialized"}
        try:
            flow_id = flow_id or f"flow-{self._now().strftime('%Y%m%d_%H%M%S')}"
            clean_filename = filename.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            blob_name = f"{TEMP_PREFIX}/{flow_id}/{clean_filename}"

            blob = self._blob(blob_name)
            content_type = (
                f"image/{file_type}"
                if file_type.lower() in ["jpg", "jpeg", "png"]
                else "application/pdf"
            )
            # Run blocking GCS operation in thread pool to avoid blocking event loop
            await asyncio.to_thread(blob.upload_from_string, image_data, content_type=content_type)
            
            # Verify the file was actually saved to GCS (also async)
            blob_exists = await asyncio.to_thread(blob.exists)
            if not blob_exists:
                return {"success": False, "error": f"File upload failed: blob does not exist after upload", "filename": filename}

            # Get size (this is a property access, should be fast, but make it async-safe)
            size = await asyncio.to_thread(lambda: blob.size) if blob.size is None else blob.size
            if size is None:
                size = len(image_data)

            # Ensure flow metadata exists and increment file counter
            meta = self.get_flow_metadata_from_s3(flow_id)
            if not meta.get("success"):
                self.create_flow_metadata_in_s3(flow_id, f"Flow {flow_id}")
                meta = self.get_flow_metadata_from_s3(flow_id)
            if meta.get("success"):
                current = meta.get("flow", {})
                total_files = current.get("total_files", 0) + 1
                self.update_flow_metadata_in_s3(flow_id, {"total_files": total_files})

            return {
                "success": True,
                "message": "File uploaded successfully to GCS",
                "s3_key": blob_name,  # keep field name for compatibility
                "s3_url": self._public_url(blob),
                "bucket": self.bucket_name,
                "filename": filename,
                "flow_id": flow_id,
                "size": size,
            }
        except Exception as exc:
            return {"success": False, "error": f"GCS upload failed: {exc}", "filename": filename}

    async def check_file_exists(self, s3_key: str) -> bool:
        """Check if file exists in GCS (async to prevent blocking)."""
        if not self.bucket:
            return False
        blob = self._blob(s3_key)
        return await asyncio.to_thread(blob.exists)

    def list_temp_files(self, max_results: int = 1000) -> Dict[str, Any]:
        if not self.client or not self.bucket:
            return {"success": False, "error": "GCS client not initialized", "files": [], "count": 0}
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=f"{TEMP_PREFIX}/", max_results=max_results)
            files = [
                {
                    "key": b.name,
                    "size": b.size,
                    "last_modified": b.updated.isoformat() if b.updated else None,
                    "url": self._public_url(b),
                }
                for b in blobs
                if not b.name.endswith("/")
            ]
            return {"success": True, "files": files, "count": len(files)}
        except Exception as exc:
            return {"success": False, "error": f"Failed to list files: {exc}", "files": [], "count": 0}

    # ---------------------------------------------------------------- organized listing (minimal)
    def list_organized_files(self, max_results: int = 1000) -> Dict[str, Any]:
        """
        Lightweight listing of organized_vouchers. Returns the same shape keys used by callers.
        """
        if not self.client or not self.bucket:
            return {"success": False, "error": "GCS client not initialized", "files": [], "count": 0}
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=f"{ORG_PREFIX}/", max_results=max_results)
            files = []
            for b in blobs:
                if b.name.endswith("/"):
                    continue
                parts = b.name.split("/")
                classification = "UNKNOWN"
                year = branch = month = date = "unknown"
                filename = parts[-1] if parts else "unknown"
                if len(parts) >= 7 and parts[0] == ORG_PREFIX:
                    branch = parts[1]
                    year = parts[2]
                    month = parts[3]
                    date = parts[4]
                    classification = parts[5]
                    filename = parts[6]

                files.append(
                    {
                        "key": b.name,
                        "filename": filename,
                        "classification": classification,
                        "year": year,
                        "branch": branch,
                        "month": month,
                        "date": date,
                        "size": b.size,
                        "last_modified": b.updated.isoformat() if b.updated else None,
                        "url": self._public_url(b),
                        "hierarchical_path": "/".join(parts[1:-1]) if len(parts) > 2 else "",
                    }
                )

            return {
                "success": True,
                "files": files,
                "count": len(files),
                "organized_by_classification": {},
                "organized_by_year": {},
                "organized_by_branch": {},
                "organized_by_month": {},
                "structure_type": "hierarchical",
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to list organized files: {exc}", "files": [], "count": 0}

    def get_organized_folders(self) -> Dict[str, Any]:
        # Minimal placeholder: return a single root folder with count
        files = self.list_organized_files()
        if not files.get("success"):
            return files
        return {
            "success": True,
            "folders": [
                {
                    "name": ORG_PREFIX,
                    "display_name": ORG_PREFIX,
                    "type": "folder",
                    "path": ORG_PREFIX,
                    "document_count": files.get("count", 0),
                    "last_modified": None,
                    "children": [],
                }
            ],
        }

    def get_organized_folder_tree(self) -> Dict[str, Any]:
        return self.list_organized_files()

    def get_folder_documents(self, classification: str) -> Dict[str, Any]:
        files = self.list_organized_files()
        if not files.get("success"):
            return files
        docs = [f for f in files.get("files", []) if f.get("classification") == classification]
        return {"success": True, "documents": docs, "count": len(docs)}

    # ---------------------------------------------------------------- move/replace stubs
    def _update_filename_with_new_type(self, original_filename: str, new_type: str) -> str:
        base, ext = os.path.splitext(original_filename)
        return f"{base}_{new_type}{ext}"

    def move_file_to_classification(self, current_key: str, new_classification: str) -> Dict[str, Any]:
        return {"success": False, "error": "Not implemented for GCS yet"}

    def move_file_update_details(self, *args, **kwargs) -> Dict[str, Any]:
        return {"success": False, "error": "Not implemented for GCS yet"}

    def replace_file(self, current_key: str, new_file_data: bytes, new_filename: str) -> Dict[str, Any]:
        try:
            dest_key = "/".join(current_key.split("/")[:-1] + [new_filename])
            blob = self._blob(dest_key)
            blob.upload_from_string(new_file_data)
            return {"success": True, "key": dest_key}
        except Exception as exc:
            return {"success": False, "error": f"Replace failed: {exc}"}

    # ---------------------------------------------------------------- flow metadata
    def _flow_meta_key(self, flow_id: str) -> str:
        return f"{FLOW_META_PREFIX}/{flow_id}/metadata.json"

    def create_flow_metadata_in_s3(
        self, flow_id: str, flow_name: str, created_at: str = None
    ) -> Dict[str, Any]:
        try:
            meta = {
                "flow_id": flow_id,
                "flow_name": flow_name,
                "created_at": created_at or self._now().isoformat(),
                "total_files": 0,
            }
            self._blob(self._flow_meta_key(flow_id)).upload_from_string(
                json.dumps(meta, indent=2), content_type="application/json"
            )
            
            # Update cache
            self._flow_metadata_cache[flow_id] = (meta, time.time())
            
            return {"success": True, "flow": meta}
        except Exception as exc:
            return {"success": False, "error": f"Failed to create flow metadata: {exc}"}

    def get_flow_metadata_from_s3(self, flow_id: str) -> Dict[str, Any]:
        try:
            # Check cache first
            if flow_id in self._flow_metadata_cache:
                cached_data, timestamp = self._flow_metadata_cache[flow_id]
                if time.time() - timestamp < self._cache_ttl:
                    return {"success": True, "flow": cached_data}
                else:
                    # Cache expired, remove it
                    del self._flow_metadata_cache[flow_id]
            
            # Fetch from GCS
            blob = self._blob(self._flow_meta_key(flow_id))
            if not blob.exists():
                return {"success": False, "error": "Flow metadata not found"}
            data = json.loads(blob.download_as_text())
            
            # Update cache
            self._flow_metadata_cache[flow_id] = (data, time.time())
            
            return {"success": True, "flow": data}
        except Exception as exc:
            return {"success": False, "error": f"Failed to read flow metadata: {exc}"}

    def update_flow_metadata_in_s3(self, flow_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_flow_metadata_from_s3(flow_id)
        if not current.get("success"):
            return current
        try:
            data = current["flow"]
            data.update(updates)
            self._blob(self._flow_meta_key(flow_id)).upload_from_string(
                json.dumps(data, indent=2), content_type="application/json"
            )
            
            # Update cache
            self._flow_metadata_cache[flow_id] = (data, time.time())
            
            return {"success": True, "flow": data}
        except Exception as exc:
            return {"success": False, "error": f"Failed to update flow metadata: {exc}"}

    def list_flows_from_s3(self, max_results: int = 500) -> Dict[str, Any]:
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=f"{FLOW_META_PREFIX}/", max_results=max_results)
            flows = []
            for b in blobs:
                if not b.name.endswith("metadata.json"):
                    continue
                data = json.loads(b.download_as_text())
                flows.append(data)
            return {"success": True, "flows": flows, "count": len(flows)}
        except Exception as exc:
            return {"success": False, "error": f"Failed to list flows: {exc}", "flows": [], "count": 0}

    def _count_flow_temp_files(self, flow_id: str) -> int:
        return len(self._get_flow_temp_files(flow_id))

    def get_flow_files_from_s3(self, flow_id: str, max_results_per_type: int = 1000) -> Dict[str, Any]:
        try:
            temp_files = self._get_flow_temp_files(flow_id, max_results=max_results_per_type)
            failed_files = self._get_flow_failed_files(flow_id, max_results=max_results_per_type)
            organized_files = self._get_flow_organized_files(flow_id, max_results=max_results_per_type)
            return {
                "success": True,
                "temp_files": temp_files,
                "failed_files": failed_files,
                "organized_files": organized_files,
                "total_files": len(temp_files) + len(failed_files) + len(organized_files),
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to list flow files: {exc}"}

    def _get_flow_temp_files(self, flow_id: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        blobs = self.client.list_blobs(self.bucket_name, prefix=f"{TEMP_PREFIX}/{flow_id}/", max_results=max_results)
        return [
            {
                "key": b.name,
                "size": b.size,
                "last_modified": b.updated.isoformat() if b.updated else None,
                "url": self._public_url(b),
            }
            for b in blobs
            if not b.name.endswith("/")
        ]

    def _get_flow_failed_files(self, flow_id: str, max_results: int = 500) -> List[Dict[str, Any]]:
        # Placeholder: look under failed folder if present
        blobs = self.client.list_blobs(self.bucket_name, prefix=f"{ORG_PREFIX}/{flow_id}/failed/", max_results=max_results)
        return [
            {"key": b.name, "size": b.size, "last_modified": b.updated.isoformat() if b.updated else None}
            for b in blobs
            if not b.name.endswith("/")
        ]

    def _get_flow_pending_files(self, flow_id: str) -> List[Dict[str, Any]]:
        return []

    def _get_flow_organized_files(self, flow_id: str, max_results: int = 1000) -> List[Dict[str, Any]]:
        blobs = self.client.list_blobs(self.bucket_name, prefix=f"{ORG_PREFIX}/{flow_id}/organized/", max_results=max_results)
        return [
            {"key": b.name, "size": b.size, "last_modified": b.updated.isoformat() if b.updated else None}
            for b in blobs
            if not b.name.endswith("/")
        ]

    # ---------------------------------------------------------------- search helpers
    def find_files_by_filename(
        self,
        filenames: Optional[List[str]] = None,
        prefixes: Optional[List[str]] = None,
        max_results: int = 200,
    ) -> List[str]:
        """
        Search known prefixes for blobs whose basename matches any of the provided filenames.
        """
        if not self.client or not self.bucket or not filenames:
            return []

        search_names = {name.lower() for name in filenames if name}
        if not search_names:
            return []

        prefixes = prefixes or [
            f"{TEMP_PREFIX}/",
            "flows/",
            f"{ORG_PREFIX}/",
            "attached_voucher/",
            "batches/",
            "test_uploads/",
        ]

        matches: List[str] = []
        for prefix in prefixes:
            try:
                blobs = self.client.list_blobs(
                    self.bucket_name,
                    prefix=prefix,
                    max_results=max_results
                )
                for blob in blobs:
                    if blob.name.endswith("/"):
                        continue
                    file_name = blob.name.split("/")[-1].lower()
                    if file_name in search_names:
                        matches.append(blob.name)
                        if len(matches) >= max_results:
                            return matches
            except Exception as exc:
                logger.warning(f"Failed to search prefix {prefix} for filenames {filenames}: {exc}")
        return matches

    def delete_flow_from_s3(self, flow_id: str, delete_temp_files: bool = False) -> Dict[str, Any]:
        try:
            prefixes = [f"{ORG_PREFIX}/{flow_id}/organized/", f"{ORG_PREFIX}/{flow_id}/failed/"]
            if delete_temp_files:
                prefixes.append(f"{TEMP_PREFIX}/{flow_id}/")
            deleted = 0
            for prefix in prefixes:
                for blob in self.client.list_blobs(self.bucket_name, prefix=prefix, max_results=1000):
                    blob.delete()
                    deleted += 1
            # delete metadata
            meta_blob = self._blob(self._flow_meta_key(flow_id))
            if meta_blob.exists():
                meta_blob.delete()
                deleted += 1
            return {"success": True, "deleted": deleted}
        except Exception as exc:
            return {"success": False, "error": f"Failed to delete flow: {exc}"}

    # ---------------------------------------------------------------- misc
    def delete_file(self, file_key: str) -> Dict[str, Any]:
        try:
            blob = self._blob(file_key)
            blob.delete()
            return {"success": True}
        except Exception as exc:
            return {"success": False, "error": f"Delete failed: {exc}"}

    def reclassify_failed_file(self, file_key: str, new_type: str) -> Dict[str, Any]:
        return {"success": False, "error": "Not implemented for GCS yet"}

    def get_flow_organized_tree(self, flow_id: str) -> Dict[str, Any]:
        return self.get_flow_files_from_s3(flow_id)

    def get_category_organized_tree(self, category: str) -> Dict[str, Any]:
        files = self.list_organized_files()
        if not files.get("success"):
            return files
        filtered = [f for f in files.get("files", []) if f.get("classification") == category]
        return {"success": True, "files": filtered, "count": len(filtered)}

    def cleanup_unknown_folders(self):
        return {"success": False, "error": "Not implemented for GCS yet"}

    # ---------------------------------------------------------------- utilities
    def _public_url(self, blob) -> str:
        # Use signed URL so callers can fetch even if the bucket is private
        try:
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=1),
                method="GET",
            )
        except Exception:
            return f"gs://{self.bucket_name}/{blob.name}"

    def generate_presigned_url(self, key: str, expiration: int = 3600, method: str = "GET") -> Dict[str, Any]:
        """
        Generate a V4 signed URL for a GCS object using service account credentials.
        Production-ready implementation following GCP best practices.
        Requires service account credentials with a private key for signing.
        
        Args:
            key: GCS object key (path without bucket name)
            expiration: URL expiration time in seconds (default: 1 hour, max: 7 days)
            method: HTTP method ("GET" for download/view, "PUT" for upload)
        
        Returns:
            Dict with success status, URL, and key
        """
        try:
            if not self.client or not self.bucket:
                return {"success": False, "error": "GCS client not initialized"}
            
            # Check if credentials support signing before attempting
            # Re-check credentials in case they became available after initialization
            if not self._supports_signing:
                # Try to reinitialize client if key file is now available
                env_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if env_key_path:
                    # Check if file exists and is readable
                    if os.path.exists(env_key_path):
                        try:
                            # Verify file is readable and contains private_key
                            import json
                            with open(env_key_path, 'r') as f:
                                key_data = json.load(f)
                                if key_data.get('private_key'):
                                    logger.info(f"🔄 Key file now available at {env_key_path}, reinitializing GCS client...")
                                    try:
                                        client_result = _get_gcs_client()
                                        if isinstance(client_result, tuple) and len(client_result) == 3:
                                            self.client, self._supports_signing, self._key_file_path = client_result
                                            self.bucket = self.client.bucket(self.bucket_name) if self.client else None
                                            logger.info(f"✅ Successfully reinitialized GCS client with signing support: {self._supports_signing}")
                                        else:
                                            self.client = client_result
                                            self.bucket = self.client.bucket(self.bucket_name) if self.client else None
                                    except Exception as e:
                                        logger.error(f"⚠️  Failed to reinitialize GCS client: {e}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                else:
                                    logger.warning(f"⚠️  Key file at {env_key_path} exists but does not contain private_key")
                        except Exception as e:
                            logger.error(f"⚠️  Error reading key file at {env_key_path}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                    else:
                        logger.warning(f"⚠️  GOOGLE_APPLICATION_CREDENTIALS points to {env_key_path} but file does not exist")
                
                # If still no signing support, provide detailed error
                if not self._supports_signing:
                    key_exists, key_path = _check_key_file_exists()
                    current_file_dir = os.path.dirname(__file__)
                    expected_fallback = os.path.join(current_file_dir, "voucher-storage-key.json")
                    
                    error_msg = (
                        "Service account credentials with private key are required for signed URLs.\n\n"
                        "Current status:\n"
                        f"  - Credentials support signing: No\n"
                        f"  - Key file found: {'Yes' if key_exists else 'No'}\n"
                        f"  - GOOGLE_APPLICATION_CREDENTIALS: {env_key_path or 'Not set'}\n"
                        f"  - Expected fallback path: {expected_fallback}\n"
                        f"  - Fallback file exists: {os.path.exists(expected_fallback) if expected_fallback else 'N/A'}\n"
                    )
                
                if not key_exists:
                    error_msg += (
                        "\nTo fix this, download the service account key from GCP Console:\n"
                        "1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts?project=rocasoft\n"
                        "2. Find service account: voucher-storage-sa@rocasoft.iam.gserviceaccount.com\n"
                        "3. Click on the service account > Keys tab\n"
                        "4. Click 'Add Key' > 'Create new key' > Select 'JSON'\n"
                        "5. Download the JSON file\n"
                        "6. Save it as 'voucher-storage-key.json' in the backend directory:\n"
                        f"   {expected_fallback}\n"
                        "   OR set GOOGLE_APPLICATION_CREDENTIALS environment variable:\n"
                        "   export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/voucher-storage-key.json\"\n"
                        "7. Restart the backend server\n"
                    )
                else:
                    # Key file exists but credentials don't support signing
                    # This could mean the file is invalid or the backend needs restart
                    try:
                        import json
                        with open(key_path, 'r') as f:
                            key_data = json.load(f)
                            has_private_key = bool(key_data.get('private_key'))
                            error_msg += (
                                f"\nKey file found at: {key_path}\n"
                                f"  - File contains private_key: {has_private_key}\n"
                                f"  - Service account: {key_data.get('client_email', 'Unknown')}\n"
                            )
                    except Exception as e:
                        error_msg += (
                            f"\nKey file found at: {key_path}\n"
                            f"  - Could not read/parse file: {e}\n"
                        )
                    
                    error_msg += (
                        "\nThe backend server may need to be restarted to load the credentials.\n"
                        "If the file is valid, restart the backend server.\n"
                    )
                
                return {"success": False, "error": error_msg}
            
            # Validate expiration (max 7 days for security)
            max_expiration = 7 * 24 * 3600  # 7 days in seconds
            if expiration > max_expiration:
                expiration = max_expiration
            
            # Normalize key - remove bucket name prefix if present
            actual_key = key
            if '/' in key:
                path_parts = key.split('/', 1)
                if path_parts[0] == self.bucket_name or path_parts[0] == 'voucher-bucket-1':
                    actual_key = path_parts[1]
            
            # Handle gs:// URLs
            if actual_key.startswith('gs://'):
                parts = actual_key.replace('gs://', '').split('/', 1)
                if len(parts) == 2:
                    actual_key = parts[1]
            
            # Get blob
            blob = self._blob(actual_key)
            
            # For GET requests, verify blob exists to provide better error messages
            # For PUT requests, we can generate signed URLs even if blob doesn't exist yet
            if method.upper() == "GET":
                try:
                    blob_exists = blob.exists()
                    if not blob_exists:
                        return {
                            "success": False,
                            "error": f"Blob does not exist: {actual_key}"
                        }
                except Exception as exists_error:
                    # If existence check fails, log it but continue - might be a permission issue
                    # The signed URL generation might still work
                    logger.warning(f"Failed to check blob existence for {actual_key}: {exists_error}")
            
            # Generate V4 signed URL following GCP best practices
            # For PUT, this works even if blob doesn't exist yet (useful for upload URLs)
            try:
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration),
                    method=method.upper()  # GET for download, PUT for upload
                )
            except Exception as url_error:
                # Re-raise to be caught by outer exception handler
                raise
            
            return {
                "success": True, 
                "url": url, 
                "key": actual_key, 
                "expires_in": expiration,
                "method": method.upper()
            }
            
        except Exception as exc:
            error_msg = str(exc)
            
            # Provide helpful error messages for common issues
            if 'private key' in error_msg.lower() or 'credentials' in error_msg.lower():
                key_exists, key_path = _check_key_file_exists()
                error_detail = (
                    "Service account credentials with private key are required for signed URLs.\n\n"
                    f"Service Account: voucher-storage-sa@rocasoft.iam.gserviceaccount.com\n"
                    f"Key ID: 787a7d7f282807b45e5b3795325a43cb945fcb75\n\n"
                    f"Key file status: {'Found' if key_exists else 'Not found'}\n"
                )
                
                if not key_exists:
                    error_detail += (
                        "\nTo download the service account key:\n"
                        "1. Go to: https://console.cloud.google.com/iam-admin/serviceaccounts?project=rocasoft\n"
                        "2. Find: voucher-storage-sa@rocasoft.iam.gserviceaccount.com\n"
                        "3. Click on the service account > Keys tab\n"
                        "4. Click 'Add Key' > 'Create new key' > Select 'JSON'\n"
                        "5. Download and save as 'voucher-storage-key.json'\n"
                        "6. Place it in: " + os.path.join(os.path.dirname(__file__), "voucher-storage-key.json") + "\n"
                        "   OR set GOOGLE_APPLICATION_CREDENTIALS environment variable\n"
                        "7. Restart the backend server\n"
                    )
                else:
                    error_detail += (
                        f"\nKey file found at: {key_path}\n"
                        "The file may be invalid or corrupted. Please verify:\n"
                        "1. The file is valid JSON\n"
                        "2. It contains a 'private_key' field\n"
                        "3. The service account has the correct permissions\n"
                        "4. Restart the backend server after any changes\n"
                    )
                
                error_detail += f"\nOriginal error: {error_msg}"
                
                return {
                    "success": False,
                    "error": error_detail
                }
            elif 'not found' in error_msg.lower() or 'does not exist' in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Blob does not exist: {actual_key if 'actual_key' in locals() else key}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to generate signed URL: {error_msg}"
                }


# Export instance to keep the same import style
s3_service = S3Service()
