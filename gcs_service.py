"""
Google Cloud Storage Service for uploading organized voucher folders
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from google.cloud import storage
from google.oauth2 import service_account
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class GCSVoucherService:
    def __init__(self):
        """Initialize Google Cloud Storage client with flexible credential loading"""
        self.bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("GCS_BUCKET") or "voucher-bucket-1"
        self.project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GCS_PROJECT_ID") or "rocasoft"
        
        # Initialize GCS client with flexible credential loading
        try:
            credentials = None
            key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            # Try environment variable first
            if key_path and os.path.exists(key_path):
                credentials = service_account.Credentials.from_service_account_file(
                    key_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info(f"Using GCS credentials from GOOGLE_APPLICATION_CREDENTIALS: {key_path}")
            else:
                # Fallback to local key file
                fallback_key = os.path.join(os.path.dirname(__file__), "voucher-storage-key.json")
                if os.path.exists(fallback_key):
                    credentials = service_account.Credentials.from_service_account_file(
                        fallback_key,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    logger.info(f"Using GCS credentials from local key file: {fallback_key}")
                else:
                    # Last resort: application-default credentials
                    logger.info("Using application-default credentials for GCS")
                    # Will use default credentials if available
            
            # Initialize client with credentials (or None for default)
            if credentials:
                self.client = storage.Client(credentials=credentials, project=self.project_id)
            else:
                self.client = storage.Client(project=self.project_id)
            
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"GCS client initialized for bucket: {self.bucket_name}, project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise

    @staticmethod
    def _extract_branch_number_from_document_no(document_no: str) -> str:
        """Extract numeric branch code from document number (e.g., MPU01-85285 -> '01')."""
        if not document_no:
            return None
        try:
            m = re.match(r"^[A-Z]+(\d{1,3})", document_no.strip())
            if m:
                digits = m.group(1)
                if len(digits) >= 2:
                    return digits
                return f"{int(digits):02d}"
            return None
        except Exception:
            return None

    @staticmethod
    def _format_branch_dir_name(branch_hint: str) -> str:
        """Format 'Branch NN' from various hints (digits, 'Branch NN', etc.)."""
        if not branch_hint:
            return ""
        try:
            if branch_hint.lower().startswith('branch '):
                return branch_hint
            if branch_hint.isdigit():
                return f"Branch {int(branch_hint):02d}"
            m = re.match(r"^(\d{1,3})", branch_hint)
            if m:
                return f"Branch {int(m.group(1)):02d}"
        except Exception:
            pass
        return branch_hint

    def upload_folder_to_gcs(self, local_folder_path: str, gcs_folder_prefix: str = None) -> Dict[str, Any]:
        """
        Upload organized voucher images to GCS with simplified structure
        
        Args:
            local_folder_path: Path to local organized_vouchers folder
            gcs_folder_prefix: Optional prefix for GCS folder structure
            
        Returns:
            Dict with upload results
        """
        try:
            local_path = Path(local_folder_path)
            if not local_path.exists():
                raise FileNotFoundError(f"Local folder not found: {local_folder_path}")
            
            uploaded_files = []
            failed_files = []

            # Preserve relative structure under organized_vouchers in GCS
            # Example: organized_vouchers/branch/year/mon/date/type/filename
            for file_path in local_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf', '.txt']:
                    try:
                        rel = file_path.resolve().relative_to(local_path.resolve())
                    except Exception:
                        rel = Path(file_path.name)

                    gcs_blob_name = f"organized_vouchers/{rel.as_posix()}"

                    try:
                        # Upload file to GCS
                        blob = self.bucket.blob(gcs_blob_name)
                        blob.upload_from_filename(str(file_path))

                        # Try to derive voucher_type from path (last dir)
                        voucher_type = file_path.parent.name

                        # Set metadata
                        blob.metadata = {
                            'voucher_type': voucher_type,
                            'original_filename': file_path.name,
                            'upload_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'file_size': str(file_path.stat().st_size),
                            'folder_structure': f"organized_vouchers/{rel.parent.as_posix()}/"
                        }
                        blob.patch()

                        uploaded_files.append({
                            'local_path': str(file_path),
                            'gcs_path': f"gs://{self.bucket_name}/{gcs_blob_name}",
                            'voucher_type': voucher_type,
                            'filename': file_path.name,
                            'size': file_path.stat().st_size
                        })

                        logger.info(f"Uploaded: {file_path.name} -> {gcs_blob_name}")

                    except Exception as e:
                        failed_files.append({
                            'file': str(file_path),
                            'error': str(e)
                        })
                        logger.error(f"Failed to upload {file_path.name}: {e}")
            
            # Create simple summary metadata file
            summary_data = {
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_vouchers': len(uploaded_files),
                'failed_uploads': len(failed_files),
                'voucher_types': list(set([f.get('voucher_type', 'unknown') for f in uploaded_files])),
                'uploaded_vouchers': uploaded_files
            }
            
            # Upload summary metadata
            summary_blob_name = f"organized_vouchers/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_blob = self.bucket.blob(summary_blob_name)
            summary_blob.upload_from_string(
                json.dumps(summary_data, indent=2),
                content_type='application/json'
            )
            
            return {
                'success': True,
                'gcs_folder_prefix': gcs_folder_prefix,
                'total_vouchers': len(uploaded_files),
                'uploaded_vouchers': uploaded_files,
                'failed_uploads': failed_files,
                'summary_url': f"gs://{self.bucket_name}/{summary_blob_name}"
            }
            
        except Exception as e:
            logger.error(f"Failed to upload folder to GCS: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def upload_processed_documents(self, processed_results: List[Dict[str, Any]], job_id: str) -> Dict[str, Any]:
        """
        Upload only the documents that were just processed in a batch job
        
        Args:
            processed_results: List of processing results from the batch job
            job_id: Batch job ID for organization
            
        Returns:
            Dict with upload results
        """
        try:
            uploaded_files = []
            failed_files = []
            
            for result in processed_results:
                document_id = result.get('document_id')
                organized_base = Path(__file__).parent.parent / "AIServices" / "organized_vouchers"

                # Prefer explicit files from processing result
                candidate_files: List[Path] = []
                for key in ('image_file', 'text_file'):
                    file_val = result.get(key)
                    if file_val:
                        p = Path(file_val)
                        if p.exists() and p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf', '.txt']:
                            candidate_files.append(p)

                # If no explicit files provided, skip this result
                if not candidate_files:
                    logger.warning(f"No files found for document {document_id} to upload to GCS")
                    continue

                # Determine relative structure from folder_path if present
                rel_structure = None
                folder_path = result.get('folder_path')
                if folder_path:
                    try:
                        rel_structure = Path(folder_path).resolve().relative_to(organized_base.resolve())
                    except Exception:
                        rel_structure = None

                # Expected structure: Branch NN/year/mon/date/voucher_type
                branch_id = None
                year = None
                month = None
                date_str = None
                voucher_type = result.get('voucher_type', 'UNKNOWN')

                if rel_structure and len(rel_structure.parts) >= 5:
                    branch_id, year, month, date_str, voucher_type = rel_structure.parts[:5]
                else:
                    # Fallback to document-derived branch or environment/date
                    derived_branch_num = self._extract_branch_number_from_document_no(result.get('document_no'))
                    branch_id = self._format_branch_dir_name(derived_branch_num or os.getenv('BRANCH_ID', ''))
                    now = datetime.now()
                    year = str(now.year)
                    import calendar as _calendar
                    month = _calendar.month_abbr[now.month].lower()
                    date_str = f"{now.day}-{now.month}-{now.year}"

                for file_path in candidate_files:
                    gcs_blob_name = f"organized_vouchers/{branch_id}/{year}/{month}/{date_str}/{voucher_type}/{file_path.name}"

                    try:
                        blob = self.bucket.blob(gcs_blob_name)
                        if blob.exists():
                            logger.info(f"File {file_path.name} already exists in GCS, skipping upload")
                            status = 'already_exists'
                        else:
                            blob.upload_from_filename(str(file_path))
                            status = 'uploaded'

                        # Set metadata
                        blob.metadata = {
                            'voucher_type': voucher_type,
                            'document_no': result.get('document_no', document_id or 'unknown'),
                            'original_filename': file_path.name,
                            'upload_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'job_id': job_id,
                            'file_size': str(file_path.stat().st_size),
                            'file_type': file_path.suffix.lower(),
                            'folder_structure': f'organized_vouchers/{branch_id}/{year}/{month}/{date_str}/{voucher_type}/',
                            'branch_id': branch_id,
                            'year': year,
                            'month': month,
                            'date': date_str,
                            'document_date': result.get('document_date'),
                            'ocr_success': result.get('success', False)
                        }
                        blob.patch()

                        uploaded_files.append({
                            'local_path': str(file_path),
                            'gcs_path': f"gs://{self.bucket_name}/{gcs_blob_name}",
                            'voucher_type': voucher_type,
                            'document_no': result.get('document_no', document_id or 'unknown'),
                            'filename': file_path.name,
                            'size': file_path.stat().st_size,
                            'file_type': file_path.suffix.lower(),
                            'status': status
                        })

                        logger.info(f"Uploaded processed document: {file_path.name} -> {gcs_blob_name}")

                    except Exception as e:
                        failed_files.append({
                            'document_no': result.get('document_no', document_id or 'unknown'),
                            'voucher_type': voucher_type,
                            'filename': file_path.name,
                            'error': str(e)
                        })
                        logger.error(f"Failed to upload {file_path.name}: {e}")
            
            # Create summary for this batch
            summary_data = {
                'job_id': job_id,
                'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_processed': len(processed_results),
                'total_uploaded': len(uploaded_files),
                'failed_uploads': len(failed_files),
                'voucher_types': list(set([f['voucher_type'] for f in uploaded_files])),
                'uploaded_documents': uploaded_files
            }
            
            # Upload batch summary to organized_vouchers root
            summary_blob_name = f"organized_vouchers/batch_summary_{job_id}.json"
            summary_blob = self.bucket.blob(summary_blob_name)
            summary_blob.upload_from_string(
                json.dumps(summary_data, indent=2),
                content_type='application/json'
            )
            
            return {
                'success': True,
                'job_id': job_id,
                'gcs_structure': 'organized_vouchers/Branch NN/{year}/{mon}/{date}/{voucher_type}/{filename}',
                'total_processed': len(processed_results),
                'total_uploaded': len(uploaded_files),
                'uploaded_documents': uploaded_files,
                'failed_uploads': failed_files,
                'summary_url': f"gs://{self.bucket_name}/{summary_blob_name}"
            }
            
        except Exception as e:
            logger.error(f"Failed to upload processed documents: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def upload_single_voucher(self, local_file_path: str, voucher_type: str, document_no: str) -> Dict[str, Any]:
        """
        Upload a single voucher file to GCS with simplified structure
        
        Args:
            local_file_path: Path to the voucher file
            voucher_type: Type of voucher (REC, PAY, MSL, etc.)
            document_no: Document number
            
        Returns:
            Dict with upload result
        """
        try:
            file_path = Path(local_file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {local_file_path}")
            
            # Create GCS blob name: organized_vouchers/branch/year/mon/date/voucher_type/filename
            import calendar as _calendar
            now = datetime.now()
            # Prefer deriving from document number (Branch NN)
            derived_branch_num = self._extract_branch_number_from_document_no(document_no)
            branch_id = self._format_branch_dir_name(derived_branch_num or os.getenv('BRANCH_ID', ''))
            year = str(now.year)
            month = _calendar.month_abbr[now.month].lower()
            date_str = f"{now.day}-{now.month}-{now.year}"
            gcs_blob_name = f"organized_vouchers/{branch_id}/{year}/{month}/{date_str}/{voucher_type}/{file_path.name}"
            
            # Upload file
            blob = self.bucket.blob(gcs_blob_name)
            blob.upload_from_filename(str(file_path))
            
            # Set metadata
            blob.metadata = {
                'voucher_type': voucher_type,
                'document_no': document_no,
                'original_filename': file_path.name,
                'upload_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'branch_id': branch_id,
                'year': year,
                'month': month,
                'date': date_str,
                'document_date': document_no, # Approximation, no date available
                'file_size': str(file_path.stat().st_size)
            }
            blob.patch()
            
            return {
                'success': True,
                'gcs_path': f"gs://{self.bucket_name}/{gcs_blob_name}",
                'voucher_type': voucher_type,
                'document_no': document_no,
                'filename': file_path.name,
                'branch_id': branch_id,
                'year': year,
                'month': month,
                'date': date_str,
                'document_date': document_no, # Approximation
                'file_size': file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to upload single voucher: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_folder_structure(self, folder_path: Path) -> Dict[str, Any]:
        """Get folder structure for metadata"""
        structure = {}
        
        for item in folder_path.iterdir():
            if item.is_dir():
                structure[item.name] = self._get_folder_structure(item)
            else:
                structure[item.name] = {
                    'type': 'file',
                    'size': item.stat().st_size,
                    'modified': str(item.stat().st_mtime)
                }
        
        return structure

    def list_uploaded_vouchers(self, prefix: str = "") -> List[Dict[str, Any]]:
        """
        List all uploaded vouchers in the bucket
        
        Args:
            prefix: GCS prefix to filter results
            
        Returns:
            List of voucher metadata
        """
        try:
            blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
            vouchers = []
            
            for blob in blobs:
                # Only list files, skip summaries
                if blob.name.endswith(('summary.json',)):
                    continue
                if not blob.name.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.txt')):
                    continue

                path_parts = blob.name.split('/')
                filename = path_parts[-1]
                voucher_type = 'unknown'
                branch_id = 'unknown'
                year = 'unknown'
                month = 'unknown'
                date_str = 'unknown'

                # Expect: organized_vouchers/branch/year/mon/date/voucher_type/filename
                if len(path_parts) >= 7 and path_parts[0] == 'organized_vouchers':
                    branch_id = path_parts[1]
                    year = path_parts[2]
                    month = path_parts[3]
                    date_str = path_parts[4]
                    voucher_type = path_parts[5]

                vouchers.append({
                    'filename': filename,
                    'voucher_type': voucher_type,
                    'branch_id': branch_id,
                    'year': year,
                    'month': month,
                    'date': date_str,
                    'document_date': blob.metadata.get('document_date') if blob.metadata else None,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'metadata': blob.metadata or {},
                    'url': f"gs://{self.bucket_name}/{blob.name}",
                    'gcs_path': blob.name
                })
            
            # Sort by scan date and voucher type
            vouchers.sort(key=lambda x: (x.get('year', ''), x.get('month', ''), x.get('date', ''), x.get('voucher_type', ''), x.get('filename', '')))
            return vouchers
            
        except Exception as e:
            logger.error(f"Failed to list vouchers: {e}")
            return []

    def download_voucher(self, gcs_path: str, local_download_path: str) -> Dict[str, Any]:
        """
        Download a voucher from GCS to local storage
        
        Args:
            gcs_path: GCS path (gs://bucket/path or just path)
            local_download_path: Local path to save the file
            
        Returns:
            Dict with download result
        """
        try:
            # Extract blob name from gs:// path
            if gcs_path.startswith('gs://'):
                blob_name = gcs_path.split('/', 3)[3]  # Remove gs://bucket/
            else:
                blob_name = gcs_path
            
            blob = self.bucket.blob(blob_name)
            
            # Ensure local directory exists
            Path(local_download_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_download_path)
            
            return {
                'success': True,
                'local_path': local_download_path,
                'file_size': Path(local_download_path).stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Failed to download voucher: {e}")
            return {
                'success': False,
                'error': str(e)
            }
