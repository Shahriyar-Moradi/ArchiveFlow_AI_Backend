"""
Document processing service adapted from Lambda OCR service
Uses Anthropic API directly and GCS for storage
"""
import os
import base64
import json
import re
import struct
import zlib
import time
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Initialize logger BEFORE any code that might use it
logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available - zero-shot image validation will be disabled")

from anthropic import Anthropic

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from services.anthropic_utils import detect_model_not_found_error
from services.json_utils import extract_json_from_text

# Module-level singleton for CLIP classifier - loads once, reused by all instances
# ZERO-SHOT VALIDATION TEMPORARILY DISABLED - Code preserved for future use
_clip_classifier_singleton = None
_clip_classifier_lock = None

# ZERO-SHOT VALIDATION TEMPORARILY DISABLED - Code preserved for future use
# def _get_clip_classifier():
#     """
#     Get or create the CLIP classifier singleton.
#     Model loads only ONCE and stays in memory for all subsequent uses.
#     
#     Returns:
#         CLIP classifier pipeline instance or None if unavailable
#     """
#     global _clip_classifier_singleton, _clip_classifier_lock
#     
#     # Import threading only if needed
#     if _clip_classifier_lock is None:
#         import threading
#         _clip_classifier_lock = threading.Lock()
#     
#     # If already loaded, return the singleton (FAST PATH - no lock needed)
#     if _clip_classifier_singleton is not None:
#         return _clip_classifier_singleton
#     
#     # Thread-safe initialization (only happens once)
#     with _clip_classifier_lock:
#         # Double-check pattern - another thread might have loaded it
#         if _clip_classifier_singleton is not None:
#             return _clip_classifier_singleton
#         
#         # Load the model (ONLY ONCE)
#         if not TRANSFORMERS_AVAILABLE:
#             logger.warning("transformers library not available - zero-shot image validation will be disabled")
#             return None
#         
#         model_name = "openai/clip-vit-large-patch14-336"
#         max_retries = 3
#         retry_delay = 2  # seconds
#         
#         for attempt in range(1, max_retries + 1):
#             try:
#                 logger.info(f"Loading CLIP classifier model (attempt {attempt}/{max_retries})...")
#                 logger.info(f"Model: {model_name}")
#                 logger.info("This may take a minute on first load (downloading from HuggingFace)...")
#                 
#                 # Check for HuggingFace cache directory
#                 hf_home = os.environ.get('HF_HOME') or os.path.expanduser('~/.cache/huggingface')
#                 logger.info(f"HuggingFace cache location: {hf_home}")
#                 
#                 # Try to load the pipeline
#                 # The pipeline will automatically download the model if not cached
#                 try:
#                     import torch
#                     
#                     logger.info("Attempting to load CLIP model with transformers pipeline...")
#                     
#                     # Determine device (GPU if available, else CPU)
#                     device = 0 if torch.cuda.is_available() else -1
#                     if device == 0:
#                         logger.info("Using GPU for CLIP model")
#                     else:
#                         logger.info("Using CPU for CLIP model")
#                     
#                     # Load pipeline - it will download from HuggingFace if not cached
#                     # By default, local_files_only=False, so it will download if needed
#                     _clip_classifier_singleton = pipeline(
#                         "zero-shot-image-classification",
#                         model=model_name,
#                         device=device
#                     )
#                     
#                     logger.info("✅ CLIP classifier model loaded successfully and cached in memory")
#                     logger.info("All subsequent classifications will use this cached model (no reload)")
#                     return _clip_classifier_singleton
#                     
#                 except Exception as pipeline_error:
#                     error_str = str(pipeline_error)
#                     logger.warning(f"Pipeline loading failed: {error_str}")
#                     
#                     # If it's a connection error, we'll retry in the outer loop
#                     if "connect" in error_str.lower() or "network" in error_str.lower():
#                         raise  # Re-raise to trigger retry logic
#                     
#                     # For other errors, try alternative loading method
#                     logger.info("Trying alternative loading method...")
#                     try:
#                         from transformers import CLIPProcessor, CLIPModel
#                         import torch
#                         
#                         logger.info("Attempting to load CLIP model components separately...")
#                         processor = CLIPProcessor.from_pretrained(model_name, local_files_only=False)
#                         model = CLIPModel.from_pretrained(model_name, local_files_only=False)
#                         
#                         # Move model to device
#                         device_str = "cuda" if torch.cuda.is_available() else "cpu"
#                         model = model.to(device_str)
#                         model.eval()
#                         
#                         # Create a simple wrapper for zero-shot classification
#                         def classify_image(image, candidate_labels):
#                             inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
#                             # Move inputs to device
#                             inputs = {k: v.to(device_str) for k, v in inputs.items()}
#                             
#                             with torch.no_grad():
#                                 outputs = model(**inputs)
#                                 logits_per_image = outputs.logits_per_image
#                                 probs = logits_per_image.softmax(dim=1)
#                             
#                             results = []
#                             for i, label in enumerate(candidate_labels):
#                                 results.append({
#                                     'label': label,
#                                     'score': float(probs[0][i])
#                                 })
#                             return sorted(results, key=lambda x: x['score'], reverse=True)
#                         
#                         # Create a simple pipeline-like object
#                         class SimpleCLIPClassifier:
#                             def __init__(self, model, processor, classify_fn):
#                                 self.model = model
#                                 self.processor = processor
#                                 self._classify = classify_fn
#                             
#                             def __call__(self, image, candidate_labels):
#                                 return self._classify(image, candidate_labels)
#                         
#                         _clip_classifier_singleton = SimpleCLIPClassifier(model, processor, classify_image)
#                         logger.info("✅ CLIP classifier model loaded successfully (alternative method)")
#                         logger.info("All subsequent classifications will use this cached model (no reload)")
#                         return _clip_classifier_singleton
#                         
#                     except Exception as alt_error:
#                         logger.warning(f"Alternative loading method also failed: {alt_error}")
#                         raise pipeline_error  # Re-raise original error to trigger retry
#                 
#             except Exception as e:
#                 error_msg = str(e)
#                 logger.error(f"Failed to load CLIP classifier (attempt {attempt}/{max_retries}): {error_msg}")
#                 
#                 # Check if it's a network/connection error
#                 if "connect" in error_msg.lower() or "network" in error_msg.lower() or "timeout" in error_msg.lower():
#                     if attempt < max_retries:
#                         logger.warning(f"Network error detected. Retrying in {retry_delay} seconds...")
#                         time.sleep(retry_delay)
#                         retry_delay *= 2  # Exponential backoff
#                         continue
#                     else:
#                         logger.error("Max retries reached. Network connection to HuggingFace failed.")
#                         logger.error("Possible solutions:")
#                         logger.error("1. Check internet connection")
#                         logger.error("2. Check firewall/proxy settings")
#                         logger.error("3. Set HF_HOME environment variable to use cached models")
#                         logger.error("4. Pre-download model: python -c 'from transformers import pipeline; pipeline(\"zero-shot-image-classification\", model=\"openai/clip-vit-large-patch14-336\")'")
#                 
#                 # Check if it's a cache/local files issue
#                 elif "local_files_only" in error_msg.lower() or "cache" in error_msg.lower():
#                     logger.warning("Cache/local files issue detected. Trying to download from HuggingFace...")
#                     if attempt < max_retries:
#                         time.sleep(retry_delay)
#                         retry_delay *= 2
#                         continue
#                 
#                 # For other errors, log and retry once more
#                 if attempt < max_retries:
#                     logger.warning(f"Retrying in {retry_delay} seconds...")
#                     time.sleep(retry_delay)
#                     retry_delay *= 2
#                 else:
#                     logger.error(f"Failed to load CLIP classifier after {max_retries} attempts.")
#                     logger.error(f"Final error: {error_msg}")
#                     import traceback
#                     logger.error(f"Full traceback:\n{traceback.format_exc()}")
#                     return None
#         
#         return None

class DocumentProcessor:
    """Document processing service using Anthropic API for OCR"""
    
    def __init__(self):
        """Initialize the Document Processor"""
        logger.info("Initializing Document Processor...")
        logger.info(f"PyMuPDF available: {PYMUPDF_AVAILABLE}")
        logger.info(f"PIL available: {PIL_AVAILABLE}")
        
        # Initialize Anthropic client
        if not settings.anthropic_api_key_configured:
            raise ValueError("ANTHROPIC_API_KEY is required for document processing")
        
        try:
            self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model = settings.ANTHROPIC_MODEL
            logger.info(f"Anthropic client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
        
        # Define voucher type prefixes
        self.voucher_types = {
            "MPU": "MPU",
            "MPV": "MPV",
            "MRT": "MRT",
            "MSL": "MSL",
            "REC": "REC",
            "PAY": "PAY",
            "MJV": "MJV"
        }
        
        # Month mapping
        self.month_names = {
            1: "jan", 2: "feb", 3: "mar", 4: "apr",
            5: "may", 6: "jun", 7: "jul", 8: "aug",
            9: "sep", 10: "oct", 11: "nov", 12: "dec"
        }
        
        # ZERO-SHOT VALIDATION TEMPORARILY DISABLED - Code preserved for future use
        # # Zero-shot image classifier configuration
        # # NOTE: Classifier is loaded lazily via singleton pattern - model loads only ONCE
        # self.CONFIDENCE_THRESHOLD = 0.90  # 90% confidence required for document validation
        # self.VALIDATION_LABELS = ["Valid document", "other"]
        # 
        # # Don't load classifier here - it will be loaded on first use via singleton
        # # This ensures the model loads only once and stays in memory
        # logger.info("DocumentProcessor initialized. CLIP classifier will load on first use (singleton pattern)")
    
    def _detect_image_format(self, image_path: str) -> str:
        """Detect actual image format from file content (magic bytes)"""
        with open(image_path, "rb") as f:
            header = f.read(16)
        
        # Check magic bytes
        if header.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif header.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'png'
        elif header.startswith(b'%PDF'):
            return 'pdf'
        elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            return 'gif'
        elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
            return 'webp'
        else:
            logger.warning(f"Unknown image format for: {image_path}")
            return 'unknown'
    
    def _normalize_image_format(self, image_path: str) -> tuple[str, str]:
        """
        Ensure image format matches its extension by converting if needed.
        Returns: (normalized_path, actual_format)
        """
        if not PIL_AVAILABLE:
            # Without PIL, we can't convert - just detect and warn
            actual_format = self._detect_image_format(image_path)
            logger.warning(f"PIL not available - cannot convert format. Actual format: {actual_format}")
            return image_path, actual_format
        
        try:
            actual_format = self._detect_image_format(image_path)
            file_ext = os.path.splitext(image_path)[1].lower().lstrip('.')
            
            # If format matches extension, no conversion needed
            if (actual_format == 'jpeg' and file_ext in ['jpg', 'jpeg']) or \
               (actual_format == file_ext):
                logger.info(f"Image format matches extension: {actual_format}")
                return image_path, actual_format
            
            # Format mismatch - need to convert
            logger.warning(f"Format mismatch detected: file is {actual_format} but extension is .{file_ext}")
            
            if actual_format == 'pdf':
                # Don't convert PDFs
                return image_path, 'pdf'
            
            # Convert image to match its extension or to PNG as default
            target_format = file_ext if file_ext in ['png', 'jpg', 'jpeg'] else 'png'
            
            # Open image with PIL
            img = Image.open(image_path)
            
            # Convert RGBA to RGB for JPEG
            if target_format in ['jpg', 'jpeg'] and img.mode in ['RGBA', 'LA', 'P']:
                logger.info(f"Converting {img.mode} to RGB for JPEG")
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ['RGBA', 'LA'] else None)
                img = rgb_img
            
            # Create new filename with correct format
            base_path = os.path.splitext(image_path)[0]
            if target_format in ['jpg', 'jpeg']:
                new_path = base_path + '_converted.jpg'
                img.save(new_path, 'JPEG', quality=95)
                logger.info(f"Converted to JPEG: {new_path}")
                return new_path, 'jpeg'
            else:
                new_path = base_path + '_converted.png'
                img.save(new_path, 'PNG')
                logger.info(f"Converted to PNG: {new_path}")
                return new_path, 'png'
                
        except Exception as e:
            logger.error(f"Error normalizing image format: {e}")
            # Return original path and detected format
            actual_format = self._detect_image_format(image_path)
            return image_path, actual_format
    
    def _encode_image_to_base64(self, image_path: str) -> tuple[str, str]:
        """
        Encode image or PDF to base64 with validation and format detection.
        Returns: (base64_data, media_type)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        file_size = os.path.getsize(image_path)
        file_ext = os.path.splitext(image_path)[1].lower()
        logger.info(f"Encoding file: {image_path} (size: {file_size} bytes, extension: {file_ext})")
        
        if file_size == 0:
            raise ValueError(f"File is empty: {image_path}")
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            logger.warning(f"Large file ({file_size / 1024 / 1024:.1f}MB)")
        
        # Normalize image format to match extension
        normalized_path, actual_format = self._normalize_image_format(image_path)
        
        # Determine correct media type based on actual format
        if actual_format == 'jpeg':
            media_type = "image/jpeg"
        elif actual_format == 'png':
            media_type = "image/png"
        elif actual_format == 'pdf':
            media_type = "application/pdf"
        elif actual_format == 'gif':
            media_type = "image/gif"
        elif actual_format == 'webp':
            media_type = "image/webp"
        else:
            # Default to PNG for unknown formats
            media_type = "image/png"
            logger.warning(f"Unknown format, defaulting to image/png")
        
        logger.info(f"Using media type: {media_type} for format: {actual_format}")
        
        with open(normalized_path, "rb") as image_file:
            image_data = image_file.read()
            if not image_data:
                raise ValueError(f"Failed to read file data from: {normalized_path}")
            
            encoded = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"File encoded successfully: {len(encoded)} base64 characters")
            
            # Clean up converted file if it's different from original
            if normalized_path != image_path and os.path.exists(normalized_path):
                try:
                    os.remove(normalized_path)
                    logger.info(f"Cleaned up converted file: {normalized_path}")
                except:
                    pass
            
            return encoded, media_type
    
    def _parse_document_date(self, date_str: Optional[str]) -> tuple[int, int, int]:
        """Parse document date and return year, month, day components"""
        if not date_str:
            now = datetime.now()
            return now.year, now.month, now.day
        
        try:
            date_formats = [
                "%d-%m-%Y",  # 02-06-2025
                "%d/%m/%Y",  # 02/06/2025
                "%Y-%m-%d",  # 2025-06-02
                "%Y/%m/%d",  # 2025/06/02
                "%m-%d-%Y",  # 06-02-2025
                "%m/%d/%Y",  # 06/02/2025
            ]
            
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date_str.strip(), fmt)
                    return date_obj.year, date_obj.month, date_obj.day
                except ValueError:
                    continue
            
            # Try to extract components
            match = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', date_str)
            if match:
                day, month, year = match.groups()
                year = int(year)
                if year < 100:
                    year += 2000 if year < 50 else 1900
                return year, int(month), int(day)
            
            match = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
            if match:
                year, month, day = match.groups()
                return int(year), int(month), int(day)
                
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
        
        # Default to current date
        now = datetime.now()
        return now.year, now.month, now.day
    
    def _create_organized_path(
        self,
        document_no: str,
        document_date: Optional[str],
        branch_id: Optional[str],
        voucher_type: str
    ) -> Optional[str]:
        """Create the organized path structure"""
        try:
            year, month, day = self._parse_document_date(document_date)
            
            # Format branch ID
            if branch_id:
                try:
                    branch_num = int(branch_id)
                    branch_folder = f"Branch {branch_num:02d}"
                except:
                    branch_folder = f"Branch {branch_id}"
            else:
                branch_folder = "Branch 01"
            
            # Get month name
            month_name = self.month_names.get(month, f"month{month:02d}")
            
            # Format date folder
            date_folder = f"{day}-{month}-{year}"
            
            # Ensure voucher type is valid
            if not voucher_type or voucher_type not in self.voucher_types:
                voucher_type = self._extract_document_no_prefix(document_no)
                if not voucher_type or voucher_type not in self.voucher_types:
                    logger.warning(f"Invalid voucher type - cannot create organized path")
                    return None
            
            # Build the path
            path_components = [
                "organized_vouchers",
                str(year),
                branch_folder,
                month_name,
                date_folder,
                voucher_type
            ]
            
            organized_path = "/".join(path_components)
            logger.info(f"Created organized path: {organized_path}")
            return organized_path
            
        except Exception as e:
            logger.error(f"Error creating organized path: {e}")
            return None
    
    def _create_general_organized_path(self, document_type: str, document_date: Optional[str], document_no: Optional[str], fallback_id: Optional[str] = None) -> Optional[str]:
        """
        Create organized path for general document types
        
        Structure: organized_documents/{document_type}/{year}/{month}/{date}/{document_no}/
        If document_no is empty, uses fallback_id or "no_doc_no"
        """
        try:
            year, month, day = self._parse_document_date(document_date)
            month_name = self.month_names.get(month, "unknown")
            
            # Sanitize document type for folder name
            doc_type_safe = document_type.lower().replace(' ', '_').replace('/', '_')
            
            # Use fallback if document_no is empty
            if document_no and document_no.strip():
                doc_no_safe = re.sub(r'[<>:"/\\|?*]', '_', document_no.strip())
            else:
                # Use fallback_id or default value
                doc_no_safe = fallback_id if fallback_id else "no_doc_no"
                logger.warning(f"Document number is empty, using fallback: {doc_no_safe}")
            
            organized_path = f"{settings.ORGANIZED_FOLDER}/{doc_type_safe}/{year}/{month_name}/{day}-{month}-{year}/{doc_no_safe}"
            
            logger.info(f"Generated general organized path: {organized_path}")
            return organized_path
            
        except Exception as e:
            logger.error(f"Error creating general organized path: {e}")
            return None
    
    def _extract_document_no_prefix(self, document_no: Optional[str]) -> Optional[str]:
        """Extract the prefix from Document No (e.g., MPU from MPU01-85285)"""
        if not document_no:
            return None
        
        match = re.match(r'^([A-Z]+)', document_no.strip())
        if match:
            prefix = match.group(1)
            return prefix if prefix in self.voucher_types else None
        return None
    
    def _classify_document_type(self, image_path: str) -> Dict[str, Any]:
        """
        Classify document type using general classification prompt
        
        Returns:
            Dict with document_type, confidence, and reasoning
        """
        max_retries = settings.OCR_MAX_RETRIES
        retry_delay = settings.OCR_RETRY_DELAY
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Classifying document type (attempt {attempt})...")
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                
                # Encode and get correct media type based on actual file content
                base64_image, media_type = self._encode_image_to_base64(image_path)
                
                doc_content_type = "document" if media_type == "application/pdf" else "image"
                
                # Classification prompt - 4 categories only
                classification_prompt = '''Analyze this document image and classify it into EXACTLY ONE of these 4 categories based on its content, structure, and layout:

1. **SPA** (Sales, Purchases, Contract): 
   - Sales agreements, purchase agreements, sales contracts, purchase contracts
   - Sales & Purchase Agreement (SPA) documents
   - Any contract or agreement related to sales or purchases of property, goods, or services
   - Broker agreements, property management contracts, tenancy contracts, renewal contracts

2. **Invoices** (Payment invoices):
   - Bills requesting payment for goods or services provided
   - Invoices with itemized charges
   - Service invoices, product invoices

3. **ID** (ID card or passport):
   - Identification documents: national ID cards, passports, driver's licenses
   - Government-issued identification documents
   - Any official identification document

4. **Proof of Payment** (Bank receipt or transfer):
   - Bank receipts, payment receipts, payment confirmations
   - Bank transfer documents, wire transfer confirmations
   - Payment vouchers, payment proof documents
   - Any document proving that a payment was made

Consider these indicators:
- Document headers and titles
- Field labels and structure
- Document date should be the date of the document, not the current date
- Presence of signatures or authorization
- Payment/amount sections
- Terms and conditions sections
- Document layout and formatting
- Content and context

IMPORTANT: You must classify into EXACTLY ONE of these 4 categories: SPA, Invoices, ID, or Proof of Payment. Do not use any other category names.

Return your classification in JSON format:
{
    "document_type": "SPA",
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this classification was chosen"
}

The document_type must be exactly one of: "SPA", "Invoices", "ID", or "Proof of Payment". Be specific and accurate. If uncertain, use lower confidence scores.'''
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": classification_prompt
                            },
                            {
                                "type": doc_content_type,
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
                
                # Make Anthropic API call (reduced tokens for faster response)
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=256,  # Reduced from 512 for faster classification
                    messages=messages
                )
                
                # Parse response
                classification_result = response.content[0].text
                logger.info(f"Classification result received: {classification_result[:200]}")
                
                # Extract JSON from response
                classification_data = extract_json_from_text(classification_result)
                if classification_data:
                    return {
                        'document_type': classification_data.get('document_type', 'Other'),
                        'confidence': float(classification_data.get('confidence', 0.5)),
                        'reasoning': classification_data.get('reasoning', '')
                    }
                else:
                    # Fallback: try to extract document type from text
                    doc_type_match = re.search(r'"document_type":\s*"([^"]+)"', classification_result, re.IGNORECASE)
                    if doc_type_match:
                        return {
                            'document_type': doc_type_match.group(1),
                            'confidence': 0.7,
                            'reasoning': 'Extracted from response text'
                        }
                    return {
                        'document_type': 'Other',
                        'confidence': 0.5,
                        'reasoning': 'Could not parse classification response'
                    }
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Classification attempt {attempt} failed: {error_message}")

                model_hint = detect_model_not_found_error(error_message, self.model)
                if model_hint:
                    raise Exception(f"OCR_MODEL_NOT_FOUND: {model_hint}") from e
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.warning("Classification failed, using default 'Other'")
                    return {
                        'document_type': 'Other',
                        'confidence': 0.0,
                        'reasoning': f'Classification failed: {error_message}'
                    }
    
    def _extract_general_document_data(self, image_path: str, document_type: str) -> str:
        """
        Extract general document data using flexible prompt based on document type
        
        Args:
            image_path: Path to the document file
            document_type: Classified document type (Invoice, Receipt, Contract, etc.)
            
        Returns:
            JSON string with extracted data
        """
        max_retries = settings.OCR_MAX_RETRIES
        retry_delay = settings.OCR_RETRY_DELAY
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Extracting general document data (attempt {attempt}) for type: {document_type}")
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                
                # Encode and get correct media type based on actual file content
                base64_image, media_type = self._encode_image_to_base64(image_path)
                
                doc_content_type = "document" if media_type == "application/pdf" else "image"
                doc_or_image_text = "document" if media_type == "application/pdf" else "image"
                
                # Use ID-specific prompt for ID documents, otherwise use general prompt
                doc_type_normalized = document_type.upper().strip() if document_type else ''
                
                if doc_type_normalized == 'ID':
                    logger.info("Using ID-specific extraction prompt for ID document")
                    # ID-specific extraction prompt
                    extraction_prompt = f'''Extract all key information from this ID document (identification card, passport, or driver's license).

Extract the following fields (include all that are present, omit those that are not):

1. **Document number/ID**: Any unique identifier, ID number, passport number, license number, etc.
2. **Document Date** (CRITICAL): The date when the ID was issued. Look for fields like "Date of Issue", "Issue Date", "Date Issued", "Valid From", etc. Extract this as "document_date" in the JSON.
3. **Expiration Date**: The date when the ID expires. Look for "Expiry Date", "Valid Until", "Date of Expiry", etc.
4. **Client Full Name** (CRITICAL for property files): The COMPLETE full name of the person shown on this ID document. This is the most important field. Look for:
   - "Name", "Full Name", "Holder Name", "Cardholder Name", "Name of Holder"
   - "Given Name" + "Surname" (combine them into full name)
   - "First Name" + "Last Name" (combine them into full name)
   - "Name in Full", "Complete Name", "Full Name of Holder"
   - Any field that contains the person's complete name
   - Extract the COMPLETE full name (first name + middle name(s) + last name) as "client_full_name" in the JSON.
5. **Date of Birth**: The person's date of birth if present
6. **Nationality**: The nationality or country of origin
7. **Address**: The address shown on the ID if present
8. **ID Type**: The type of ID (National ID, Passport, Driver's License, etc.)

Extraction Rules:
- Extract text EXACTLY as shown (preserve formatting, spaces, hyphens)
- For dates, preserve original format
- If a field is not found, omit it from JSON (don't use null or empty strings)
- Extract all relevant information comprehensively
- **MOST IMPORTANT**: For client_full_name, extract the COMPLETE full name of the person from the ID card. This is critical for matching documents. Look carefully for the name field - it may be labeled as "Name", "Full Name", "Holder Name", or may be in a "Given Name" + "Surname" format. Combine separate name fields into one complete full name.
- **CRITICAL**: ID documents (passports, driver's licenses, national IDs) do NOT contain property information. Do NOT extract property_reference, property_name, or property_address from ID documents. Only extract client_full_name for matching purposes. Property information is only found in SPA, Invoice, and Proof of Payment documents, NOT in ID documents.

Return in JSON format with all extracted fields:
{{
    "document_number": "123456789",
    "document_id": "P1234567",
    "document_date": "2020-01-15",
    "expiration_date": "2030-01-15",
    "client_full_name": "Ahmed Al Maktoum",
    "date_of_birth": "1985-05-20",
    "nationality": "UAE",
    "address": "123 Main Street, Dubai",
    "id_type": "National ID",
    "additional_info": "Any other relevant information"
}}

Be thorough and accurate. The client_full_name field is CRITICAL - make sure to extract the complete full name of the person from the ID card.'''
                else:
                    # General extraction prompt for other document types
                    logger.info(f"Using general extraction prompt for document type: {document_type}")
                    extraction_prompt = f'''Extract all key information from this {document_type.lower()} document. 

Extract the following fields (include all that are present, omit those that are not):

1. **Document number/ID**: Any unique identifier, invoice number, receipt number, contract number, etc.
2. **Document Date** (CRITICAL): The primary date of the document - this is the date when the document was issued, created, or signed. Look for fields like "Date", "Document Date", "Issue Date", "Created Date", "Date of Issue", etc. This should be the main date of the document, not the current date. Extract this as "document_date" in the JSON.
3. **Other Date(s)**: Due date, effective date, expiration date, delivery date - extract all other dates found
4. **Amount/Total**: Total amount, subtotal, tax, fees, discounts - include currency (USD, EUR, AED, etc.)
5. **Client Full Name** (CRITICAL for property files): The full name of the client/buyer/tenant. This is the person who is buying or renting the property. Look for fields like "Buyer Name", "Client Name", "Tenant Name", "Purchaser Name", "Party Name", etc. Extract the COMPLETE full name (first name + last name) as "client_full_name" in the JSON.
6. **Property Reference** (CRITICAL for property files): The property unit number, listing code, or internal property identifier. Look for fields like "Unit Number", "Property Reference", "Listing Code", "Property ID", "Unit ID", "Apartment Number", "Villa Number", etc. Extract this as "property_reference" in the JSON.
7. **Transaction Type** (for SPA documents): Determine if this is a "BUY", "RENT", or "SELL" transaction. Look for keywords like "purchase", "sale", "rent", "lease", "tenancy", etc. Extract as "transaction_type" in the JSON (should be "BUY", "RENT", or "SELL").
8. **Parties involved**: 
   - Buyer, seller, client, customer, vendor, supplier
   - Names, addresses, contact information
   - For contracts: parties to the agreement
   - For real estate: buyer, seller, landlord, tenant, agent
9. **Items/services listed**: 
   - Line items, products, services
   - Quantities, descriptions, unit prices
   - For contracts: services or deliverables
   - For real estate: property details, address, square footage
10. **Terms and conditions**: 
   - Payment terms, delivery terms
   - Contract terms, conditions, clauses
   - Legal terms, warranties, guarantees
11. **Signature/authorization info**:
   - Signatures present (yes/no)
   - Signatory names and titles
   - Authorization stamps or seals
   - Notary information if present

Extraction Rules:
- Extract text EXACTLY as shown (preserve formatting, spaces, hyphens)
- For dates, preserve original format
- For amounts, include currency symbol or code
- For multi-page PDFs, extract from all pages
- If a field is not found, omit it from JSON (don't use null or empty strings)
- Extract all relevant information comprehensively
- For client_full_name: Extract the complete full name of the buyer/client/tenant (the person who will own or rent the property)
- For property_reference: Extract unit number, listing code, or any property identifier mentioned in the document
- For transaction_type: Only extract for SPA documents, should be "BUY", "RENT", or "SELL"

Return in JSON format with all extracted fields:
{{
    "document_number": "INV-2025-001",
    "document_id": "DOC-12345",
    "document_date": "2025-01-15",
    "issue_date": "2025-01-15",
    "due_date": "2025-02-15",
    "total_amount": "1500.00",
    "currency": "USD",
    "subtotal": "1300.00",
    "tax": "200.00",
    "client_full_name": "Ahmed Al Maktoum",
    "property_reference": "Unit 101",
    "transaction_type": "BUY",
    "buyer": {{
        "name": "Company Name",
        "address": "123 Main St",
        "contact": "contact@company.com"
    }},
    "seller": {{
        "name": "Vendor Name",
        "address": "456 Vendor Ave"
    }},
    "items": [
        {{
            "description": "Product/Service Name",
            "quantity": "2",
            "unit_price": "650.00",
            "total": "1300.00"
        }}
    ],
    "terms": "Net 30 days, payment due upon receipt",
    "signatures": {{
        "present": true,
        "signatories": ["John Doe", "Jane Smith"],
        "titles": ["Manager", "Director"]
    }},
    "additional_info": "Any other relevant information"
}}

Adapt the extraction based on the document type ({document_type}). Be thorough and accurate.'''
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": extraction_prompt
                            },
                            {
                                "type": doc_content_type,
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
                
                # Make Anthropic API call (optimized tokens for faster response)
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1536,  # Reduced from 2048 for faster extraction
                    messages=messages
                )
                
                # Parse response
                extraction_result = response.content[0].text
                if doc_type_normalized == 'ID':
                    logger.info(f"ID document extraction result received (length: {len(extraction_result)} chars)")
                else:
                    logger.info(f"General extraction result received (length: {len(extraction_result)} chars)")
                return extraction_result
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"General extraction attempt {attempt} failed: {error_message}")

                model_hint = detect_model_not_found_error(error_message, self.model)
                if model_hint:
                    raise Exception(f"OCR_MODEL_NOT_FOUND: {model_hint}") from e
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"EXTRACTION_FAILED: {error_message}")
    
    def _extract_transaction_data(self, image_path: str) -> str:
        """Extract transaction data using Anthropic OCR"""
        max_retries = settings.OCR_MAX_RETRIES
        retry_delay = settings.OCR_RETRY_DELAY
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempting Anthropic OCR (attempt {attempt})...")
                logger.info(f"Image path: {image_path}")
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file does not exist: {image_path}")
                
                # Encode and get correct media type based on actual file content
                base64_image, media_type = self._encode_image_to_base64(image_path)
                
                logger.info(f"Media type: {media_type}")
                
                # Build content based on file type
                if media_type == "application/pdf":
                    doc_content_type = "document"
                    doc_or_image_text = "document/voucher"
                else:
                    doc_content_type = "image"
                    doc_or_image_text = "voucher image"
                
                # Prepare messages for Anthropic API
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": '''OCR and extract important information from voucher/invoice images or PDF documents accurately.

You need to extract:
1. **Document No** (e.g., "MPU01-85285") - Extract EXACTLY as shown without any spaces and hyphens
2. **Document Date** (e.g., "02/06/2025") - Extract EXACTLY as shown without any spaces and hyphens
3. **Branch ID** (extracted from Document No) - Extract EXACTLY as shown without any spaces and hyphens
4. **Invoice Amount USD** (e.g., "15000.00" or "15,000.00") - Extract USD amount if present
5. **Invoice Amount AED/DHS** (e.g., "55000.00") - Extract AED or DHS amount if present
6. **Gold Weight** (e.g., "20000.000" grams) - Extract weight in grams (CRITICAL for matching)
7. **Purity** (e.g., "1.000", "0.995", "22K", "24K") - Extract purity value (CRITICAL for matching)
8. **Discount Rate** (e.g., "5.0" or "-10.50$/OZ") - Extract discount rate if available

Extraction Rules:
- Extract the COMPLETE Document No preserving all spaces and hyphens
- DO NOT modify, sanitize, or change the format
- Keep it exactly: "MPU01-85285" not "MPU01_-_85285"
- For PDFs: Extract from the first page if multi-page document
- Extract BOTH currencies if available (USD and AED/DHS)
- Gold Weight should be in grams (remove commas: "20,000.00" → "20000.00")
- Purity can be decimal (1.000, 0.995) or karat (22K, 24K)
- If a field is not found, omit it from JSON

Return in JSON format:
{
    "document_no": "MPU01-85285",
    "category_type": "MPU",
    "branch_id": "01",
    "document_date": "02/06/2025",
    "filename": "MPU01-85285",
    "invoice_amount_usd": "2154100.49",
    "invoice_amount_aed": "7914165.20",
    "gold_weight": "20000.000",
    "purity": "1.000",
    "discount_rate": "-10.50$/OZ"
}'''
                            },
                            {
                                "type": doc_content_type,
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": f"I understand perfectly. I will:\n\n1. Extract the **COMPLETE Document No** exactly as displayed (e.g., 'MPU01-85285')\n2. without modification\n3. Use this complete Document No as the filename\n4. Extract Category Type ('MPU') for folder organization\n5. Extract Branch ID ('01') for sub-folder structure\n6. Extract Document Date in original format\n\nReady to process your {doc_or_image_text}!"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Now process this {doc_or_image_text} and return the JSON response:"
                            },
                            {
                                "type": doc_content_type,
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
                
                # Make Anthropic API call
                response = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=messages
                )
                
                # Parse response
                ocr_result = response.content[0].text
                logger.info(f"Anthropic OCR result received")
                return ocr_result
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"OCR attempt {attempt} failed: {error_message}")

                model_hint = detect_model_not_found_error(error_message, self.model)
                if model_hint:
                    raise Exception(f"OCR_MODEL_NOT_FOUND: {model_hint}") from e
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"OCR_FAILED: {error_message}")
    
    def _convert_image_to_pdf(self, image_path: str) -> Optional[str]:
        """Convert image to PDF format using pure Python"""
        try:
            # Create PDF file path
            pdf_path = image_path.rsplit('.', 1)[0] + '_0001.pdf'
            
            logger.info(f"Converting {image_path} to PDF...")
            
            # Read image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            if len(image_bytes) == 0:
                logger.error("Image file is empty")
                return None
            
            # Get image dimensions
            if image_bytes[0:2] == b'\xff\xd8':  # JPEG
                try:
                    width, height = self._get_jpeg_dimensions(image_bytes)
                    filter_type = '/DCTDecode'
                    colorspace = '/DeviceRGB'
                    bpc = 8
                    image_data = image_bytes
                    logger.info(f"JPEG image: {width}x{height}")
                except Exception as jpeg_error:
                    logger.error(f"Failed to parse JPEG: {jpeg_error}")
                    return None
                
            elif image_bytes[0:8] == b'\x89PNG\r\n\x1a\n':  # PNG
                width = struct.unpack('>I', image_bytes[16:20])[0]
                height = struct.unpack('>I', image_bytes[20:24])[0]
                bit_depth = image_bytes[24]
                color_type = image_bytes[25]
                
                logger.info(f"PNG image: {width}x{height}, bit_depth={bit_depth}, color_type={color_type}")
                
                # Extract IDAT chunks
                image_data = self._extract_png_idat(image_bytes)
                if not image_data:
                    raise ValueError("Failed to extract PNG image data")
                
                # Determine colorspace
                if color_type == 0:
                    colorspace = '/DeviceGray'
                    components = 1
                elif color_type == 2:
                    colorspace = '/DeviceRGB'
                    components = 3
                elif color_type == 3:
                    colorspace = '/DeviceRGB'
                    components = 3
                elif color_type == 4:
                    colorspace = '/DeviceGray'
                    components = 1
                elif color_type == 6:
                    colorspace = '/DeviceRGB'
                    components = 3
                else:
                    colorspace = '/DeviceRGB'
                    components = 3
                
                filter_type = '/FlateDecode'
                bpc = bit_depth
            else:
                raise ValueError("Unsupported image format (only JPEG and PNG supported)")
            
            # Create minimal PDF
            pdf_content = []
            pdf_content.append(b'%PDF-1.4\n%\xE2\xE3\xCF\xD3\n')
            
            # Catalog
            obj1_start = sum(len(x) for x in pdf_content)
            pdf_content.append(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n')
            
            # Pages
            obj2_start = sum(len(x) for x in pdf_content)
            pdf_content.append(b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n')
            
            # Page
            obj3_start = sum(len(x) for x in pdf_content)
            pdf_content.append(b'3 0 obj\n')
            pdf_content.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] '.encode())
            pdf_content.append(b'/Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>\nendobj\n')
            
            # Content stream
            obj4_start = sum(len(x) for x in pdf_content)
            stream = f'q\n{width} 0 0 {height} 0 0 cm\n/Im1 Do\nQ\n'.encode()
            pdf_content.append(f'4 0 obj\n<< /Length {len(stream)} >>\nstream\n'.encode())
            pdf_content.append(stream)
            pdf_content.append(b'\nendstream\nendobj\n')
            
            # Image
            obj5_start = sum(len(x) for x in pdf_content)
            pdf_content.append(b'5 0 obj\n')
            pdf_content.append(f'<< /Type /XObject /Subtype /Image /Width {width} /Height {height} '.encode())
            pdf_content.append(f'/ColorSpace {colorspace} /BitsPerComponent {bpc} '.encode())
            
            if filter_type == '/FlateDecode':
                pdf_content.append(b'/Filter /FlateDecode ')
                pdf_content.append(b'/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns ')
                pdf_content.append(f'{width} >> '.encode())
            else:
                pdf_content.append(f'/Filter {filter_type} '.encode())
            
            pdf_content.append(f'/Length {len(image_data)} >>\nstream\n'.encode())
            pdf_content.append(image_data)
            pdf_content.append(b'\nendstream\nendobj\n')
            
            # xref and trailer
            xref_start = sum(len(x) for x in pdf_content)
            pdf_content.append(b'xref\n0 6\n0000000000 65535 f \n')
            pdf_content.append(f'{obj1_start:010d} 00000 n \n'.encode())
            pdf_content.append(f'{obj2_start:010d} 00000 n \n'.encode())
            pdf_content.append(f'{obj3_start:010d} 00000 n \n'.encode())
            pdf_content.append(f'{obj4_start:010d} 00000 n \n'.encode())
            pdf_content.append(f'{obj5_start:010d} 00000 n \n'.encode())
            pdf_content.append(b'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n')
            pdf_content.append(f'{xref_start}\n'.encode())
            pdf_content.append(b'%%EOF\n')
            
            # Write PDF
            with open(pdf_path, 'wb') as f:
                for chunk in pdf_content:
                    f.write(chunk)
            
            if os.path.exists(pdf_path):
                pdf_size = os.path.getsize(pdf_path)
                logger.info(f"Successfully converted to PDF: {pdf_path} ({pdf_size} bytes)")
                return pdf_path
            else:
                logger.error("PDF file was not created")
                return None
                
        except Exception as e:
            logger.error(f"Error converting to PDF: {e}")
            return None
    
    def _extract_png_idat(self, png_bytes: bytes) -> Optional[bytes]:
        """Extract and combine all IDAT chunks from PNG"""
        if png_bytes[0:8] != b'\x89PNG\r\n\x1a\n':
            return None
        
        idat_data = b''
        pos = 8
        
        while pos < len(png_bytes):
            if pos + 8 > len(png_bytes):
                break
            
            chunk_length = struct.unpack('>I', png_bytes[pos:pos+4])[0]
            chunk_type = png_bytes[pos+4:pos+8]
            
            if chunk_type == b'IDAT':
                chunk_data = png_bytes[pos+8:pos+8+chunk_length]
                idat_data += chunk_data
            
            if chunk_type == b'IEND':
                break
            
            pos += 4 + 4 + chunk_length + 4
        
        return idat_data if idat_data else None
    
    def _get_jpeg_dimensions(self, jpeg_bytes: bytes) -> tuple[int, int]:
        """Extract width and height from JPEG"""
        try:
            i = 0
            if jpeg_bytes[0:2] != b'\xff\xd8':
                raise ValueError("Not a valid JPEG file")
            
            i = 2
            
            while i < len(jpeg_bytes) - 10:
                if jpeg_bytes[i] != 0xFF:
                    i += 1
                    continue
                
                while i < len(jpeg_bytes) and jpeg_bytes[i] == 0xFF:
                    i += 1
                
                if i >= len(jpeg_bytes):
                    break
                    
                marker = jpeg_bytes[i]
                i += 1
                
                if 0xC0 <= marker <= 0xCF and marker not in [0xC4, 0xC8, 0xCC]:
                    if i + 5 < len(jpeg_bytes):
                        i += 3
                        height = (jpeg_bytes[i] << 8) | jpeg_bytes[i+1]
                        width = (jpeg_bytes[i+2] << 8) | jpeg_bytes[i+3]
                        logger.info(f"JPEG dimensions found: {width}x{height}")
                        return width, height
                
                if i + 1 < len(jpeg_bytes):
                    length = (jpeg_bytes[i] << 8) | jpeg_bytes[i+1]
                    i += length
                else:
                    break
            
            raise ValueError("Could not find JPEG dimensions")
            
        except Exception as e:
            logger.error(f"Error parsing JPEG dimensions: {e}")
            raise ValueError(f"Could not find JPEG dimensions: {e}")
    
    # ZERO-SHOT VALIDATION TEMPORARILY DISABLED - Code preserved for future use
    # def _validate_document_image(self, image_path: str) -> Dict[str, Any]:
    #     """
    #     Validate if an image is a valid document using zero-shot classification.
    #     
    #     Uses the shared singleton CLIP classifier (model loads only once).
    #     
    #     Args:
    #         image_path: Path to the image file
    #         
    #     Returns:
    #         Dict with validation results:
    #         - validation_status: "valid" | "failed" | "need_review"
    #         - validation_confidence: float score from classifier
    #         - validation_label: string label from classifier
    #         - error: optional error message
    #     """
    #     # Get the shared singleton classifier (loads only once on first call)
    #     clip_classifier = _get_clip_classifier()
    #     
    #     # Check if classifier is available
    #     if not clip_classifier:
    #         logger.warning("CLIP classifier not available - SKIPPING validation and allowing document to proceed")
    #         logger.warning("This may happen if HuggingFace models cannot be downloaded (network issues)")
    #         return {
    #             'validation_status': 'skipped',  # Skip validation if classifier unavailable
    #             'validation_confidence': 1.0,  # High confidence to allow processing
    #             'validation_label': 'classifier_unavailable',
    #             'error': 'Image validation skipped - classifier not loaded'
    #         }
    #     
    #     # Check if PIL is available for image loading
    #     if not PIL_AVAILABLE:
    #         logger.warning("PIL not available - SKIPPING validation and allowing document to proceed")
    #         return {
    #             'validation_status': 'skipped',  # Skip validation if PIL unavailable
    #             'validation_confidence': 1.0,  # High confidence to allow processing
    #             'validation_label': 'pil_unavailable',
    #             'error': 'Image validation skipped - PIL not installed'
    #         }
    #     
    #     try:
    #         logger.info(f"Validating image with zero-shot classification: {image_path}")
    #         
    #         # Load image using PIL
    #         try:
    #             img = Image.open(image_path)
    #             # Convert RGBA to RGB if needed
    #             if img.mode in ['RGBA', 'LA', 'P']:
    #                 rgb_img = Image.new('RGB', img.size, (255, 255, 255))
    #                 if img.mode == 'P':
    #                     img = img.convert('RGBA')
    #                 rgb_img.paste(img, mask=img.split()[-1] if img.mode in ['RGBA', 'LA'] else None)
    #                 img = rgb_img
    #             elif img.mode != 'RGB':
    #                 img = img.convert('RGB')
    #         except Exception as e:
    #             logger.error(f"Failed to load image for validation: {e}")
    #             return {
    #                 'validation_status': 'failed',  # Fail-safe: reject if can't load image
    #                 'validation_confidence': 0.0,
    #                 'validation_label': 'load_error',
    #                 'error': f"Failed to load image for validation: {str(e)}"
    #             }
    #         
    #         # Run zero-shot classification using the shared singleton classifier
    #         try:
    #             scores = clip_classifier(img, candidate_labels=self.VALIDATION_LABELS)
    #             
    #             # Get the top result
    #             if not scores or len(scores) == 0:
    #                 logger.error("Classifier returned empty results - REJECTING image")
    #                 return {
    #                     'validation_status': 'failed',  # Fail-safe: reject if no results
    #                     'validation_confidence': 0.0,
    #                     'validation_label': 'empty_results',
    #                     'error': 'Classifier returned empty results'
    #                 }
    #             
    #             top_result = scores[0]
    #             label = top_result.get('label', '').strip()
    #             score = float(top_result.get('score', 0.0))
    #             
    #             logger.info(f"Validation result: label='{label}', score={score:.4f}")
    #             logger.info(f"All scores: {scores}")
    #             
    #             # Determine validation status based on confidence thresholds
    #             # Check if label matches "Valid document" (case-insensitive)
    #             is_valid_document_label = label.lower() == "valid document" or label.lower() == "valid_document"
    #             
    #             # Use the configured confidence threshold (0.90 = 90%)
    #             confidence_threshold = self.CONFIDENCE_THRESHOLD
    #             
    #             if score < 0.40:
    #                 # Image is NOT a document
    #                 validation_status = 'failed'
    #                 logger.warning(f"Image REJECTED: Confidence {score:.4f} < 0.40 (NOT a document)")
    #             elif score < confidence_threshold:
    #                 # Image might be a document but unclear - needs review
    #                 validation_status = 'need_review'
    #                 logger.warning(f"Image needs REVIEW: Confidence {score:.4f} between 0.40-{confidence_threshold:.2f} (unclear - requires {confidence_threshold*100:.0f}% for acceptance)")
    #             else:
    #                 # High confidence (>= threshold) - check if it's actually "Valid document"
    #                 if is_valid_document_label:
    #                     validation_status = 'valid'
    #                     logger.info(f"Image ACCEPTED: Confidence {score:.4f} >= {confidence_threshold:.2f}, label='{label}' (Valid document)")
    #                 else:
    #                     # High confidence but wrong label (e.g., "other")
    #                     validation_status = 'failed'
    #                     logger.warning(f"Image REJECTED: High confidence {score:.4f} but wrong label '{label}' (not 'Valid document')")
    #             
    #             return {
    #                 'validation_status': validation_status,
    #                 'validation_confidence': score,
    #                 'validation_label': label
    #             }
    #             
    #         except Exception as e:
    #             logger.error(f"Error during zero-shot classification: {e}")
    #             logger.error("REJECTING image due to classification error (fail-safe mode)")
    #             return {
    #                 'validation_status': 'failed',  # Fail-safe: reject on classification error
    #                 'validation_confidence': 0.0,
    #                 'validation_label': 'classification_error',
    #                 'error': f"Classification error: {str(e)}"
    #             }
    #             
    #     except Exception as e:
    #         logger.error(f"Unexpected error in image validation: {e}")
    #         logger.error("REJECTING image due to unexpected error (fail-safe mode)")
    #         return {
    #             'validation_status': 'failed',  # Fail-safe: reject on unexpected error
    #             'validation_confidence': 0.0,
    #             'validation_label': 'unexpected_error',
    #             'error': f"Unexpected error: {str(e)}"
    #         }
    
    def process_document(self, image_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document - classify type first, then extract data using OCR
        
        Args:
            image_path: Path to the document file
            original_filename: Original filename for reference
            
        Returns:
            Dict with processing results including classification and extracted data
        """
        try:
            logger.info(f"Processing document: {image_path}")
            
            file_ext = os.path.splitext(image_path)[1].lower()
            is_pdf = file_ext == '.pdf'
            
            # ZERO-SHOT VALIDATION TEMPORARILY DISABLED - Code preserved for future use
            # Step 0: Zero-shot image validation (skip for PDFs)
            # validation_result = {
            #     'validation_status': 'valid',
            #     'validation_confidence': 1.0,
            #     'validation_label': 'skipped'
            # }
            # 
            # if not is_pdf:
            #     # Only validate image files, skip PDFs
            #     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            #     if any(file_ext == ext for ext in image_extensions):
            #         logger.info(f"🔍 Running zero-shot image validation for: {original_filename or image_path}")
            #         logger.info(f"   File extension: {file_ext}")
            #         validation_result = self._validate_document_image(image_path)
            #         validation_status = validation_result.get('validation_status', 'valid')
            #         validation_confidence = validation_result.get('validation_confidence', 1.0)
            #         validation_label = validation_result.get('validation_label', '')
            #         
            #         logger.info(f"✅ Validation completed: status={validation_status}, confidence={validation_confidence:.4f}, label={validation_label}")
            #         
            #         # Handle skipped validation (classifier unavailable)
            #         if validation_status == 'skipped':
            #             logger.warning(f"Image validation skipped: {validation_label} - proceeding with document processing")
            #             # Continue processing - don't return early
            #         
            #         # Handle failed validation (< 0.40)
            #         elif validation_status == 'failed':
            #             logger.warning(f"Image validation failed: Image is NOT a document (confidence: {validation_confidence:.4f})")
            #             return {
            #                 'success': False,
            #                 'error': f'Image is not a valid document (confidence: {validation_confidence:.4f})',
            #                 'validation_status': 'failed',
            #                 'validation_confidence': validation_confidence,
            #                 'validation_label': validation_label,
            #                 'document_type': 'Invalid',
            #                 'classification': 'UNKNOWN',
            #                 'ocr_text': None,
            #                 'extracted_data': {}
            #             }
            #         
            #         # Handle need_review status (0.40 - 0.90)
            #         elif validation_status == 'need_review':
            #             logger.warning(f"Image validation: Needs human review (confidence: {validation_confidence:.4f})")
            #             return {
            #                 'success': False,
            #                 'error': f'Image might be a document but requires review (confidence: {validation_confidence:.4f})',
            #                 'validation_status': 'need_review',
            #                 'validation_confidence': validation_confidence,
            #                 'validation_label': validation_label,
            #                 'document_type': 'Unclear',
            #                 'classification': 'UNKNOWN',
            #                 'ocr_text': None,
            #                 'extracted_data': {}
            #             }
            #         
            #         # If validation passed (>= 0.90 and "Valid document"), continue processing
            #         else:
            #             logger.info("✅ Image validation PASSED - proceeding with OCR and classification")
            #     else:
            #         logger.info(f"Skipping validation for file type: {file_ext}")
            # else:
            #     logger.info("Skipping validation for PDF file")
            
            # Skip validation - always proceed with processing
            validation_result = {
                'validation_status': 'skipped',
                'validation_confidence': 1.0,
                'validation_label': 'validation_disabled'
            }
            logger.info("Zero-shot validation disabled - proceeding with document processing")
            
            if file_ext == '.pdf':
                logger.info("Processing PDF directly via Claude API")
            
            # Step 1: Classify document type (can be skipped for faster processing)
            if hasattr(settings, 'SKIP_CLASSIFICATION') and settings.SKIP_CLASSIFICATION:
                logger.info("Skipping classification step for faster processing")
                document_type = 'Other'
                classification_confidence = 0.5
                classification_reasoning = 'Classification skipped for performance'
            else:
                logger.info("Step 1: Classifying document type...")
                classification_result = self._classify_document_type(image_path)
                document_type = classification_result.get('document_type', 'Other')
                classification_confidence = classification_result.get('confidence', 0.0)
                classification_reasoning = classification_result.get('reasoning', '')
                logger.info(f"Document classified as: {document_type} (confidence: {classification_confidence:.2f})")
            
            # Step 2: Extract data based on document type
            logger.info(f"Step 2: Extracting data for {document_type}...")
            
            # Initialize result structure
            result = {
                'success': True,
                'document_type': document_type,
                'classification': 'UNKNOWN',
                'classification_confidence': classification_confidence,
                'classification_reasoning': classification_reasoning,
                'document_no': None,
                'document_date': None,
                'branch_id': None,
                'ocr_text': None,
                'extracted_data': {},
                'confidence': classification_confidence,
                'method': 'anthropic_ocr',
                'organized_path': None,
                'complete_filename': None,
                'invoice_amount_usd': None,
                'invoice_amount_aed': None,
                'gold_weight': None,
                'purity': None,
                'discount_rate': None,
                'is_valid_voucher': False,
                'needs_attachment': False,
                # Validation metadata
                'validation_status': validation_result.get('validation_status', 'valid'),
                'validation_confidence': validation_result.get('validation_confidence', 1.0),
                'validation_label': validation_result.get('validation_label', '')
            }
            
            # Use voucher-specific extraction for vouchers, general extraction for others
            if document_type.lower() == 'voucher':
                logger.info("Using voucher-specific extraction method")
                transaction_data = self._extract_transaction_data(image_path)
                result['ocr_text'] = transaction_data
                extraction_method = 'voucher_specific'
            else:
                logger.info(f"Using general extraction method for {document_type}")
                transaction_data = self._extract_general_document_data(image_path, document_type)
                result['ocr_text'] = transaction_data
                extraction_method = 'general'
            
            result['extraction_method'] = extraction_method
            
            # Try to parse JSON response
            try:
                json_data = extract_json_from_text(transaction_data)
                if json_data:
                    result['extracted_data'] = json_data
                    
                    # Handle voucher-specific extraction
                    if extraction_method == 'voucher_specific':
                        # Extract data from JSON (all fields optional)
                        doc_no = json_data.get('document_no', '')
                        result['document_no'] = doc_no.strip() if isinstance(doc_no, str) else str(doc_no) if doc_no else ''
                        
                        doc_date = json_data.get('document_date', '')
                        result['document_date'] = doc_date.strip() if isinstance(doc_date, str) else str(doc_date) if doc_date else ''
                        
                        branch = json_data.get('branch_id', '')
                        result['branch_id'] = branch.strip() if isinstance(branch, str) else str(branch) if branch else ''
                        
                        raw_classification = json_data.get('category_type', '')
                        raw_classification = raw_classification.strip() if isinstance(raw_classification, str) else str(raw_classification) if raw_classification else ''
                        
                        # Extract classification from document_no if missing
                        if not raw_classification and result['document_no']:
                            extracted_prefix = self._extract_document_no_prefix(result['document_no'])
                            if extracted_prefix:
                                raw_classification = extracted_prefix
                                logger.info(f"Extracted classification from document_no: '{raw_classification}'")
                        
                        if raw_classification and raw_classification in self.voucher_types:
                            result['classification'] = raw_classification
                            result['is_valid_voucher'] = True
                            logger.info(f"Valid voucher type: '{raw_classification}'")
                        else:
                            result['classification'] = raw_classification if raw_classification else 'UNKNOWN'
                            result['is_valid_voucher'] = False
                            logger.warning(f"Invalid voucher type '{raw_classification}' - will treat as attachment")
                        
                        filename = json_data.get('filename', '')
                        result['complete_filename'] = filename.strip() if isinstance(filename, str) else str(filename) if filename else ''
                        
                        # Extract financial fields (all optional)
                        try:
                            usd_amount = json_data.get('invoice_amount_usd', '')
                            if usd_amount:
                                usd_str = str(usd_amount).strip().replace(',', '')
                                result['invoice_amount_usd'] = usd_str if usd_str else None
                            else:
                                result['invoice_amount_usd'] = None
                        except:
                            result['invoice_amount_usd'] = None
                        
                        try:
                            aed_amount = json_data.get('invoice_amount_aed', '')
                            if aed_amount:
                                aed_str = str(aed_amount).strip().replace(',', '')
                                result['invoice_amount_aed'] = aed_str if aed_str else None
                            else:
                                result['invoice_amount_aed'] = None
                        except:
                            result['invoice_amount_aed'] = None
                        
                        try:
                            weight = json_data.get('gold_weight', '')
                            if weight:
                                weight_str = str(weight).strip().replace(',', '')
                                result['gold_weight'] = weight_str if weight_str else None
                            else:
                                result['gold_weight'] = None
                        except:
                            result['gold_weight'] = None
                        
                        try:
                            purity = json_data.get('purity', '')
                            result['purity'] = str(purity).strip() if purity else None
                        except:
                            result['purity'] = None
                        
                        try:
                            discount = json_data.get('discount_rate', '')
                            result['discount_rate'] = str(discount).strip() if discount else None
                        except:
                            result['discount_rate'] = None
                        
                        logger.info(f"Extracted: Document No: {result.get('document_no', 'N/A')}, Date: {result.get('document_date', 'N/A')}, Branch: {result.get('branch_id', 'N/A')}, Category: {result['classification']}")
                    
                    # Handle general extraction
                    else:
                        # Extract common fields from general extraction (all optional)
                        result['document_no'] = (
                            json_data.get('document_number') or 
                            json_data.get('document_id') or 
                            json_data.get('document_no') or 
                            json_data.get('contract_number') or 
                            json_data.get('agreement_number') or 
                            json_data.get('reference_number') or
                            ''
                        )
                        if isinstance(result['document_no'], str):
                            result['document_no'] = result['document_no'].strip()
                        else:
                            result['document_no'] = str(result['document_no']) if result['document_no'] else ''
                        
                        # Extract date (prioritize document_date, then issue_date, then other date fields)
                        result['document_date'] = (
                            json_data.get('document_date') or 
                            json_data.get('issue_date') or 
                            json_data.get('date') or 
                            json_data.get('created_date') or
                            json_data.get('date_of_issue') or
                            json_data.get('effective_date') or
                            json_data.get('contract_date') or
                            ''
                        )
                        if isinstance(result['document_date'], str):
                            result['document_date'] = result['document_date'].strip()
                        else:
                            result['document_date'] = str(result['document_date']) if result['document_date'] else ''
                        
                        # Extract client_full_name (for property files)
                        client_full_name = json_data.get('client_full_name') or (json_data.get('buyer', {}) if isinstance(json_data.get('buyer'), dict) else {}).get('name') or ''
                        if isinstance(client_full_name, str):
                            result['client_full_name_extracted'] = client_full_name.strip()
                        else:
                            result['client_full_name_extracted'] = str(client_full_name).strip() if client_full_name else ''
                        
                        # Extract property_reference (for property files) - SKIP for ID documents
                        # ID documents don't contain property information, only client name
                        doc_type_normalized = document_type.upper().strip() if document_type else ''
                        if doc_type_normalized != 'ID':
                            property_reference = json_data.get('property_reference') or json_data.get('property_id') or json_data.get('unit_number') or ''
                            if isinstance(property_reference, str):
                                result['property_reference_extracted'] = property_reference.strip()
                            else:
                                result['property_reference_extracted'] = str(property_reference).strip() if property_reference else ''
                        else:
                            # ID documents don't contain property information
                            result['property_reference_extracted'] = None
                            logger.info("ID document - skipping property_reference extraction (ID documents don't contain property info)")
                        
                        # Extract transaction_type (for SPA documents)
                        transaction_type = json_data.get('transaction_type', '').upper()
                        if transaction_type in ['BUY', 'RENT', 'SELL']:
                            result['transaction_type'] = transaction_type
                        else:
                            result['transaction_type'] = None
                        
                        # Extract amount and currency (optional)
                        total_amount = json_data.get('total_amount') or json_data.get('amount') or json_data.get('contract_value') or json_data.get('price') or ''
                        currency = json_data.get('currency', '')
                        if total_amount:
                            # Clean amount string
                            amount_str = str(total_amount).replace(',', '').strip()
                            if currency and isinstance(currency, str):
                                if currency.upper() == 'USD':
                                    result['invoice_amount_usd'] = amount_str
                                elif currency.upper() in ['AED', 'DHS']:
                                    result['invoice_amount_aed'] = amount_str
                            else:
                                # Store in invoice_amount_aed as default if no currency specified
                                result['invoice_amount_aed'] = amount_str
                        
                        # Set classification to document_type
                        result['classification'] = document_type
                        result['complete_filename'] = result['document_no'] or original_filename or 'document'
                        
                        # Store all extracted data
                        logger.info(f"Extracted general document data: Type={document_type}, Doc No={result.get('document_no', 'N/A')}, Date={result.get('document_date', 'N/A')}, Client={result.get('client_full_name_extracted', 'N/A')}, Property={result.get('property_reference_extracted', 'N/A')}")
                    
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"JSON parsing failed, falling back to regex: {e}")
                
                # Fallback to regex extraction
                document_no_match = re.search(r'Document No:\s*([A-Z0-9\s\-]+)', transaction_data, re.IGNORECASE)
                document_date_match = re.search(r'Document Date:\s*([\d/-]+)', transaction_data, re.IGNORECASE)
                branch_id_match = re.search(r'Branch ID:\s*([0-9]+)', transaction_data, re.IGNORECASE)
                
                if document_no_match:
                    result['document_no'] = document_no_match.group(1).strip()
                    result['classification'] = self._extract_document_no_prefix(result['document_no']) or 'UNKNOWN'
                    result['complete_filename'] = result['document_no']
                    result['is_valid_voucher'] = result['classification'] in self.voucher_types
                    
                    if document_date_match:
                        result['document_date'] = document_date_match.group(1).strip()
                    if branch_id_match:
                        result['branch_id'] = branch_id_match.group(1).strip()
                else:
                    # Try alternative patterns
                    alt_match = re.search(r'([A-Z]{2,}\d{2,}[\s\-]*\d+)', transaction_data)
                    if alt_match:
                        result['document_no'] = alt_match.group(1).strip()
                        result['classification'] = self._extract_document_no_prefix(result['document_no']) or 'UNKNOWN'
                        result['complete_filename'] = result['document_no']
                        result['is_valid_voucher'] = result['classification'] in self.voucher_types
                    else:
                        # No document number found via regex
                        # Check if this is a general document type that doesn't require document_no
                        general_doc_types = ['Proof of Payment', 'ID', 'SPA', 'Invoices', 'Invoice']
                        is_general_doc = document_type in general_doc_types or any(dt.lower() in document_type.lower() for dt in general_doc_types)
                        
                        if extraction_method == 'voucher_specific' and not is_general_doc:
                            # For vouchers with voucher_specific extraction, missing document_no is critical
                            filename = os.path.basename(image_path)
                            result['document_no'] = os.path.splitext(filename)[0]
                            result['complete_filename'] = result['document_no']
                            result['classification'] = 'UNKNOWN'
                            result['success'] = False
                            result['error'] = "Could not extract Document No from document"
                            result['organized_path'] = None
                            logger.warning(f"Voucher document missing document_no - marking as failed: {document_type}")
                        else:
                            # For general documents (Proof of Payment, ID, SPA, etc.), missing document_no is acceptable
                            logger.info(f"No document number found for document type '{document_type}' - treating as general document")
                            result['document_no'] = ''
                            result['classification'] = document_type
                            result['complete_filename'] = original_filename or 'document'
                            result['is_valid_voucher'] = False
                            # Keep success=True for general documents without document_no
                            if not result.get('success'):
                                result['success'] = True
                            logger.info(f"Document will be processed as general document type: {document_type}")
            
            # Generate organized path
            if result['success']:
                # For vouchers, use existing voucher path structure
                if result.get('is_valid_voucher') and result['classification'] in self.voucher_types:
                    # Vouchers require document_no
                    if result['document_no']:
                        result['organized_path'] = self._create_organized_path(
                            document_no=result['document_no'],
                            document_date=result['document_date'],
                            branch_id=result['branch_id'],
                            voucher_type=result['classification']
                        )
                        logger.info(f"Valid voucher - will organize to: {result['organized_path']}")
                    else:
                        logger.warning("Valid voucher but no document_no - marking as attachment")
                        result['organized_path'] = None
                        result['needs_attachment'] = True
                elif not result.get('is_valid_voucher') and document_type.lower() == 'voucher':
                    # Attachment voucher (no document_no)
                    result['organized_path'] = None
                    result['needs_attachment'] = True
                    logger.info("Attachment document - will search for matching valid voucher")
                else:
                    # For general documents, create path even without document_no
                    # Use timestamp as fallback identifier
                    import time
                    fallback_id = f"doc_{int(time.time() * 1000)}"
                    result['organized_path'] = self._create_general_organized_path(
                        document_type=document_type,
                        document_date=result['document_date'],
                        document_no=result.get('document_no'),
                        fallback_id=fallback_id
                    )
                    logger.info(f"General document - will organize to: {result['organized_path']}")
            
            # Convert image to PDF
            pdf_path = None
            if result['success']:
                try:
                    file_extension = os.path.splitext(image_path)[1].lower()
                    
                    if file_extension != '.pdf':
                        logger.info(f"Converting image to PDF: {image_path}")
                        pdf_path = self._convert_image_to_pdf(image_path)
                        if pdf_path:
                            result['pdf_path'] = pdf_path
                            result['converted_to_pdf'] = True
                            logger.info(f"Successfully converted to PDF: {pdf_path}")
                        else:
                            result['pdf_path'] = image_path
                            result['converted_to_pdf'] = False
                    else:
                        result['pdf_path'] = image_path
                        result['converted_to_pdf'] = False
                except Exception as e:
                    logger.error(f"Error during PDF conversion: {e}")
                    result['pdf_path'] = image_path
                    result['converted_to_pdf'] = False
            else:
                result['pdf_path'] = image_path
                result['converted_to_pdf'] = False
            
            logger.info(f"Processing result: success={result['success']}, classification={result['classification']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'document_no': None,
                'document_type': 'Other',
                'classification': 'UNKNOWN',
                'classification_confidence': 0.0,
                'extracted_data': {},
                'method': 'error',
                'organized_path': None
            }

