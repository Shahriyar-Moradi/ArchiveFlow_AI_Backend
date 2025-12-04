#!/usr/bin/env python3
"""
Test script for zero-shot image validation using CLIP classifier
Tests if the CLIP model loads correctly and validates images properly
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if required libraries are available"""
    logger.info("=" * 60)
    logger.info("TEST 1: Checking Required Imports")
    logger.info("=" * 60)
    
    results = {}
    
    # Test PIL
    try:
        from PIL import Image
        results['PIL'] = True
        logger.info("✅ PIL (Pillow) is available")
    except ImportError:
        results['PIL'] = False
        logger.error("❌ PIL (Pillow) is NOT available - install with: pip install Pillow")
    
    # Test transformers
    try:
        from transformers import pipeline
        results['transformers'] = True
        logger.info("✅ transformers library is available")
    except ImportError:
        results['transformers'] = False
        logger.error("❌ transformers library is NOT available - install with: pip install transformers")
    
    # Test torch (optional but recommended)
    try:
        import torch
        results['torch'] = True
        logger.info(f"✅ torch is available (version: {torch.__version__})")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        results['torch'] = False
        logger.warning("⚠️  torch is NOT available - install with: pip install torch")
    
    return results

def test_clip_classifier_loading():
    """Test if CLIP classifier loads correctly"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Loading CLIP Classifier")
    logger.info("=" * 60)
    
    try:
        from services.document_processor import _get_clip_classifier
        
        logger.info("Attempting to load CLIP classifier...")
        logger.info("This may take 1-2 minutes on first load (model download)...")
        
        classifier = _get_clip_classifier()
        
        if classifier is None:
            logger.error("❌ CLIP classifier failed to load - check error messages above")
            return None
        
        logger.info("✅ CLIP classifier loaded successfully!")
        logger.info(f"   Model type: {type(classifier)}")
        
        # Test with a simple classification
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image (white square)
            test_img = Image.new('RGB', (224, 224), color='white')
            
            logger.info("Testing classifier with a simple test image...")
            results = classifier(test_img, candidate_labels=["Valid document", "other"])
            
            if results and len(results) > 0:
                logger.info("✅ Classifier is working!")
                logger.info(f"   Test result: {results[0]}")
                return classifier
            else:
                logger.error("❌ Classifier returned empty results")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error testing classifier: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to import or load CLIP classifier: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_validation_with_image(image_path: str):
    """Test validation with a specific image file"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"TEST 3: Validating Image: {image_path}")
    logger.info("=" * 60)
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        from services.document_processor import DocumentProcessor
        
        logger.info("Initializing DocumentProcessor...")
        processor = DocumentProcessor()
        
        logger.info(f"Running validation on: {image_path}")
        result = processor._validate_document_image(image_path)
        
        validation_status = result.get('validation_status')
        confidence = result.get('validation_confidence', 0.0)
        label = result.get('validation_label', '')
        error = result.get('error')
        
        logger.info("")
        logger.info("Validation Results:")
        logger.info(f"  Status: {validation_status}")
        logger.info(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        logger.info(f"  Label: {label}")
        if error:
            logger.info(f"  Error: {error}")
        
        # Interpret results
        if validation_status == 'valid':
            logger.info("✅ Image VALIDATED as a document (confidence >= 90%)")
        elif validation_status == 'need_review':
            logger.info("⚠️  Image needs REVIEW (confidence 40-90%)")
        elif validation_status == 'failed':
            logger.info("❌ Image REJECTED (confidence < 40% or validation error)")
        else:
            logger.warning(f"⚠️  Unknown validation status: {validation_status}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during validation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_full_document_processing(image_path: str):
    """Test full document processing including validation"""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"TEST 4: Full Document Processing: {image_path}")
    logger.info("=" * 60)
    
    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        return None
    
    try:
        from services.document_processor import DocumentProcessor
        
        logger.info("Initializing DocumentProcessor...")
        processor = DocumentProcessor()
        
        logger.info(f"Processing document: {image_path}")
        result = processor.process_document(image_path, original_filename=os.path.basename(image_path))
        
        success = result.get('success', False)
        validation_status = result.get('validation_status', 'unknown')
        validation_confidence = result.get('validation_confidence', 0.0)
        error = result.get('error', '')
        
        logger.info("")
        logger.info("Processing Results:")
        logger.info(f"  Success: {success}")
        logger.info(f"  Validation Status: {validation_status}")
        logger.info(f"  Validation Confidence: {validation_confidence:.4f}")
        if error:
            logger.info(f"  Error: {error}")
        
        if success:
            logger.info("✅ Document processed successfully!")
            logger.info(f"  Document Type: {result.get('document_type', 'N/A')}")
            logger.info(f"  Classification: {result.get('classification', 'N/A')}")
        else:
            logger.error("❌ Document processing failed")
            if validation_status == 'failed':
                logger.error("   Reason: Image validation failed (not a document)")
            elif validation_status == 'need_review':
                logger.warning("   Reason: Image needs human review")
            else:
                logger.error(f"   Reason: {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during document processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Run all tests"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("ZERO-SHOT IMAGE VALIDATION TEST SUITE")
    logger.info("=" * 60)
    logger.info("")
    
    # Test 1: Check imports
    import_results = test_imports()
    
    if not import_results.get('PIL'):
        logger.error("")
        logger.error("❌ PIL is required - cannot continue tests")
        return 1
    
    if not import_results.get('transformers'):
        logger.error("")
        logger.error("❌ transformers is required - cannot continue tests")
        return 1
    
    # Test 2: Load CLIP classifier
    classifier = test_clip_classifier_loading()
    
    if classifier is None:
        logger.error("")
        logger.error("❌ CLIP classifier failed to load - cannot continue tests")
        return 1
    
    # Test 3 & 4: Test with actual image files (if provided)
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        logger.info("")
        logger.info(f"Testing with provided image: {image_path}")
        
        # Test validation only
        validation_result = test_validation_with_image(image_path)
        
        # Test full processing
        processing_result = test_full_document_processing(image_path)
        
    else:
        logger.info("")
        logger.info("ℹ️  No image file provided for testing")
        logger.info("   Usage: python test_zero_shot_validation.py <path_to_image>")
        logger.info("   Example: python test_zero_shot_validation.py test_document.png")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUITE COMPLETE")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

