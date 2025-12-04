#!/usr/bin/env python3
"""
Test script to verify GCP integration is working correctly
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config():
    """Test configuration loading"""
    print("\n=== Testing Configuration ===")
    try:
        from config import settings
        print("✅ Config loaded successfully")
        print(f"   GCS Bucket: {settings.GCS_BUCKET_NAME}")
        print(f"   GCS Project: {settings.GCS_PROJECT_ID}")
        print(f"   Firestore Project: {settings.FIRESTORE_PROJECT_ID}")
        print(f"   Anthropic Model: {settings.ANTHROPIC_MODEL}")
        print(f"   Anthropic configured: {settings.anthropic_api_key_configured}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_imports():
    """Test if all GCP services can be imported"""
    print("\n=== Testing Service Imports ===")
    all_success = True
    
    # Test document_processor
    print("  Testing document_processor...")
    try:
        from services.document_processor import DocumentProcessor
        print("    ✅ DocumentProcessor imported")
    except Exception as e:
        print(f"    ❌ DocumentProcessor error: {e}")
        all_success = False
    
    # Test firestore_service
    print("  Testing firestore_service...")
    try:
        from services.firestore_service import FirestoreService
        print("    ✅ FirestoreService imported")
    except Exception as e:
        print(f"    ❌ FirestoreService error: {e}")
        all_success = False
    
    # Test task_queue
    print("  Testing task_queue...")
    try:
        from services.task_queue import TaskQueue
        print("    ✅ TaskQueue imported")
    except Exception as e:
        print(f"    ❌ TaskQueue error: {e}")
        all_success = False
    
    # Test gcs_service
    print("  Testing gcs_service...")
    try:
        from gcs_service import GCSVoucherService
        print("    ✅ GCSVoucherService imported")
    except Exception as e:
        print(f"    ❌ GCSVoucherService error: {e}")
        all_success = False
    
    # Test category_mapper
    print("  Testing category_mapper...")
    try:
        from services.category_mapper import map_backend_to_ui_category
        print("    ✅ Category mapper imported")
    except Exception as e:
        print(f"    ❌ Category mapper error: {e}")
        all_success = False
    
    return all_success

def test_main_import():
    """Test if main.py can be imported"""
    print("\n=== Testing Main Application ===")
    try:
        from main import app
        print(f"✅ Main.py imported successfully")
        print(f"   App title: {app.title}")
        print(f"   App version: {app.version}")
        return True
    except Exception as e:
        print(f"❌ Main.py import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_routes():
    """Test if key API routes are defined"""
    print("\n=== Testing API Routes ===")
    try:
        from main import app
        routes = [route.path for route in app.routes]
        
        key_routes = [
            "/api/aws/upload",
            "/api/aws/batch-upload",
            "/api/aws/test",
            "/api/aws/test-integration",
            "/api/batches",
            "/health"
        ]
        
        all_found = True
        for route in key_routes:
            if route in routes:
                print(f"  ✅ {route}")
            else:
                print(f"  ❌ {route} - NOT FOUND")
                all_found = False
        
        print(f"\n  Total routes: {len(routes)}")
        return all_found
    except Exception as e:
        print(f"❌ Route testing error: {e}")
        return False

def test_service_initialization():
    """Test if services can be initialized"""
    print("\n=== Testing Service Initialization ===")
    all_success = True
    
    # Test if services were initialized in main.py
    try:
        from main import document_processor, firestore_service, gcs_service, task_queue
        
        if document_processor:
            print("  ✅ document_processor initialized")
        else:
            print("  ⚠️  document_processor not initialized (may need credentials)")
        
        if firestore_service:
            print("  ✅ firestore_service initialized")
        else:
            print("  ⚠️  firestore_service not initialized (may need credentials)")
        
        if gcs_service:
            print("  ✅ gcs_service initialized")
        else:
            print("  ⚠️  gcs_service not initialized (may need credentials)")
        
        if task_queue:
            print("  ✅ task_queue initialized")
        else:
            print("  ⚠️  task_queue not initialized")
        
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        all_success = False
    
    return all_success

def main():
    """Run all tests"""
    print("=" * 60)
    print("GCP Integration Test Suite")
    print("=" * 60)
    
    results = {
        "Config": test_config(),
        "Service Imports": test_service_imports(),
        "Main Import": test_main_import(),
        "API Routes": test_api_routes(),
        "Service Initialization": test_service_initialization()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("=" * 60)
        print("\nNote: Service initialization failures may be due to missing credentials.")
        print("This is expected if GCP credentials are not configured yet.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

