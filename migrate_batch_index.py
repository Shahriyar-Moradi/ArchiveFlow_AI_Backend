#!/usr/bin/env python3
"""
Migration Script: Build Batch Index from Existing Batches

This script scans all existing batches in S3 and creates a centralized
batch index file at batches/_batch_index.json for fast batch listing.

Usage:
    python migrate_batch_index.py
"""

import sys
import os
from datetime import datetime

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s3_service import S3Service
from aws_config import aws_config

def migrate_batch_index():
    """Build batch index from existing batches"""
    
    print("=" * 60)
    print("🔄 Batch Index Migration Script")
    print("=" * 60)
    print()
    
    # Initialize S3 service
    print("📡 Initializing S3 service...")
    s3_service = S3Service()
    
    if not aws_config.aws_available or not s3_service.s3_client:
        print("❌ AWS not configured. Please check your .env file.")
        print("   Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
        return False
    
    print(f"✅ Connected to S3 bucket: {s3_service.bucket_name}")
    print()
    
    # Get all existing batches using the slow method
    print("📂 Scanning existing batches in S3...")
    print("   (This may take a while if you have many batches)")
    print()
    
    result = s3_service._list_batches_from_s3_slow()
    
    if not result['success']:
        print(f"❌ Failed to scan batches: {result.get('error')}")
        return False
    
    batches = result['batches']
    print(f"✅ Found {len(batches)} existing batches")
    print()
    
    if len(batches) == 0:
        print("ℹ️  No batches found. Creating empty index.")
        empty_index = {
            'batches': [],
            'total_batches': 0,
            'last_updated': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Write empty index
        try:
            import json
            s3_service.s3_client.put_object(
                Bucket=s3_service.bucket_name,
                Key='batches/_batch_index.json',
                Body=json.dumps(empty_index, indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            print("✅ Created empty batch index")
            return True
        except Exception as e:
            print(f"❌ Failed to create empty index: {e}")
            return False
    
    # Display batch information
    print("📋 Batch Summary:")
    print("-" * 60)
    for i, batch in enumerate(batches[:5], 1):
        print(f"  {i}. {batch.get('batch_name', 'Unknown')} ({batch.get('batch_id', 'N/A')})")
        print(f"     Files: {batch.get('file_count', 0)}, Status: {batch.get('status', 'unknown')}")
    
    if len(batches) > 5:
        print(f"  ... and {len(batches) - 5} more batches")
    print("-" * 60)
    print()
    
    # Build index
    print("🔨 Building batch index...")
    
    index = {
        'batches': batches,
        'total_batches': len(batches),
        'last_updated': datetime.now().isoformat(),
        'version': '1.0',
        'migration_info': {
            'migrated_at': datetime.now().isoformat(),
            'migrated_count': len(batches),
            'script_version': '1.0'
        }
    }
    
    # Write index to S3
    print("💾 Writing batch index to S3...")
    try:
        import json
        index_key = 'batches/_batch_index.json'
        s3_service.s3_client.put_object(
            Bucket=s3_service.bucket_name,
            Key=index_key,
            Body=json.dumps(index, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        
        print(f"✅ Successfully created batch index: s3://{s3_service.bucket_name}/{index_key}")
        print()
        print("=" * 60)
        print("🎉 Migration Complete!")
        print("=" * 60)
        print()
        print(f"✅ Indexed {len(batches)} batches")
        print(f"📁 Index file: batches/_batch_index.json")
        print()
        print("Next steps:")
        print("1. Restart your backend server")
        print("2. Refresh the frontend")
        print("3. Check console logs for 'FAST method' message")
        print("4. Batch loading should now be 10-30x faster!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to write batch index: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

def verify_index():
    """Verify the batch index was created successfully"""
    print()
    print("🔍 Verifying batch index...")
    
    s3_service = S3Service()
    
    if not aws_config.aws_available or not s3_service.s3_client:
        print("❌ AWS not configured")
        return False
    
    result = s3_service.get_batch_index()
    
    if result['success']:
        index = result['index']
        print(f"✅ Index verified successfully!")
        print(f"   Total batches: {index.get('total_batches', 0)}")
        print(f"   Last updated: {index.get('last_updated', 'Unknown')}")
        print(f"   Version: {index.get('version', 'Unknown')}")
        return True
    else:
        print(f"❌ Index verification failed: {result.get('error')}")
        return False

if __name__ == '__main__':
    print()
    success = migrate_batch_index()
    
    if success:
        verify_index()
        sys.exit(0)
    else:
        print()
        print("❌ Migration failed. Please check the error messages above.")
        sys.exit(1)

