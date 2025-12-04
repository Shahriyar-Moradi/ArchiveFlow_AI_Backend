#!/usr/bin/env python3
"""
Test script to manually upload a file to attached_voucher/ folder
This simulates what Lambda does when it stores an unmatched attachment
"""

import boto3
import sys
import os
from datetime import datetime

# Configuration
BUCKET = "rocabucket-staging"
REGION = "me-central-1"

print("=" * 60)
print("🧪 Test Pending Documents - Manual S3 Upload")
print("=" * 60)
print()

# Get batch ID
print("📋 Available options:")
print("   1. Use existing batch: batch-20251031_184914-c27b847e")
print("   2. Enter custom batch ID")
choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"

if choice == "1":
    BATCH_ID = "batch-20251031_184914-c27b847e"
else:
    BATCH_ID = input("Enter batch ID: ").strip()

print(f"\n✅ Using batch: {BATCH_ID}")
print()

# Get file path
print("📁 Enter path to test image:")
print("   Example: ~/Desktop/test.png")
print("   Or just press Enter to create a dummy file")
test_file_path = input("\nFile path: ").strip()

# Create dummy file if none provided
if not test_file_path:
    print("\n📄 Creating dummy test image...")
    test_file_path = "/tmp/test_pending_document.txt"
    with open(test_file_path, 'w') as f:
        f.write(f"Test pending document\nCreated: {datetime.now()}\nBatch: {BATCH_ID}\n")
    print(f"   Created: {test_file_path}")

# Expand user path
test_file_path = os.path.expanduser(test_file_path)

# Check file exists
if not os.path.exists(test_file_path):
    print(f"\n❌ Error: File not found: {test_file_path}")
    sys.exit(1)

# Get document number
document_no = input("\n📝 Enter document number (e.g., TIS01-12345) [default: TEST-001]: ").strip() or "TEST-001"

# Determine file extension
file_ext = os.path.splitext(test_file_path)[1] or ".png"
s3_key = f"attached_voucher/{BATCH_ID}/{document_no}{file_ext}"

print("\n" + "=" * 60)
print("📤 Uploading to S3...")
print(f"   Bucket: {BUCKET}")
print(f"   Region: {REGION}")
print(f"   S3 Key: {s3_key}")
print("=" * 60)

# Initialize S3
try:
    s3 = boto3.client('s3', region_name=REGION)
    
    # Upload file
    s3.upload_file(test_file_path, BUCKET, s3_key)
    
    print("\n✅ SUCCESS! File uploaded to S3")
    print()
    print("=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    print(f"1. Open browser: http://localhost:8080/batch/{BATCH_ID}")
    print(f"2. You should see:")
    print(f"   - Total Documents: 1 (or more)")
    print(f"   - Batch Status: Completed")
    print(f"   - Tab: 'Pending Documents (1)'")
    print(f"3. Click 'Pending Documents' tab")
    print(f"4. See your file: {document_no}{file_ext}")
    print(f"5. Try actions: View, Match, Delete")
    print()
    print(f"🔍 Verify in S3:")
    print(f"   aws s3 ls s3://{BUCKET}/attached_voucher/{BATCH_ID}/ --region {REGION}")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error uploading file: {e}")
    print("\n💡 Make sure you have:")
    print("   1. AWS credentials configured (~/.aws/credentials)")
    print("   2. Correct permissions to upload to S3")
    print("   3. Correct bucket name and region")
    sys.exit(1)

