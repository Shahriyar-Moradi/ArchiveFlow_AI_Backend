#!/usr/bin/env python3
"""
Query Organized Files from Firestore
Shows how to find and filter organized voucher files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.firestore_service import FirestoreService
from google.cloud.firestore import FieldFilter

def main():
    # Initialize Firestore
    firestore = FirestoreService()
    
    print("=" * 80)
    print("  FIRESTORE ORGANIZED FILES QUERY")
    print("=" * 80)
    print()
    
    # Query 1: All completed/organized documents
    print("📊 Query 1: All Organized Files (processing_status = 'completed')")
    print("-" * 80)
    
    query = firestore.documents_collection.where(
        filter=FieldFilter('processing_status', '==', 'completed')
    )
    
    docs = list(query.limit(10).stream())
    print(f"Found {len(docs)} organized documents (showing first 10)\n")
    
    for doc in docs:
        data = doc.to_dict()
        print(f"Document ID: {doc.id}")
        print(f"  Filename: {data.get('filename', 'N/A')}")
        print(f"  Organized Path: {data.get('organized_path', 'N/A')}")
        print(f"  Category: {data.get('metadata', {}).get('ui_category', 'N/A')}")
        print(f"  Document No: {data.get('metadata', {}).get('document_no', 'N/A')}")
        print(f"  Flow ID: {data.get('flow_id', 'N/A')}")
        print()
    
    # Query 2: Filter by category
    print("\n📊 Query 2: Proof of Payment Documents")
    print("-" * 80)
    
    query = firestore.documents_collection.where(
        filter=FieldFilter('metadata.ui_category', '==', 'Proof of Payment')
    )
    
    docs = list(query.limit(5).stream())
    print(f"Found {len(docs)} Proof of Payment documents (showing first 5)\n")
    
    for doc in docs:
        data = doc.to_dict()
        print(f"Document: {data.get('filename', 'N/A')}")
        print(f"  Path: {data.get('organized_path', 'N/A')}")
        print(f"  Amount: AED {data.get('metadata', {}).get('invoice_amount_aed', 'N/A')}")
        print()
    
    # Query 3: Documents with organized_path (any organized file)
    print("\n📊 Query 3: All Documents with Organized Paths")
    print("-" * 80)
    
    # Get all documents and filter client-side (Firestore doesn't support "field exists")
    all_docs = firestore.documents_collection.limit(100).stream()
    
    organized_count = 0
    for doc in all_docs:
        data = doc.to_dict()
        if data.get('organized_path'):
            organized_count += 1
            if organized_count <= 5:  # Show first 5
                # Parse organized path
                path = data.get('organized_path', '')
                parts = path.split('/')
                
                print(f"Document: {data.get('filename', 'N/A')}")
                if len(parts) >= 5:
                    print(f"  Category: {parts[1]}")
                    print(f"  Year: {parts[2]}")
                    print(f"  Month: {parts[3]}")
                    print(f"  Date: {parts[4]}")
                print(f"  Full Path: {path}")
                print()
    
    print(f"Total organized documents found: {organized_count}")
    
    # Query 4: Group by category
    print("\n📊 Query 4: Count by Category")
    print("-" * 80)
    
    categories = {}
    all_docs = firestore.documents_collection.where(
        filter=FieldFilter('processing_status', '==', 'completed')
    ).limit(1000).stream()
    
    for doc in all_docs:
        data = doc.to_dict()
        category = data.get('metadata', {}).get('ui_category', 'Unknown')
        categories[category] = categories.get(category, 0) + 1
    
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} documents")
    
    print("\n" + "=" * 80)
    print("✅ Query Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()

