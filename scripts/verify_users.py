"""
Script to verify users exist and can be authenticated
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simple_auth import simple_auth

def verify_users():
    """Verify the users exist and passwords work"""
    print("=" * 60)
    print("Verifying Users")
    print("=" * 60)
    
    test_users = [
        {"email": "admin@example.com", "password": "Admin"},
        {"email": "agent1@example.com", "password": "agent1"}
    ]
    
    for test_user in test_users:
        email = test_user["email"]
        password = test_user["password"]
        
        print(f"\nTesting: {email}")
        result = simple_auth.login(email, password)
        
        if result.get("success"):
            user = result.get("user", {})
            print(f"  ✅ Login successful")
            print(f"  - ID: {user.get('id')}")
            print(f"  - Name: {user.get('full_name')}")
            print(f"  - Role: {user.get('role')}")
        else:
            print(f"  ❌ Login failed: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    verify_users()

