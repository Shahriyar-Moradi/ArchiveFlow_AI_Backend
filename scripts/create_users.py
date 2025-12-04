"""
Script to create initial users for the system
Run this script to create admin and agent users
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from simple_auth import simple_auth

def create_users():
    """Create initial users"""
    print("=" * 60)
    print("Creating Initial Users")
    print("=" * 60)
    
    users_to_create = [
        {
            "email": "admin@example.com",
            "password": "Admin",
            "full_name": "Admin User",
            "role": "admin"
        },
        {
            "email": "agent1@example.com",
            "password": "agent1",
            "full_name": "Agent 1",
            "role": "agent"
        }
    ]
    
    created = []
    skipped = []
    
    for user_data in users_to_create:
        email = user_data["email"]
        password = user_data["password"]
        full_name = user_data["full_name"]
        role = user_data["role"]
        
        # Check if user already exists
        user_exists = False
        for user in simple_auth.users.values():
            if user.get('email', '').lower() == email.lower():
                user_exists = True
                # Update role if needed
                if user.get('role') != role:
                    user['role'] = role
                    simple_auth.save_users()
                    print(f"✅ Updated role for existing user: {email} -> {role}")
                else:
                    print(f"⏭️  User already exists: {email}")
                skipped.append(email)
                break
        
        if not user_exists:
            # Create user
            result = simple_auth.register(email, password, full_name)
            if result.get("success"):
                # Update role if not agent
                if role == "admin":
                    user_id = result["user"]["id"]
                    simple_auth.update_user_role(user_id, "admin")
                    print(f"✅ Created admin user: {email} / {password}")
                else:
                    print(f"✅ Created agent user: {email} / {password}")
                created.append(email)
            else:
                print(f"❌ Failed to create user {email}: {result.get('error')}")
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Created: {len(created)} users")
    print(f"Skipped (already exist): {len(skipped)} users")
    
    if created:
        print("\nNew users created:")
        for email in created:
            print(f"  - {email}")
    
    if skipped:
        print("\nExisting users:")
        for email in skipped:
            print(f"  - {email}")
    
    print("\n✅ User creation complete!")
    print("\nLogin credentials:")
    print("  1. Admin: admin@example.com / Admin")
    print("  2. Agent: agent1@example.com / agent1")
    print("=" * 60)

if __name__ == "__main__":
    create_users()

