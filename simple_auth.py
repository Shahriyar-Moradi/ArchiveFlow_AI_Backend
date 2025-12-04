"""
Simple Authentication System
Just the basics: register, login, verify
"""

import os
import json
import secrets
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

import bcrypt
from jose import jwt

logger = logging.getLogger(__name__)

# Simple configuration
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
USERS_FILE = Path(__file__).parent / "users.json"

class SimpleAuth:
    """Simple authentication class"""
    
    def __init__(self):
        self.users = self.load_users()
        self.create_demo_user()
        self.create_initial_users()
    
    def load_users(self) -> Dict:
        """Load users from JSON file"""
        if USERS_FILE.exists():
            try:
                with open(USERS_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save users to JSON file"""
        with open(USERS_FILE, 'w') as f:
            json.dump(self.users, f, indent=2, default=str)
    
    def create_demo_user(self):
        """Create demo admin user if doesn't exist"""
        demo_email = "admin@example.com"
        
        # Check if demo user already exists
        for user in self.users.values():
            if user['email'] == demo_email:
                # Ensure existing admin user has role field
                if 'role' not in user:
                    user['role'] = 'admin'
                    self.save_users()
                return
        
        # Create demo admin user
        user_id = "demo_admin_user"
        demo_user = {
            "id": user_id,
            "email": demo_email,
            "password": self.hash_password("advisorynext"),
            "full_name": "Admin User",
            "role": "admin",  # Admin role
            "created_at": datetime.now().isoformat()
        }
        
        self.users[user_id] = demo_user
        self.save_users()
        print(f"✅ Demo admin user created: {demo_email} / advisorynext")
    
    def create_initial_users(self):
        """Create initial admin and agent users if they don't exist"""
        initial_users = [
            {
                "email": "admin@example.com",
                "password": "Admin",
                "full_name": "Admin User",
                "role": "admin",
                "user_id": "admin_example_user"
            },
            {
                "email": "agent1@example.com",
                "password": "agent1",
                "full_name": "Agent 1",
                "role": "agent",
                "user_id": "agent1_example_user"
            }
        ]
        
        for user_data in initial_users:
            email = user_data["email"]
            password = user_data["password"]
            full_name = user_data["full_name"]
            role = user_data["role"]
            user_id = user_data["user_id"]
            
            # Check if user already exists
            user_exists = False
            existing_user_id = None
            for uid, existing_user in self.users.items():
                if existing_user.get('email', '').lower() == email.lower():
                    user_exists = True
                    existing_user_id = uid
                    # Update role if needed
                    if existing_user.get('role') != role:
                        existing_user['role'] = role
                        self.save_users()
                        print(f"✅ Updated role for {email} -> {role}")
                    # Verify password is correct - if login fails, update password
                    if not self.verify_password(password, existing_user.get('password', '')):
                        existing_user['password'] = self.hash_password(password)
                        self.save_users()
                        print(f"✅ Updated password for {email}")
                    break
            
            if not user_exists:
                # Create user
                user = {
                    "id": user_id,
                    "email": email,
                    "password": self.hash_password(password),
                    "full_name": full_name,
                    "role": role,
                    "created_at": datetime.now().isoformat(),
                    "is_active": True
                }
                
                self.users[user_id] = user
                self.save_users()
                print(f"✅ Created {role} user: {email} / {password}")
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password"""
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    
    def create_token(self, user_id: str, email: str) -> str:
        """Create JWT token"""
        expire = datetime.utcnow() + timedelta(days=7)  # 7 days
        data = {
            "sub": user_id,
            "email": email,
            "exp": expire
        }
        return jwt.encode(data, SECRET_KEY, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            return payload
        except:
            return None
    
    def register(self, email: str, password: str, full_name: str) -> Dict:
        """
        Register new agent account.
        NOTE: Registration is disabled. This method is kept for internal use only.
        All new users are created with 'agent' role by default.
        """
        # Normalize email
        email = email.strip().lower()
        
        # Check if user exists
        for user in self.users.values():
            if user.get('email', '').lower() == email:
                return {"success": False, "error": "Email already registered"}
        
        # Validate password (additional check, though backend also validates)
        if len(password) < 8:
            return {"success": False, "error": "Password must be at least 8 characters"}
        
        # Validate full name
        if not full_name or not full_name.strip():
            return {"success": False, "error": "Full name is required"}
        
        # Create user
        user_id = secrets.token_urlsafe(16)
        user = {
            "id": user_id,
            "email": email,
            "password": self.hash_password(password),
            "full_name": full_name.strip(),
            "role": "agent",  # Default role is agent
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        self.users[user_id] = user
        self.save_users()
        
        logger.info(f"✅ New agent registered: {email} (ID: {user_id})")
        
        # Create token
        token = self.create_token(user_id, email)
        
        return {
            "success": True,
            "message": "Account created successfully",
            "user": {
                "id": user_id,
                "email": email,
                "full_name": full_name.strip(),
                "role": "agent"
            },
            "tokens": {
                "access_token": token,
                "refresh_token": token,
                "token_type": "bearer"
            }
        }
    
    def login(self, email: str, password: str) -> Dict:
        """
        Login user with email and password.
        Returns JWT token for authenticated requests.
        """
        # Normalize email
        email = email.strip().lower()
        
        # Find user
        user = None
        for u in self.users.values():
            if u.get('email', '').lower() == email:
                user = u
                break
        
        if not user:
            logger.warning(f"Login attempt with non-existent email: {email}")
            return {"success": False, "error": "Invalid email or password"}
        
        # Check if user is active
        if user.get('is_active') is False:
            logger.warning(f"Login attempt for inactive user: {email}")
            return {"success": False, "error": "Account is inactive. Please contact administrator."}
        
        # Verify password
        if not self.verify_password(password, user['password']):
            logger.warning(f"Failed login attempt for: {email}")
            return {"success": False, "error": "Invalid email or password"}
        
        # Create token
        token = self.create_token(user['id'], user['email'])
        
        logger.info(f"✅ User logged in: {email} (ID: {user['id']}, Role: {user.get('role', 'agent')})")
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user.get('role', 'agent')  # Default to agent if role not set
            },
            "tokens": {
                "access_token": token,
                "refresh_token": token,
                "token_type": "bearer"
            }
        }
    
    def get_user(self, token: str) -> Dict:
        """Get user from token"""
        payload = self.verify_token(token)
        if not payload:
            return {"success": False, "error": "Invalid token"}
        
        user_id = payload.get('sub')
        user = self.users.get(user_id)
        
        if not user:
            return {"success": False, "error": "User not found"}
        
        return {
            "success": True,
            "user": {
                "id": user['id'],
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user.get('role', 'agent')  # Default to agent if role not set
            }
        }

    def is_admin(self, user_id: str) -> bool:
        """Check if user is admin"""
        user = self.users.get(user_id)
        if not user:
            return False
        return user.get('role') == 'admin'
    
    def get_all_users(self) -> list:
        """Get all users (admin only)"""
        return [
            {
                "id": user_id,
                "email": user['email'],
                "full_name": user['full_name'],
                "role": user.get('role', 'agent'),
                "created_at": user.get('created_at')
            }
            for user_id, user in self.users.items()
        ]
    
    def update_user_role(self, user_id: str, role: str) -> bool:
        """Update user role (admin only)"""
        if role not in ['admin', 'agent']:
            return False
        
        user = self.users.get(user_id)
        if not user:
            return False
        
        user['role'] = role
        self.save_users()
        return True

# Global instance
simple_auth = SimpleAuth()
