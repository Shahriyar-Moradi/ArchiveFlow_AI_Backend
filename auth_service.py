"""
Authentication service for RizanAI
Handles user registration, login, and JWT token management
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import aiofiles

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User data file path
USERS_FILE = Path(__file__).parent / "users.json"

class User(BaseModel):
    """User model"""
    id: str
    email: str
    hashed_password: str
    full_name: str
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None

class UserCreate(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str

class Token(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Token data model"""
    user_id: Optional[str] = None
    email: Optional[str] = None

class AuthService:
    """Authentication service"""
    
    def __init__(self):
        self.users_file = USERS_FILE
        self.users: Dict[str, User] = {}
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_id, user_data in data.items():
                        # Convert datetime strings back to datetime objects
                        user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        if user_data.get('last_login'):
                            user_data['last_login'] = datetime.fromisoformat(user_data['last_login'])
                        self.users[user_id] = User(**user_data)
                print(f"✅ Loaded {len(self.users)} users from {self.users_file}")
            else:
                print("📂 No users file found, starting with empty user database")
        except Exception as e:
            print(f"❌ Error loading users: {e}")
            self.users = {}
    
    async def save_users(self):
        """Save users to JSON file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            users_data = {}
            for user_id, user in self.users.items():
                user_dict = user.dict()
                user_dict['created_at'] = user.created_at.isoformat()
                if user.last_login:
                    user_dict['last_login'] = user.last_login.isoformat()
                users_data[user_id] = user_dict
            
            async with aiofiles.open(self.users_file, 'w') as f:
                await f.write(json.dumps(users_data, indent=2))
            print(f"💾 Saved {len(self.users)} users to {self.users_file}")
        except Exception as e:
            print(f"❌ Error saving users: {e}")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            email: str = payload.get("email")
            
            if user_id is None or email is None:
                return None
            
            return TokenData(user_id=user_id, email=email)
        except JWTError:
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def register_user(self, user_create: UserCreate) -> Dict[str, Any]:
        """Register a new user"""
        # Check if user already exists
        if self.get_user_by_email(user_create.email):
            return {
                "success": False,
                "error": "Email already registered",
                "error_code": "EMAIL_EXISTS"
            }
        
        # Validate password strength
        if len(user_create.password) < 8:
            return {
                "success": False,
                "error": "Password must be at least 8 characters long",
                "error_code": "WEAK_PASSWORD"
            }
        
        # Create new user
        user_id = secrets.token_urlsafe(16)
        hashed_password = self.get_password_hash(user_create.password)
        
        new_user = User(
            id=user_id,
            email=user_create.email,
            hashed_password=hashed_password,
            full_name=user_create.full_name,
            created_at=datetime.now()
        )
        
        self.users[user_id] = new_user
        await self.save_users()
        
        # Create tokens
        access_token = self.create_access_token(data={"sub": user_id, "email": user_create.email})
        refresh_token = self.create_refresh_token(data={"sub": user_id, "email": user_create.email})
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": user_id,
                "email": user_create.email,
                "full_name": user_create.full_name,
                "created_at": new_user.created_at.isoformat()
            },
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        }
    
    async def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        
        if not user:
            return {
                "success": False,
                "error": "Invalid email or password",
                "error_code": "INVALID_CREDENTIALS"
            }
        
        if not user.is_active:
            return {
                "success": False,
                "error": "Account is deactivated",
                "error_code": "ACCOUNT_DEACTIVATED"
            }
        
        if not self.verify_password(password, user.hashed_password):
            return {
                "success": False,
                "error": "Invalid email or password",
                "error_code": "INVALID_CREDENTIALS"
            }
        
        # Update last login
        user.last_login = datetime.now()
        await self.save_users()
        
        # Create tokens
        access_token = self.create_access_token(data={"sub": user.id, "email": user.email})
        refresh_token = self.create_refresh_token(data={"sub": user.id, "email": user.email})
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "last_login": user.last_login.isoformat()
            },
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        }
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        token_data = self.verify_token(refresh_token)
        
        if not token_data or not token_data.user_id:
            return {
                "success": False,
                "error": "Invalid refresh token",
                "error_code": "INVALID_REFRESH_TOKEN"
            }
        
        user = self.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            return {
                "success": False,
                "error": "User not found or inactive",
                "error_code": "USER_NOT_FOUND"
            }
        
        # Create new access token
        access_token = self.create_access_token(data={"sub": user.id, "email": user.email})
        
        return {
            "success": True,
            "message": "Token refreshed successfully",
            "tokens": {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
            }
        }
    
    async def get_current_user(self, token: str) -> Dict[str, Any]:
        """Get current user from token"""
        token_data = self.verify_token(token)
        
        if not token_data or not token_data.user_id:
            return {
                "success": False,
                "error": "Invalid token",
                "error_code": "INVALID_TOKEN"
            }
        
        user = self.get_user_by_id(token_data.user_id)
        if not user or not user.is_active:
            return {
                "success": False,
                "error": "User not found or inactive",
                "error_code": "USER_NOT_FOUND"
            }
        
        return {
            "success": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
        }

# Global auth service instance
auth_service = AuthService()
