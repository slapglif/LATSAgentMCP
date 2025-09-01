"""
Sample authentication module for testing LATS investigation
Contains intentional bugs and patterns for testing
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class AuthenticationError(Exception):
    """Custom authentication exception"""
    pass


class SessionManager:
    """Manages user sessions"""
    
    def __init__(self):
        self.sessions = {}  # BUG: In-memory storage loses sessions on restart
        self.timeout = timedelta(hours=1)
    
    def create_session(self, user_id: str) -> str:
        """Create a new session for user"""
        session_id = hashlib.md5(f"{user_id}{datetime.now()}".encode()).hexdigest()
        self.sessions[session_id] = {
            'user_id': user_id,
            'created': datetime.now(),
            'last_access': datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Check if session is valid"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        # BUG: Doesn't actually check timeout
        return True
    
    def get_user_id(self, session_id: str) -> Optional[str]:
        """Get user ID from session"""
        if session_id in self.sessions:
            return self.sessions[session_id]['user_id']
        return None


class LoginHandler:
    """Handles user login logic"""
    
    def __init__(self, user_db: Dict[str, Dict[str, Any]]):
        self.user_db = user_db
        self.session_manager = SessionManager()
        self.max_attempts = 3
        self.failed_attempts = {}  # Track failed login attempts
    
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user and create session"""
        # Check failed attempts
        if username in self.failed_attempts:
            if self.failed_attempts[username] >= self.max_attempts:
                raise AuthenticationError("Account locked due to too many failed attempts")
        
        # Validate credentials
        if username not in self.user_db:
            self._record_failed_attempt(username)
            raise AuthenticationError("Invalid username or password")
        
        user = self.user_db[username]
        
        # BUG: Plain text password comparison (should use hashing)
        if user['password'] != password:
            self._record_failed_attempt(username)
            raise AuthenticationError("Invalid username or password")
        
        # Check if account is active
        if not user.get('is_active', True):
            raise AuthenticationError("Account is disabled")
        
        # Create session
        session_id = self.session_manager.create_session(username)
        
        # Reset failed attempts on successful login
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        return {
            'session_id': session_id,
            'user_id': username,
            'role': user.get('role', 'user'),
            'login_time': datetime.now().isoformat()
        }
    
    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = 0
        self.failed_attempts[username] += 1
    
    def logout(self, session_id: str) -> bool:
        """Logout user by invalidating session"""
        if session_id in self.session_manager.sessions:
            # BUG: Doesn't actually remove the session
            return True
        return False
    
    def reset_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Reset user password"""
        if username not in self.user_db:
            return False
        
        user = self.user_db[username]
        
        # BUG: No password strength validation
        if user['password'] == old_password:
            user['password'] = new_password
            return True
        
        return False


class PermissionChecker:
    """Check user permissions"""
    
    ROLE_PERMISSIONS = {
        'admin': ['read', 'write', 'delete', 'admin'],
        'user': ['read', 'write'],
        'guest': ['read']
    }
    
    @classmethod
    def has_permission(cls, role: str, action: str) -> bool:
        """Check if role has permission for action"""
        if role not in cls.ROLE_PERMISSIONS:
            return False
        
        # BUG: Case-sensitive comparison
        return action in cls.ROLE_PERMISSIONS[role]
    
    @classmethod
    def get_permissions(cls, role: str) -> list:
        """Get all permissions for a role"""
        return cls.ROLE_PERMISSIONS.get(role, [])