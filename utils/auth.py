import streamlit as st
import bcrypt
import json
import os
from typing import Dict, Optional

class AuthManager:
    def __init__(self, credentials_file: str = "config/credentials.json"):
        self.credentials_file = credentials_file
        self._ensure_credentials_file_exists()
        self.credentials = self._load_credentials()

    def _ensure_credentials_file_exists(self):
        """Create credentials file and directory if they don't exist."""
        os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
        if not os.path.exists(self.credentials_file):
            # Default credentials (admin:password123)
            default_credentials = {
                "admin": {
                    "password": self.hash_password("chirac@LTDK123"),
                    "role": "admin"
                }
            }
            with open(self.credentials_file, 'w') as f:
                json.dump(default_credentials, f)

    def _load_credentials(self) -> Dict:
        """Load credentials from JSON file."""
        with open(self.credentials_file, 'r') as f:
            return json.load(f)

    def _save_credentials(self):
        """Save credentials to JSON file."""
        with open(self.credentials_file, 'w') as f:
            json.dump(self.credentials, f, indent=4)

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Authenticate a user and return user info if successful."""
        if username in self.credentials:
            stored_password = self.credentials[username]["password"]
            if self.verify_password(password, stored_password):
                return {
                    "username": username,
                    "role": self.credentials[username].get("role", "user")
                }
        return None

    def add_user(self, username: str, password: str, role: str = "user") -> bool:
        """Add a new user (admin only)."""
        if username in self.credentials:
            return False
        self.credentials[username] = {
            "password": self.hash_password(password),
            "role": role
        }
        self._save_credentials()
        return True

def check_authentication():
    """Check if user is authenticated, redirect to login if not."""
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.error("Please log in to access this page")
        st.session_state['current_page'] = "0_Login"
        st.stop()

def logout():
    """Handle logout functionality."""
    if 'authenticated' in st.session_state:
        del st.session_state['authenticated']
        del st.session_state['user_info']
    st.session_state['current_page'] = "0_Login"
    st.success("Successfully logged out")
    st.rerun()