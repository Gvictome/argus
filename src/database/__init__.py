"""
Database module

Handles:
- SQLite database management
- CRUD operations for all entities
- Data migrations
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager


class Database:
    """
    SQLite database manager for THE EYE

    Handles all persistent storage
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def initialize(self) -> bool:
        """
        Initialize database and create tables

        Returns:
            True if successful
        """
        try:
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self._create_tables()
            return True
        except Exception as e:
            print(f"Database initialization failed: {e}")
            return False

    def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()

        # Known faces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            )
        """)

        # Events log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                data TEXT,
                media_path TEXT
            )
        """)

        # Devices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                protocol TEXT NOT NULL,
                config TEXT,
                state TEXT,
                online INTEGER DEFAULT 0,
                last_seen TIMESTAMP
            )
        """)

        # Automations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS automations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                trigger_config TEXT NOT NULL,
                action_config TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_triggered TIMESTAMP
            )
        """)

        # Audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT
            )
        """)

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        self.connection.commit()

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e

    # ========================================================================
    # Face Operations
    # ========================================================================

    def add_face(self, face_id: str, name: str, embedding: bytes) -> bool:
        """Add a known face"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO faces (id, name, embedding) VALUES (?, ?, ?)",
                (face_id, name, embedding)
            )
        return True

    def get_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face by ID"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM faces WHERE id = ?", (face_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_faces(self) -> List[Dict[str, Any]]:
        """List all known faces"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT id, name, created_at, last_seen FROM faces")
            return [dict(row) for row in cursor.fetchall()]

    def update_face_seen(self, face_id: str):
        """Update last_seen timestamp for a face"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE faces SET last_seen = ? WHERE id = ?",
                (datetime.now(), face_id)
            )

    def delete_face(self, face_id: str) -> bool:
        """Delete a face"""
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        return True

    # ========================================================================
    # Event Operations
    # ========================================================================

    def log_event(self, event_id: str, event_type: str, source: str, data: str, media_path: Optional[str] = None) -> bool:
        """Log an event"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO events (id, type, source, data, media_path) VALUES (?, ?, ?, ?, ?)",
                (event_id, event_type, source, data, media_path)
            )
        return True

    def get_events(self, limit: int = 50, offset: int = 0, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get events with pagination"""
        with self.get_cursor() as cursor:
            if event_type:
                cursor.execute(
                    "SELECT * FROM events WHERE type = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (event_type, limit, offset)
                )
            else:
                cursor.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                )
            return [dict(row) for row in cursor.fetchall()]

    # ========================================================================
    # Device Operations
    # ========================================================================

    def save_device(self, device_id: str, name: str, device_type: str, protocol: str, config: str, state: str) -> bool:
        """Save or update a device"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO devices (id, name, type, protocol, config, state, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (device_id, name, device_type, protocol, config, state, datetime.now()))
        return True

    def get_devices(self) -> List[Dict[str, Any]]:
        """List all devices"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM devices")
            return [dict(row) for row in cursor.fetchall()]

    # ========================================================================
    # User Operations
    # ========================================================================

    def create_user(self, user_id: str, username: str, password_hash: str, role: str = "user") -> bool:
        """Create a new user"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)",
                (user_id, username, password_hash, role)
            )
        return True

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========================================================================
    # Audit Operations
    # ========================================================================

    def log_audit(self, user: Optional[str], action: str, details: str, ip_address: Optional[str] = None) -> bool:
        """Log an audit entry"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO audit_log (user, action, details, ip_address) VALUES (?, ?, ?, ?)",
                (user, action, details, ip_address)
            )
        return True

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def shutdown(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
