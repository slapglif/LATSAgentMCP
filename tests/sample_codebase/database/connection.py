"""
Database connection module
Sample code with various patterns and issues
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager


class DatabaseError(Exception):
    """Database operation error"""
    pass


class ConnectionPool:
    """Simple connection pool"""
    
    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = []
        self.in_use = set()
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        # BUG: No proper connection reuse
        if len(self.connections) < self.max_connections:
            conn = sqlite3.connect(self.db_path)
            self.connections.append(conn)
            self.in_use.add(conn)
            return conn
        else:
            # BUG: Just returns first connection without checking if in use
            return self.connections[0]
    
    def release_connection(self, conn: sqlite3.Connection):
        """Release connection back to pool"""
        if conn in self.in_use:
            self.in_use.remove(conn)
    
    def close_all(self):
        """Close all connections"""
        for conn in self.connections:
            conn.close()
        self.connections.clear()
        self.in_use.clear()


class QueryBuilder:
    """Build SQL queries"""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.conditions = []
        self.order_by = None
        self.limit = None
    
    def where(self, column: str, operator: str, value: Any):
        """Add WHERE condition"""
        # BUG: SQL injection vulnerability - no parameterization
        self.conditions.append(f"{column} {operator} '{value}'")
        return self
    
    def order(self, column: str, direction: str = 'ASC'):
        """Add ORDER BY clause"""
        # BUG: No validation of direction
        self.order_by = f"{column} {direction}"
        return self
    
    def limit_to(self, count: int):
        """Add LIMIT clause"""
        self.limit = count
        return self
    
    def build_select(self, columns: List[str] = None) -> str:
        """Build SELECT query"""
        cols = '*' if columns is None else ', '.join(columns)
        query = f"SELECT {cols} FROM {self.table_name}"
        
        if self.conditions:
            query += " WHERE " + " AND ".join(self.conditions)
        
        if self.order_by:
            query += f" ORDER BY {self.order_by}"
        
        if self.limit:
            query += f" LIMIT {self.limit}"
        
        return query


class DatabaseManager:
    """Manage database operations"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.pool = ConnectionPool(db_path)
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup"""
        conn = self.pool.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            cursor.close()
            self.pool.release_connection(conn)
    
    def execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """Execute a query and return results"""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def insert_record(self, table: str, data: Dict[str, Any]) -> int:
        """Insert a record into table"""
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        # BUG: No validation of table name
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()))
            return cursor.lastrowid
    
    def update_record(self, table: str, data: Dict[str, Any], 
                     where: Dict[str, Any]) -> int:
        """Update records in table"""
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        
        # BUG: Empty where clause updates all records
        query = f"UPDATE {table} SET {set_clause}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        params = list(data.values()) + list(where.values())
        
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def delete_records(self, table: str, where: Dict[str, Any]) -> int:
        """Delete records from table"""
        # BUG: No protection against deleting all records
        where_clause = ' AND '.join([f"{k} = ?" for k in where.keys()])
        query = f"DELETE FROM {table}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(where.values()))
            return cursor.rowcount
    
    def create_table(self, table_name: str, schema: Dict[str, str]):
        """Create a new table"""
        columns = []
        for col_name, col_type in schema.items():
            # BUG: No validation of column types
            columns.append(f"{col_name} {col_type}")
        
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        
        with self.get_cursor() as cursor:
            cursor.execute(query)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.execute_query(query, (table_name,))
        return len(result) > 0