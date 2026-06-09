"""Database connection abstraction for different database types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from explorer.registry import DatabaseEntity


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""

    @abstractmethod
    def connect(self) -> Any:
        """Establish connection to the database."""

    @abstractmethod
    def execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""

    @abstractmethod
    def get_schema_info(self) -> dict:
        """Get database schema information (schemas, tables, columns)."""

    @abstractmethod
    def get_statistics(self) -> dict:
        """Get database statistics (row counts, sizes, etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL-specific connection implementation."""

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> Any:
        """Establish PostgreSQL connection."""
        try:
            import psycopg2
        except ImportError as e:
            raise ImportError(
                "psycopg2 is required for PostgreSQL connections. "
                "Install it with: pip install psycopg2-binary"
            ) from e

        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        return self._conn

    def execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        if not self._conn:
            raise RuntimeError("Not connected to database")

        with self._conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
            return []

    def get_schema_info(self) -> dict:
        """Get PostgreSQL schema information."""
        # Query information_schema for schemas (excluding system schemas)
        schemas_query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
            ORDER BY schema_name
        """
        schemas = self.execute_query(schemas_query)

        result = {"schemas": [], "total_tables": 0, "total_columns": 0}
        for schema in schemas:
            schema_name = schema["schema_name"]
            tables = self._get_tables_for_schema(schema_name)
            result["schemas"].append({"name": schema_name, "tables": tables})
            result["total_tables"] += len(tables)
            result["total_columns"] += sum(len(t["columns"]) for t in tables)

        return result

    def _get_tables_for_schema(self, schema_name: str) -> list[dict]:
        """Get tables and columns for a schema."""
        query = """
            SELECT 
                t.table_name,
                t.table_type,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.ordinal_position
            FROM information_schema.tables t
            LEFT JOIN information_schema.columns c 
                ON t.table_name = c.table_name 
                AND t.table_schema = c.table_schema
            WHERE t.table_schema = %s
            ORDER BY t.table_name, c.ordinal_position
        """
        rows = self.execute_query(query, (schema_name,))

        # Group by table
        tables: dict[str, dict] = {}
        for row in rows:
            table_name = row["table_name"]
            if table_name not in tables:
                tables[table_name] = {
                    "name": table_name,
                    "type": row["table_type"],
                    "columns": [],
                }
            if row["column_name"]:
                tables[table_name]["columns"].append(
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"],
                        "position": row["ordinal_position"],
                    }
                )

        return list(tables.values())

    def get_statistics(self) -> dict:
        """Get database statistics."""
        stats = {
            "database_size": self._get_database_size(),
            "table_stats": self._get_table_statistics(),
        }
        return stats

    def _get_database_size(self) -> dict:
        """Get database size information."""
        query = """
            SELECT 
                pg_database_size(current_database()) as size_bytes,
                pg_size_pretty(pg_database_size(current_database())) as size_pretty
        """
        result = self.execute_query(query)
        return result[0] if result else {"size_bytes": 0, "size_pretty": "0 bytes"}

    def _get_table_statistics(self) -> list[dict]:
        """Get statistics for all tables."""
        query = """
            SELECT 
                schemaname,
                tablename,
                pg_total_relation_size(schemaname||'.'||tablename) as total_bytes,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY total_bytes DESC
            LIMIT 100
        """
        return self.execute_query(query)

    def close(self) -> None:
        """Close the PostgreSQL connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


@contextmanager
def database_connection(db_entity: DatabaseEntity, credentials: dict):
    """Context manager for database connections.
    
    Args:
        db_entity: DatabaseEntity with connection details
        credentials: Dict with 'user' and 'password' keys
        
    Yields:
        DatabaseConnection instance
        
    Example:
        with database_connection(db_entity, {"user": "admin", "password": "secret"}) as conn:
            schema = conn.get_schema_info()
    """
    if db_entity.db_type == "postgresql":
        conn = PostgreSQLConnection(
            host=db_entity.host,
            port=db_entity.port,
            database=db_entity.database_name,
            user=credentials["user"],
            password=credentials["password"],
        )
    else:
        raise ValueError(f"Unsupported database type: {db_entity.db_type}")

    try:
        conn.connect()
        yield conn
    finally:
        conn.close()

# Made with Bob
