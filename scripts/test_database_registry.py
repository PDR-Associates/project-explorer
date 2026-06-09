#!/usr/bin/env python3
"""Test script for database registry operations."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from explorer.registry import DatabaseEntity, ProjectRegistry, ProjectStatus


def test_database_registry():
    """Test database entity registration and retrieval."""
    print("Testing Database Registry Operations\n")
    print("=" * 60)
    
    # Use a test database file
    registry = ProjectRegistry(db_path="data/test_registry.db")
    
    # Test 1: Register a database
    print("\n1. Registering a test PostgreSQL database...")
    db_entity = DatabaseEntity(
        slug="test-postgres",
        display_name="Test PostgreSQL",
        db_type="postgresql",
        host="localhost",
        port=5432,
        database_name="testdb",
        description="Test database for development",
    )
    
    try:
        registry.register_database(db_entity)
        print("   ✓ Database registered successfully")
    except Exception as e:
        print(f"   ✗ Failed to register: {e}")
        return False
    
    # Test 2: Retrieve the database
    print("\n2. Retrieving the database...")
    retrieved = registry.get_database("test-postgres")
    if retrieved:
        print(f"   ✓ Retrieved: {retrieved.display_name}")
        print(f"     Type: {retrieved.db_type}")
        print(f"     Host: {retrieved.host}:{retrieved.port}")
        print(f"     Database: {retrieved.database_name}")
    else:
        print("   ✗ Failed to retrieve database")
        return False
    
    # Test 3: List databases
    print("\n3. Listing all databases...")
    databases = registry.list_databases()
    print(f"   ✓ Found {len(databases)} database(s)")
    for db in databases:
        print(f"     - {db.slug}: {db.display_name} ({db.db_type})")
    
    # Test 4: Update database status
    print("\n4. Updating database status...")
    registry.update_database_status("test-postgres", ProjectStatus.ACTIVE)
    updated = registry.get_database("test-postgres")
    if updated and updated.status == ProjectStatus.ACTIVE:
        print(f"   ✓ Status updated to: {updated.status.value}")
    else:
        print("   ✗ Failed to update status")
        return False
    
    # Test 5: Record a survey
    print("\n5. Recording a survey result...")
    registry.record_database_survey(
        slug="test-postgres",
        schema_count=2,
        table_count=10,
        column_count=50,
        survey_data={
            "schemas": ["public", "app"],
            "surveyed_at": "2026-06-08T00:00:00",
        },
    )
    surveys = registry.get_database_surveys("test-postgres")
    if surveys:
        print(f"   ✓ Survey recorded: {len(surveys)} survey(s) in history")
        latest = surveys[0]
        print(f"     Schemas: {latest['schema_count']}")
        print(f"     Tables: {latest['table_count']}")
        print(f"     Columns: {latest['column_count']}")
    else:
        print("   ✗ Failed to record survey")
        return False
    
    # Test 6: Check if database exists
    print("\n6. Checking database existence...")
    exists = registry.database_exists("test-postgres")
    not_exists = registry.database_exists("nonexistent")
    if exists and not not_exists:
        print("   ✓ Existence check works correctly")
    else:
        print("   ✗ Existence check failed")
        return False
    
    # Test 7: Remove the database
    print("\n7. Removing the database...")
    registry.remove_database("test-postgres")
    removed = registry.get_database("test-postgres")
    if not removed:
        print("   ✓ Database removed successfully")
    else:
        print("   ✗ Failed to remove database")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = test_database_registry()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
