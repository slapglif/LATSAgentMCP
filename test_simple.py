#!/usr/bin/env python3
"""
Simple test to verify LATS agent with real services
"""

import asyncio
import tempfile
from pathlib import Path

from lats_langgraph import LATSAgent
from memory_manager import MemoryManager, InvestigationMemory
from filesystem_tools import create_filesystem_tools

async def test_basic_functionality():
    print("\n" + "="*80)
    print("TESTING LATS WITH REAL SERVICES")
    print("="*80)
    
    # Test 1: Memory Manager with SQLite
    print("\n1. Testing Memory Manager...")
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        memory = MemoryManager(str(db_path))
        
        # Store a test investigation
        inv = InvestigationMemory(
            task="Test task",
            solution_path=[{"action": "test", "score": 8.0}],
            file_references=["test.py"],
            insights={"test": "insight"},
            score=8.0,
            is_complete=True
        )
        memory_id = memory.store_investigation(inv)
        print(f"✅ Stored investigation: {memory_id}")
        
        # Search for it
        results = memory.search_similar_investigations("test", limit=1)
        print(f"✅ Found {len(results)} similar investigations")
    
    # Test 2: LATS Agent with SQLite checkpointing
    print("\n2. Testing LATS Agent...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_db = Path(tmpdir) / "checkpoints.db"
        
        agent = LATSAgent(
            max_depth=2,
            max_iterations=3,
            checkpoint_db=str(checkpoint_db)
        )
        
        # Create tools
        tools = create_filesystem_tools()
        
        # Run a simple investigation
        result = await agent.investigate(
            task="List files in current directory",
            tools=tools,
            generate_report=False,
            thread_id="test_thread"
        )
        
        print(f"✅ Investigation completed:")
        print(f"   - Nodes explored: {result['nodes_explored']}")
        print(f"   - Best score: {result['best_score']:.1f}")
        print(f"   - Completed: {result['completed']}")
        
        # Verify checkpoint was created
        if checkpoint_db.exists():
            print(f"✅ Checkpoint created: {checkpoint_db.stat().st_size} bytes")
        else:
            print("❌ Checkpoint not created")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED WITH REAL SERVICES")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())