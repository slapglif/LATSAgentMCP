#!/usr/bin/env python3
"""Debug LATS agent tree expansion issue"""

import asyncio
import sys
from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools

async def test_lats_simple():
    """Test LATS with a simple task"""
    print("=" * 80)
    print("Testing LATS Agent Tree Expansion")
    print("=" * 80)
    
    # Create agent with debug logging
    agent = LATSAgent(
        max_depth=5,
        log_file="debug_lats_expansion.log"
    )
    
    # Get tools
    tools = create_filesystem_tools()
    
    # Simple task
    task = "List the files in the current directory"
    
    print(f"\n📋 Task: {task}")
    print(f"🔧 Tools available: {len(tools)}")
    print(f"🌳 Max depth: {agent.max_depth}")
    
    try:
        # Run investigation
        result = await agent.investigate(
            task=task,
            tools=tools,
            generate_report=False  # Skip report generation for debugging
        )
        
        print("\n" + "=" * 40)
        print("INVESTIGATION RESULTS:")
        print("=" * 40)
        print(f"✅ Completed: {result['completed']}")
        print(f"📊 Nodes explored: {result['nodes_explored']}")
        print(f"🌳 Max depth reached: {result['max_depth']}")
        print(f"⏱️ Duration: {result['duration_seconds']:.2f}s")
        
        # Check tree structure
        if 'tree' in result:
            print(f"\n🌲 Tree structure:")
            import json
            print(json.dumps(result['tree'], indent=2))
        
        # Check if root has children
        root = result.get('root')
        if root:
            print(f"\n🌱 Root node children: {len(root.children) if hasattr(root, 'children') else 'N/A'}")
            if hasattr(root, 'children') and root.children:
                for i, child in enumerate(root.children[:3]):  # Show first 3 children
                    print(f"  Child {i+1}: {child.action}")
                    print(f"    - Depth: {child.depth}")
                    print(f"    - Has children: {len(child.children) > 0 if hasattr(child, 'children') else False}")
        
    except Exception as e:
        print(f"\n❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_lats_simple())