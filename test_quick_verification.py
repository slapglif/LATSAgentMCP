#!/usr/bin/env python3
"""Quick verification of LATS improvements"""

import asyncio
from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools

async def quick_test():
    """Quick test to verify improvements are working"""
    
    agent = LATSAgent(max_depth=3)
    tools = create_filesystem_tools()
    
    # Simple security task
    result = await agent.investigate(
        task="Check for eval() usage in Python files",
        tools=tools,
        generate_report=True
    )
    
    print(f"\n✅ Completed: {result['completed']}")
    print(f"📊 Nodes: {result['nodes_explored']}")
    print(f"🌳 Depth: {result['max_depth']}")
    
    # Check if report mentions LATS correctly
    if 'report' in result:
        if 'Language Agent Tree Search' in result['report']:
            print("✅ Report correctly identifies LATS")
        if 'Log Aggregation' in result['report']:
            print("❌ Report still has wrong LATS acronym")
    
    return result

if __name__ == "__main__":
    print("Quick LATS Verification Test")
    print("=" * 40)
    result = asyncio.run(quick_test())
    print("\n✅ Test complete!")