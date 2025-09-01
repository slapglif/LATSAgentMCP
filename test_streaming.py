#!/usr/bin/env python3
"""Test streaming functionality directly"""

import asyncio
from lats_langgraph import LATSAgent
from filesystem_tools import read_file_with_lines, list_directory_tree, search_in_files

async def test_streaming():
    """Test streaming with a simple task"""
    
    print("🧪 Testing LATS Streaming")
    print("=" * 50)
    
    # Create a simple progress callback
    progress_events = []
    
    async def progress_callback(progress_data):
        progress_events.append(progress_data)
        event_type = progress_data['event_type']
        data = progress_data['data']
        
        print(f"🔄 {event_type}: {data}")
        
        # Force output flush
        import sys
        sys.stdout.flush()
    
    # Initialize agent with streaming
    agent = LATSAgent(max_depth=5, progress_callback=progress_callback)
    
    # Simple tools
    tools = [read_file_with_lines, list_directory_tree, search_in_files]
    
    print("🚀 Starting streaming test...")
    
    try:
        result = await agent.investigate(
            "Find Python class definitions", 
            tools, 
            generate_report=False
        )
        
        print(f"\n✅ Investigation completed!")
        print(f"   Progress events captured: {len(progress_events)}")
        print(f"   Event types: {[e['event_type'] for e in progress_events]}")
        print(f"   Result: {result.get('best_action', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming())