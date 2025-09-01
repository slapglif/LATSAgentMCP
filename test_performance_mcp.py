#!/usr/bin/env python3
"""Test performance analysis via MCP interface"""

import asyncio
import sys
sys.path.append('.')

# Import the actual function implementation, not the MCP wrapper
from lats_langgraph import LATSAgent
from filesystem_tools import read_file_with_lines, list_directory_tree, search_in_files, analyze_code_structure, find_dependencies

async def test_performance_analysis():
    """Test performance analysis with the enhanced MCP"""
    print("🚀 Testing Performance Analysis via MCP")
    print("=" * 60)
    
    try:
        # Create progress callback to capture streaming updates
        progress_updates = []
        
        async def progress_callback(progress_data):
            progress_updates.append(progress_data)
            event_type = progress_data['event_type']
            data = progress_data['data']
            
            # Print key progress events
            if event_type == "investigation_started":
                print(f"🚀 Investigation started: {data.get('task', '')}")
            elif event_type == "context_management":
                action = data.get('action', '')
                if action == "chunking_large_content":
                    print(f"📊 Chunking large content: {data.get('estimated_tokens', 0)} tokens")
                elif action == "context_summarized":
                    print(f"💾 Context summarized: {data.get('summary_length', 0)} chars")
            elif event_type == "node_selected":
                print(f"🎯 Node selected: {data.get('action', 'N/A')}")
            elif event_type == "action_executing":
                print(f"⚡ Executing: {data.get('action', 'Unknown')}")
            elif event_type == "investigation_completed":
                print(f"🏁 Investigation completed in {data.get('duration_seconds', 0):.2f}s")
        
        # Initialize LATS agent with context management
        agent = LATSAgent(
            max_depth=6,  # Moderate depth for testing
            progress_callback=progress_callback,
            max_context_tokens=25000  # Test context management
        )
        
        # Tools for analysis
        tools = [read_file_with_lines, list_directory_tree, search_in_files, analyze_code_structure, find_dependencies]
        
        # Run performance analysis
        result = await agent.investigate(
            "Identify performance bottlenecks and optimization opportunities",
            tools,
            generate_report=True
        )
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
            
        print(f"\n🎉 Analysis Complete!")
        print(f"   Task: {result.get('task', 'N/A')}")
        print(f"   Duration: {result.get('duration_seconds', 0):.2f}s")
        print(f"   Nodes explored: {result.get('nodes_explored', 0)}")
        print(f"   Max depth: {result.get('max_depth', 0)}")
        print(f"   Best score: {result.get('best_score', 0):.2f}")
        print(f"   Completed: {result.get('completed', False)}")
        
        # Show agent configuration
        print(f"\n⚙️ Agent Configuration:")
        print(f"   Max depth: {agent.max_depth}")
        print(f"   Max context tokens: {agent.max_context_tokens}")
        print(f"   Current context size: {agent.current_context_size}")
        
        # Show streaming progress
        if progress_updates:
            print(f"\n📡 Streaming Progress:")
            print(f"   Total progress events: {len(progress_updates)}")
            event_types = {}
            for update in progress_updates:
                event_type = update.get('event_type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            for event_type, count in event_types.items():
                print(f"   {event_type}: {count} events")
                
        # Check if context management was triggered
        context_events = [u for u in progress_updates if u.get('event_type') == 'context_management']
        if context_events:
            print(f"\n💾 Context Management:")
            for event in context_events:
                action = event['data'].get('action', 'unknown')
                print(f"   {action}: {event['data']}")
        
        # Show memory usage if available
        if hasattr(agent, 'memory_manager') and agent.memory_manager:
            print(f"\n🧠 Memory Manager: Available")
        else:
            print(f"\n🧠 Memory Manager: Not initialized")
                
        # Show best findings
        best_action = result.get('best_action', 'N/A')
        best_obs = result.get('best_observation', 'N/A')
        if best_obs and best_obs != 'N/A':
            print(f"\n🏆 Best Finding:")
            print(f"   Action: {best_action}")
            print(f"   Observation: {best_obs[:300]}{'...' if len(str(best_obs)) > 300 else ''}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_performance_analysis())