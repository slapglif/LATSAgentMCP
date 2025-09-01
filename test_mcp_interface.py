#!/usr/bin/env python3
"""Test the MCP server interface properly"""

import asyncio
import sys
sys.path.append('.')

from mcp_server import mcp

async def test_mcp_server():
    """Test MCP server tools and interface"""
    
    print("🔧 Testing MCP Server Interface")
    print("=" * 50)
    
    # Test 1: List all tools
    print("\n📋 Step 1: Listing all registered tools")
    try:
        tools = await mcp.get_tools()
        print(f"✅ Successfully retrieved tools: {len(tools)} total")
        print(f"📋 Tools type: {type(tools)}")
        
        if isinstance(tools, list):
            for i, tool in enumerate(tools, 1):
                print(f"\n{i}. 🛠️ Tool: {tool}")
                print(f"   🔍 Type: {type(tool)}")
                if hasattr(tool, 'name'):
                    print(f"   📝 Name: {tool.name}")
                if hasattr(tool, 'description'):
                    print(f"   📝 Description: {tool.description}")
        else:
            print(f"🔍 Tools object: {tools}")
                    
    except Exception as e:
        print(f"❌ Failed to list tools: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Test each tool
    print(f"\n🧪 Step 2: Testing individual MCP tools")
    
    test_cases = [
        {
            "name": "quick_codebase_overview",
            "params": {},
            "description": "Get codebase overview"
        },
        {
            "name": "find_code_patterns", 
            "params": {"pattern": "class", "file_extension": "*.py"},
            "description": "Find class patterns"
        },
        {
            "name": "analyze_file_structure",
            "params": {"file_path": "lats_langgraph.py"},
            "description": "Analyze file structure"
        },
        {
            "name": "get_analysis_recommendations",
            "params": {"task": "understand architecture"},
            "description": "Get analysis recommendations"
        }
    ]
    
    for test_case in test_cases:
        tool_name = test_case["name"] 
        params = test_case["params"]
        desc = test_case["description"]
        
        print(f"\n🔍 Testing: {tool_name} - {desc}")
        try:
            result = await mcp._call_tool(tool_name, params)
            
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  ⚠️ Tool returned error: {result['error']}")
                else:
                    print(f"  ✅ Success! Keys: {list(result.keys())}")
                    
                    # Show interesting results
                    if "total_files" in result:
                        print(f"     📁 Files found: {result['total_files']}")
                    if "results" in result and isinstance(result["results"], str):
                        preview = result["results"][:200]
                        print(f"     📋 Result preview: {preview}{'...' if len(result['results']) > 200 else ''}")
                    if "recommended_config" in result:
                        config = result["recommended_config"]
                        print(f"     ⚙️ Recommended depth: {config.get('max_depth', 'N/A')}")
                        print(f"     📈 Strategy: {config.get('strategy', 'N/A')}")
            else:
                print(f"  ✅ Success! Result type: {type(result)}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Test 3: Test streaming tool if available
    print(f"\n📡 Step 3: Testing streaming capabilities")
    try:
        # Test analyze_codebase with streaming enabled
        print("🚀 Testing analyze_codebase with streaming...")
        result = await mcp._call_tool("analyze_codebase", {
            "task": "Quick test of streaming",
            "max_depth": 3,
            "streaming": True,
            "adaptive_mode": True
        })
        
        if isinstance(result, dict):
            print(f"  ✅ Streaming tool completed!")
            print(f"     📊 Duration: {result.get('duration_seconds', 0):.2f}s")
            print(f"     🔍 Nodes explored: {result.get('nodes_explored', 0)}")
            
            # Check for streaming progress
            progress = result.get('progress_updates', [])
            if progress:
                print(f"     📡 Progress events: {len(progress)}")
                event_types = {}
                for event in progress:
                    event_type = event.get('event_type', 'unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
                
                for event_type, count in event_types.items():
                    print(f"       • {event_type}: {count}")
            else:
                print(f"     ⚠️ No progress events captured")
                
    except Exception as e:
        print(f"  ❌ Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 MCP Interface Testing Complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())