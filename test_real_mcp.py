#!/usr/bin/env python3
"""Complete MCP server test through real STDIO protocol"""

import asyncio
import json
import subprocess
import sys

async def test_mcp_server_properly():
    """Test MCP server with proper protocol implementation"""
    
    print("🚀 COMPREHENSIVE MCP SERVER TESTING")
    print("=" * 60)
    
    # Start MCP server process
    print("\n📡 Step 1: Starting MCP server...")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "mcp_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    async def send_message(message_dict):
        """Send a JSON-RPC message"""
        message = json.dumps(message_dict) + "\n"
        process.stdin.write(message.encode())
        await process.stdin.drain()
    
    async def read_response():
        """Read a response from the server"""
        line = await process.stdout.readline()
        if line:
            return json.loads(line.decode().strip())
        return None
    
    try:
        # Step 2: Initialize connection
        print("📋 Step 2: Initializing MCP connection...")
        await send_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        })
        
        init_response = await read_response()
        if init_response and "result" in init_response:
            print("✅ MCP connection initialized successfully")
            capabilities = init_response["result"].get("capabilities", {})
            print(f"   Server capabilities: {list(capabilities.keys())}")
        else:
            print(f"❌ Initialization failed: {init_response}")
            return False
        
        # Send initialized notification
        await send_message({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
        
        # Step 3: List tools
        print("\n🛠️ Step 3: Listing available tools...")
        await send_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        })
        
        tools_response = await read_response()
        if tools_response and "result" in tools_response:
            tools = tools_response["result"].get("tools", [])
            print(f"✅ Found {len(tools)} tools:")
            
            for i, tool in enumerate(tools, 1):
                name = tool.get("name", "Unknown")
                desc = tool.get("description", "No description")[:80]
                print(f"   {i}. {name}: {desc}{'...' if len(tool.get('description', '')) > 80 else ''}")
                
                # Show input schema if available
                input_schema = tool.get("inputSchema", {})
                if input_schema and "properties" in input_schema:
                    params = list(input_schema["properties"].keys())
                    print(f"      Parameters: {', '.join(params[:5])}{'...' if len(params) > 5 else ''}")
        else:
            print(f"❌ Failed to list tools: {tools_response}")
            return False
        
        # Step 4: Test individual tools
        print("\n🧪 Step 4: Testing MCP tools...")
        
        test_cases = [
            {
                "id": 3,
                "tool": "quick_codebase_overview",
                "args": {},
                "desc": "Quick codebase overview"
            },
            {
                "id": 4, 
                "tool": "find_code_patterns",
                "args": {"pattern": "def ", "file_extension": "*.py"},
                "desc": "Find function definitions"
            },
            {
                "id": 5,
                "tool": "analyze_file_structure", 
                "args": {"file_path": "mcp_server.py"},
                "desc": "Analyze MCP server file"
            }
        ]
        
        for test_case in test_cases:
            tool_name = test_case["tool"]
            args = test_case["args"]
            desc = test_case["desc"]
            test_id = test_case["id"]
            
            print(f"\n🔍 Testing: {tool_name}")
            print(f"   📋 {desc}")
            print(f"   📥 Arguments: {args}")
            
            await send_message({
                "jsonrpc": "2.0",
                "id": test_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": args
                }
            })
            
            response = await read_response()
            
            if response and "result" in response:
                result = response["result"]
                print(f"   ✅ Success!")
                
                # Show key results based on tool type
                if "content" in result and isinstance(result["content"], list):
                    for content_item in result["content"][:2]:  # Show first 2 items
                        if content_item.get("type") == "text":
                            text = content_item.get("text", "")[:200]
                            print(f"      📄 Content: {text}{'...' if len(text) >= 200 else ''}")
                
            elif response and "error" in response:
                error = response["error"]
                print(f"   ❌ Tool Error: {error.get('message', 'Unknown error')}")
                if "data" in error and error["data"]:
                    print(f"      Details: {error['data']}")
            else:
                print(f"   ❌ Unexpected response: {response}")
        
        # Step 5: Test streaming tool
        print(f"\n📡 Step 5: Testing streaming analysis...")
        print("🚀 Running analyze_codebase with streaming...")
        
        await send_message({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "analyze_codebase",
                "arguments": {
                    "task": "Quick streaming test",
                    "generate_report": False,
                    "max_depth": 3,
                    "streaming": True
                }
            }
        })
        
        # Read response (this may take a while for LATS analysis)
        print("   ⏳ Waiting for analysis to complete...")
        response = await asyncio.wait_for(read_response(), timeout=60)
        
        if response and "result" in response:
            result = response["result"]
            print(f"   ✅ Streaming analysis completed!")
            
            # Check for result content
            if "content" in result and isinstance(result["content"], list):
                for content_item in result["content"]:
                    if content_item.get("type") == "text":
                        text = content_item.get("text", "")
                        if "duration_seconds" in text:
                            print(f"      ⏱️ Found duration information")
                        if "nodes_explored" in text:
                            print(f"      🔍 Found exploration metrics")
                        if "best_score" in text:
                            print(f"      📊 Found scoring information")
                            
                print(f"      📋 Analysis result received with {len(result['content'])} content items")
            else:
                print(f"      📋 Result: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                
        elif response and "error" in response:
            error = response["error"]
            print(f"   ❌ Streaming Error: {error.get('message', 'Unknown error')}")
        else:
            print(f"   ❌ No response or timeout")
        
        print(f"\n🎉 MCP Server Testing Complete!")
        return True
        
    except asyncio.TimeoutError:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        process.terminate()
        await process.wait()

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server_properly())
    sys.exit(0 if success else 1)