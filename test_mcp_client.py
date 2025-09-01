#!/usr/bin/env python3
"""Test MCP server through proper STDIO protocol"""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict, List

class MCPClient:
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.request_id = 0
    
    async def start(self):
        """Start the MCP server process"""
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Initialize the connection
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })
        
        # Wait for initialize response
        response = await self._read_response()
        if not response or "result" not in response:
            raise Exception(f"Failed to initialize: {response}")
        
        # Send initialized notification
        await self._send_notification("initialized", {})
        
        print("✅ MCP client connected successfully")
        
    async def _send_request(self, method: str, params: Dict[str, Any]) -> int:
        """Send a JSON-RPC request"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
        return self.request_id
    
    async def _send_notification(self, method: str, params: Dict[str, Any]):
        """Send a JSON-RPC notification"""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
    
    async def _read_response(self) -> Dict[str, Any]:
        """Read a response from the server"""
        try:
            line = await self.process.stdout.readline()
            if not line:
                return None
            
            return json.loads(line.decode().strip())
        except Exception as e:
            print(f"Failed to read response: {e}")
            return None
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        request_id = await self._send_request("tools/list", {})
        response = await self._read_response()
        
        if response and "result" in response and "tools" in response["result"]:
            return response["result"]["tools"]
        return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool"""
        request_id = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        response = await self._read_response()
        return response
    
    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()

async def test_mcp_protocol():
    """Test the MCP server through proper protocol"""
    
    print("🔧 Testing MCP Server via STDIO Protocol")
    print("=" * 60)
    
    client = MCPClient(["python", "mcp_server.py"])
    
    try:
        # Start the server
        print("\n📡 Step 1: Connecting to MCP server...")
        await client.start()
        
        # List tools
        print("\n📋 Step 2: Listing available tools...")
        tools = await client.list_tools()
        
        print(f"✅ Found {len(tools)} tools:")
        for i, tool in enumerate(tools, 1):
            name = tool.get("name", "Unknown")
            desc = tool.get("description", "No description")
            print(f"  {i}. 🛠️ {name}")
            print(f"     📝 {desc[:100]}{'...' if len(desc) > 100 else ''}")
        
        # Test individual tools
        print(f"\n🧪 Step 3: Testing tool functionality...")
        
        test_cases = [
            {
                "name": "quick_codebase_overview",
                "args": {},
                "description": "Get codebase overview"
            },
            {
                "name": "find_code_patterns",
                "args": {"pattern": "class", "file_extension": "*.py"},
                "description": "Find class patterns" 
            },
            {
                "name": "analyze_file_structure",
                "args": {"file_path": "lats_langgraph.py"},
                "description": "Analyze file structure"
            },
            {
                "name": "get_analysis_recommendations", 
                "args": {"task": "understand architecture"},
                "description": "Get recommendations"
            }
        ]
        
        for test_case in test_cases:
            tool_name = test_case["name"]
            args = test_case["args"] 
            desc = test_case["description"]
            
            print(f"\n🔍 Testing: {tool_name}")
            print(f"   📋 {desc}")
            print(f"   📥 Args: {args}")
            
            try:
                response = await client.call_tool(tool_name, args)
                
                if response and "result" in response:
                    result = response["result"]
                    print(f"   ✅ Success!")
                    
                    # Show key results
                    if isinstance(result, dict):
                        if "error" in result:
                            print(f"     ⚠️ Tool error: {result['error']}")
                        else:
                            # Show interesting fields
                            for key in ["total_files", "file_counts", "recommended_config", "structure"]:
                                if key in result:
                                    value = result[key]
                                    if key == "structure" and isinstance(value, str):
                                        print(f"     📁 Structure: {len(value)} chars")
                                    elif key == "recommended_config":
                                        if isinstance(value, dict):
                                            depth = value.get("max_depth", "N/A")
                                            strategy = value.get("strategy", "N/A")
                                            print(f"     ⚙️ Recommended: depth={depth}, strategy={strategy}")
                                    else:
                                        print(f"     📊 {key}: {value}")
                    
                elif response and "error" in response:
                    error = response["error"]
                    print(f"   ❌ RPC Error: {error.get('message', 'Unknown error')}")
                    
                else:
                    print(f"   ❌ Unexpected response: {response}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Test streaming tool
        print(f"\n📡 Step 4: Testing streaming analysis...")
        print("🚀 Running analyze_codebase_adaptive...")
        
        try:
            response = await client.call_tool("analyze_codebase_adaptive", {
                "task": "Quick test analysis",
                "urgency": "high",
                "focus": "focused",
                "streaming": True
            })
            
            if response and "result" in response:
                result = response["result"]
                print(f"   ✅ Streaming analysis completed!")
                
                duration = result.get("duration_seconds", 0)
                nodes = result.get("nodes_explored", 0)
                depth = result.get("max_depth", 0)
                
                print(f"     ⏱️ Duration: {duration:.2f}s")
                print(f"     🔍 Nodes explored: {nodes}")
                print(f"     📏 Max depth: {depth}")
                
                # Check adaptive config
                adaptive_config = result.get("adaptive_config", {})
                if adaptive_config:
                    print(f"     🧠 Adaptive config applied:")
                    print(f"        • Urgency: {adaptive_config.get('urgency')}")
                    print(f"        • Focus: {adaptive_config.get('focus')}")
                    
            else:
                print(f"   ❌ Streaming test failed: {response}")
                
        except Exception as e:
            print(f"   ❌ Streaming exception: {e}")
        
    finally:
        await client.stop()
    
    print(f"\n🎉 MCP Protocol Testing Complete!")

if __name__ == "__main__":
    asyncio.run(test_mcp_protocol())