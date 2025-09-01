#!/usr/bin/env python3
"""Debug MCP protocol issues"""

import asyncio
import json
import subprocess
import sys

async def debug_mcp():
    print("🔧 Debugging MCP Protocol")
    print("=" * 40)
    
    # Start server
    process = await asyncio.create_subprocess_exec(
        sys.executable, "mcp_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    async def send_and_read(message_dict):
        message = json.dumps(message_dict) + "\n"
        process.stdin.write(message.encode())
        await process.stdin.drain()
        
        line = await process.stdout.readline()
        if line:
            return json.loads(line.decode().strip())
        return None
    
    try:
        # Initialize
        print("1. Initializing...")
        init_response = await send_and_read({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        })
        print(f"Init result: {init_response}")
        
        # Send initialized
        await send_and_read({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        })
        
        # Try different tools/list formats
        print("\n2. Testing tools/list variations...")
        
        variations = [
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 4, "method": "list_tools", "params": {}},
        ]
        
        for i, variation in enumerate(variations):
            print(f"  Trying variation {i+1}: {variation['method']}")
            response = await send_and_read(variation)
            print(f"  Response: {response}")
            
            if response and "result" in response:
                print("  ✅ Success!")
                break
            elif response and "error" in response:
                print(f"  ❌ Error: {response['error']['message']}")
        
        # Try calling a tool directly
        print("\n3. Testing direct tool call...")
        tool_response = await send_and_read({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "quick_codebase_overview",
                "arguments": {}
            }
        })
        print(f"Tool call response: {tool_response}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        await process.wait()

if __name__ == "__main__":
    asyncio.run(debug_mcp())