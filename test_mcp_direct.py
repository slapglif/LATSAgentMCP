#!/usr/bin/env python3
"""Direct test of MCP server functionality"""

import asyncio
import sys
sys.path.insert(0, '/home/ubuntu/lats')

# Import the actual async functions, not the MCP tool wrappers
import mcp_server

async def test_mcp_tools():
    """Test all MCP tools directly"""
    print("=" * 80)
    print("TESTING MCP TOOLS DIRECTLY")
    print("=" * 80)
    
    # Test 1: Quick overview
    print("\n📋 Test 1: Quick Codebase Overview")
    try:
        # Get the actual function from the FunctionTool wrapper
        result = await mcp_server.quick_codebase_overview.fn(codebase_path=".")
        print(f"✅ Success: Found {result.get('total_files', 0)} files")
        if 'file_counts' in result:
            print(f"   File types: {result['file_counts']}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Find patterns
    print("\n📋 Test 2: Find Code Patterns")
    try:
        result = await mcp_server.find_code_patterns.fn(
            pattern="list_directory",
            file_extension="*.py"
        )
        if 'pagination' in result:
            print(f"✅ Success: Found {result['pagination']['total_items']} matches")
        else:
            print(f"✅ Success: Pattern search completed")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Analyze file structure
    print("\n📋 Test 3: Analyze File Structure")
    try:
        result = await mcp_server.analyze_file_structure.fn("lats_langgraph.py")
        print(f"✅ Success: Analyzed file structure")
        if 'structure' in result:
            print(f"   Structure preview: {result['structure'][:200]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Deep analysis with our improvements
    print("\n📋 Test 4: Deep Analysis (Testing Scoring Improvements)")
    try:
        result = await mcp_server.analyze_codebase.fn(
            task="Find security vulnerabilities related to eval() usage",
            generate_report=False,
            max_depth=3,
            streaming=False
        )
        print(f"✅ Success: Analysis completed")
        print(f"   Nodes explored: {result.get('nodes_explored', 'N/A')}")
        print(f"   Max depth: {result.get('max_depth', 'N/A')}")
        print(f"   Duration: {result.get('duration_seconds', 'N/A')}s")
        
        # Check if scoring improvements are working
        if 'tree' in result:
            print("\n   Checking scoring patterns:")
            tree = result['tree']
            if 'children' in tree:
                for child in tree['children'][:3]:
                    action = child.get('action', 'N/A')
                    score = child.get('value', 0)
                    if 'list_directory' in action:
                        print(f"     📁 {action}: {score} (should be ≤6 for security task)")
                    else:
                        print(f"     ⚡ {action}: {score}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Report generation
    print("\n📋 Test 5: Report Generation")
    try:
        result = await mcp_server.analyze_codebase.fn(
            task="Quick security check",
            generate_report=True,
            max_depth=2,
            streaming=False
        )
        if 'report' in result:
            print(f"✅ Success: Report generated")
            # Check for correct LATS acronym
            if 'Language Agent Tree Search' in result['report']:
                print("   ✅ Report correctly identifies LATS")
            if 'Log Aggregation' in result['report']:
                print("   ❌ Report still has wrong LATS acronym")
            
            # Save report for inspection
            with open('test_report_mcp.md', 'w') as f:
                f.write(result['report'])
            print("   📄 Report saved to test_report_mcp.md")
        else:
            print("⚠️ No report generated")
            
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 80)
    print("MCP TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())