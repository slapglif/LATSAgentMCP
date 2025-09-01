#!/usr/bin/env python3
"""
Raw LATS test - shows all real agent outputs, tool calls, and findings
No sanitized messages, just the raw agent working
"""

import asyncio
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
# Removed test mode - using real Ollama

from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools


async def run_raw_lats():
    """Run LATS and show ALL raw outputs"""
    
    # Change to sample codebase
    sample_dir = Path(__file__).parent / 'sample_codebase'
    os.chdir(sample_dir)
    
    print("RAW LATS AGENT TEST - SHOWING ALL OUTPUTS")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print("Task: Find authentication bugs")
    print()
    
    # Initialize LATS agent
    agent = LATSAgent(max_depth=3, max_iterations=8)
    tools = create_filesystem_tools()
    
    task = "Find authentication bugs and security vulnerabilities"
    
    print("STARTING LATS INVESTIGATION...")
    print("-" * 60)
    
    # Run the investigation with all outputs
    result = await agent.investigate(task, tools)
    
    print("\n" + "=" * 60)
    print("RAW RESULTS:")
    print("=" * 60)
    
    print(f"Task: {result['task']}")
    print(f"Completed: {result['completed']}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Max depth: {result['max_depth']}")
    print(f"Best score: {result['best_score']}")
    
    print(f"\nBest action: {result['best_action']}")
    print(f"\nBest observation (first 1000 chars):")
    print("-" * 40)
    if result['best_observation']:
        print(result['best_observation'][:1000])
        if len(result['best_observation']) > 1000:
            print("...")
    
    print(f"\nFULL TREE STRUCTURE:")
    print("-" * 40)
    _print_full_tree(result['tree'], 0)
    
    return result


def _print_full_tree(node, depth, max_depth=10):
    """Print full tree with all observations"""
    if depth > max_depth:
        return
    
    indent = "  " * depth
    
    if node.get("action"):
        print(f"{indent}ACTION: {node['action']}")
        print(f"{indent}SCORE: {node.get('value', 0):.1f}, VISITS: {node.get('visits', 0)}")
        
        if node.get("observation"):
            obs = node["observation"]
            print(f"{indent}OBSERVATION:")
            # Show first 500 chars of observation
            obs_lines = obs.split('\n')[:20]  # First 20 lines
            for line in obs_lines:
                print(f"{indent}  {line}")
            if len(obs.split('\n')) > 20:
                print(f"{indent}  ...")
        print()
    else:
        print(f"{indent}ROOT NODE")
        print(f"{indent}SCORE: {node.get('value', 0):.1f}, VISITS: {node.get('visits', 0)}")
        print()
    
    for child in node.get("children", []):
        if not child.get("truncated"):
            _print_full_tree(child, depth + 1, max_depth)


async def show_manual_tool_calls():
    """Show what the tools actually return"""
    
    print("\n" + "=" * 60)
    print("MANUAL TOOL EXECUTION EXAMPLES")
    print("=" * 60)
    
    from filesystem_tools import read_file_with_lines, search_in_files, list_directory_tree
    
    print("1. READING auth/login.py:")
    print("-" * 30)
    result = read_file_with_lines('auth/login.py')
    lines = result.split('\n')[:30]  # First 30 lines
    for line in lines:
        print(line)
    print()
    
    print("2. SEARCHING for 'password':")
    print("-" * 30)
    result = search_in_files('password', '.', '*.py', case_sensitive=False)
    print(result)
    print()
    
    print("3. SEARCHING for 'BUG':")
    print("-" * 30)
    result = search_in_files('BUG', '.', '*.py', case_sensitive=False)
    print(result)
    print()
    
    print("4. DIRECTORY STRUCTURE:")
    print("-" * 30)
    result = list_directory_tree('.', max_depth=2)
    print(result)


async def main():
    """Run raw LATS test"""
    
    try:
        # Run LATS and show all outputs
        result = await run_raw_lats()
        
        # Show manual tool examples
        await show_manual_tool_calls()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"LATS explored {result['nodes_explored']} nodes")
        print(f"Achieved {result['best_score']:.1f}/10 confidence")
        print(f"Investigation {'COMPLETED' if result['completed'] else 'PARTIAL'}")
        
        # Verify real execution
        has_real_content = False
        if result['best_observation']:
            # Check if observation contains actual file content or search results
            obs = result['best_observation'].lower()
            if any(keyword in obs for keyword in ['def ', 'import ', 'class ', 'found ', 'matches']):
                has_real_content = True
        
        if has_real_content and result['nodes_explored'] > 3:
            print("\n✅ VERIFIED: Real tool execution and exploration")
            return 0
        else:
            print("\n❌ FAILED: Not enough real content found")
            return 1
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)