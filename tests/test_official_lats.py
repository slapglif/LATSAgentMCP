#!/usr/bin/env python3
"""
Test using official LangGraph LATS implementation
Agent explores and discovers vulnerabilities autonomously
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


async def test_real_lats_exploration():
    """Test real LATS agent with autonomous exploration"""
    
    # Change to sample codebase - agent starts here with NO prior knowledge
    sample_dir = Path(__file__).parent / 'sample_codebase'
    os.chdir(sample_dir)
    
    print("=" * 80)
    print("🤖 OFFICIAL LANGGRAPH LATS INVESTIGATION")
    print("=" * 80)
    print(f"📍 Working directory: {os.getcwd()}")
    print("🧠 Agent knowledge: NONE - must discover everything")
    print("🎯 Task: Find authentication bugs and security vulnerabilities")
    print("-" * 80)
    
    # Initialize official LATS agent
    agent = LATSAgent(max_depth=4, max_iterations=15)
    tools = create_filesystem_tools()
    
    print(f"\n🚀 Agent initialized with {len(tools)} tools available")
    print("🌳 Starting autonomous tree search...")
    
    task = "Find all authentication bugs and security vulnerabilities in this codebase"
    
    # Run the investigation
    result = await agent.investigate(task, tools)
    
    print(f"\n{'='*80}")
    print("🏁 INVESTIGATION RESULTS")  
    print(f"{'='*80}")
    
    print(f"✅ Status: {'COMPLETED' if result['completed'] else 'IN PROGRESS'}")
    print(f"🌳 Nodes explored: {result['nodes_explored']}")
    print(f"📏 Max depth reached: {result['max_depth']}")
    print(f"🏆 Best score: {result['best_score']:.1f}/10")
    
    if result['best_action']:
        print(f"\n🎯 BEST DISCOVERY:")
        print(f"   Action: {result['best_action']}")
        if result['best_observation']:
            obs = result['best_observation'][:300] + "..." if len(result['best_observation']) > 300 else result['best_observation']
            print(f"   Finding: {obs}")
    
    # Show tree exploration
    print(f"\n🌳 EXPLORATION TREE:")
    print("-" * 50)
    _print_tree(result['tree'], indent=0)
    
    return result


def _print_tree(node_data, indent=0, max_indent=6):
    """Print tree structure"""
    if indent > max_indent:
        return
    
    prefix = "  " * indent
    if node_data.get("action"):
        action = node_data["action"][:60] + "..." if len(node_data["action"]) > 60 else node_data["action"]
        print(f"{prefix}🔸 {action}")
        print(f"{prefix}  📊 Score: {node_data.get('value', 0):.1f}, Visits: {node_data.get('visits', 0)}")
        
        if node_data.get("observation"):
            obs = node_data["observation"][:100] + "..." if len(node_data["observation"]) > 100 else node_data["observation"]
            print(f"{prefix}  💡 {obs}")
    else:
        print(f"{prefix}🌱 ROOT")
    
    for child in node_data.get("children", []):
        if not child.get("truncated"):
            _print_tree(child, indent + 1, max_indent)


async def demonstrate_lats_steps():
    """Show LATS algorithm steps in detail"""
    
    print(f"\n{'='*80}")
    print("🔬 LATS ALGORITHM DEMONSTRATION")
    print(f"{'='*80}")
    print("\nThis shows how the official LangGraph LATS works:")
    print()
    
    steps = [
        "1. 🌱 INITIALIZE: Create root node with task",
        "2. 🎯 SELECT: Choose most promising node (highest UCT score)",
        "3. 🌿 EXPAND: Generate N possible actions from selected node", 
        "4. ⚡ SIMULATE: Execute each action using real tools",
        "5. 🤔 REFLECT: Evaluate action quality (1-10 score)",
        "6. ⬆️ BACKPROPAGATE: Update parent nodes with scores",
        "7. 🔄 REPEAT: Continue until solution found or max depth",
        "8. 🏆 SOLUTION: Return best path through tree"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print("\n🧮 UCT SCORING FORMULA:")
    print("   UCT = exploitation + exploration")
    print("   UCT = (value/visits) + c * sqrt(ln(parent_visits)/visits)")
    print("   where c = exploration constant (1.414)")
    
    print("\n🎯 AGENT CAPABILITIES:")
    capabilities = [
        "✅ Autonomous file discovery (no pre-provided files)",
        "✅ Real tool execution (filesystem operations)",
        "✅ LLM-powered action generation", 
        "✅ Monte Carlo tree search",
        "✅ Quality-based reflection and scoring",
        "✅ Systematic exploration with UCT selection",
        "✅ Backpropagation of learning through tree"
    ]
    
    for cap in capabilities:
        print(f"   {cap}")


async def main():
    """Run comprehensive LATS demonstration"""
    
    print("🚀 Testing Official LangGraph LATS Implementation")
    print("📝 Agent will autonomously explore the codebase and find vulnerabilities")
    print()
    
    try:
        # Run the real investigation
        result = await test_real_lats_exploration()
        
        # Show how LATS works
        await demonstrate_lats_steps()
        
        print(f"\n{'='*80}")
        print("🎊 DEMONSTRATION COMPLETE")
        print(f"{'='*80}")
        
        if result['completed']:
            print("✅ SUCCESS: Agent found high-confidence vulnerabilities!")
        elif result['best_score'] >= 7.0:
            print("✅ PARTIAL SUCCESS: Agent found potential security issues")  
        else:
            print("⚠️ EXPLORATION: Agent gathered information for analysis")
        
        print(f"🌳 Tree contained {result['nodes_explored']} nodes")
        print(f"🏆 Best discovery scored {result['best_score']:.1f}/10")
        
        # Success criteria
        if result['nodes_explored'] >= 5 and result['best_score'] >= 6.0:
            print("\n🎉 LATS algorithm is working correctly!")
            print("✅ Real exploration, real inference, real tree search")
            return 0
        else:
            print("\n⚠️ Investigation could be improved")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error during LATS execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)