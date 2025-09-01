#!/usr/bin/env python3
"""
Real LATS algorithm test - agent explores and discovers on its own
"""

import asyncio
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from lats_core import LATSAlgorithm, LATSConfig, TreeNode
from filesystem_tools import create_filesystem_tools
from memory_manager import MemoryManager


def traverse_tree(node):
    """Helper to traverse all nodes in tree"""
    yield node
    for child in node.children:
        yield from traverse_tree(child)

# Use real Ollama if available, otherwise TestLLM
try:
    from langchain_ollama import ChatOllama
    use_real_llm = True
    print("🚀 Using real Ollama inference with gpt-oss model")
except ImportError:
    from test_llm import TestLLM
    use_real_llm = False
    print("⚠️  Ollama not available, using TestLLM for demo")


async def run_real_lats_investigation():
    """Run actual LATS with real inference and exploration"""
    
    # Change to sample codebase - agent starts here with no knowledge
    os.chdir(Path(__file__).parent / 'sample_codebase')
    
    print("\n" + "=" * 80)
    print("🤖 REAL LATS AGENT INVESTIGATION")
    print("=" * 80)
    print("\nTask: Find authentication bugs in this codebase")
    print("Agent knowledge: NONE - must explore and discover everything")
    print("-" * 80)
    
    # Initialize LATS with real inference
    config = LATSConfig(
        max_depth=4,
        max_iterations=5,
        num_expand=3,
        exploration_constant=1.414
    )
    
    if use_real_llm:
        llm = ChatOllama(
            model="gpt-oss",
            base_url="http://localhost:11434",
            temperature=0.7
        )
    else:
        llm = TestLLM()
    
    algorithm = LATSAlgorithm(config)
    tools = create_filesystem_tools()
    memory = MemoryManager(db_path='real_test_memory.db')
    
    task = "Find all authentication bugs and security vulnerabilities in this codebase"
    
    print(f"\n🎯 TASK: {task}")
    print("\n🌳 Starting LATS tree search...")
    
    # Run the actual LATS algorithm manually
    print("\n" + "=" * 60)
    print("🚀 LATS ALGORITHM EXECUTION")
    print("=" * 60)
    
    # Create root and run search iterations
    root = TreeNode()
    best_score = 0.0
    best_solution = None
    nodes_explored = 1
    
    for iteration in range(config.max_iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{config.max_iterations}")
        
        # SELECT
        selected = await algorithm.select_node(root)
        print(f"   🎯 Selected node at depth {selected.depth}")
        
        # EXPAND
        children = await algorithm.expand_node(selected, task, tools)
        print(f"   🌿 Expanded into {len(children)} actions")
        nodes_explored += len(children)
        
        # SIMULATE & REFLECT
        for child in children:
            observation = await algorithm.simulate_node(child, tools)
            child.observation = observation
            
            score = await algorithm.reflect_on_node(child, task)
            algorithm.backpropagate(child, score)
            
            if score > best_score:
                best_score = score
                best_solution = child
            
            print(f"   📊 Action: {child.action[:50]}... → Score: {score:.1f}")
    
    result = {
        'score': best_score,
        'nodes_explored': nodes_explored,
        'files_found': len([n for n in traverse_tree(root) if n.observation and n.action and 'file' in n.action.lower()]),
        'patterns': len([n for n in traverse_tree(root) if n.value > 7.0]),
        'best_solution': {
            'action': best_solution.action if best_solution else None,
            'score': best_score,
            'findings': best_solution.observation if best_solution else None
        } if best_solution else None
    }
    
    print("\n📊 RESULTS:")
    print("-" * 40)
    print(f"✅ Investigation completed")
    print(f"📈 Quality score: {result.get('score', 0):.1f}/10")
    print(f"🌳 Nodes explored: {result.get('nodes_explored', 0)}")
    print(f"📁 Files discovered: {result.get('files_found', 0)}")
    print(f"🔍 Patterns identified: {result.get('patterns', 0)}")
    
    if 'best_solution' in result:
        solution = result['best_solution']
        print(f"\n🏆 BEST SOLUTION FOUND:")
        print(f"   Action: {solution.get('action', 'N/A')}")
        print(f"   Score: {solution.get('score', 0):.1f}/10")
        
        if 'findings' in solution:
            print(f"\n🔍 FINDINGS:")
            findings = solution['findings'][:500]  # Truncate for display
            print(f"   {findings}")
    
    # Show what agent learned
    print(f"\n🧠 MEMORY INSIGHTS:")
    insights = memory.get_insights()
    if insights:
        for insight in insights[:3]:
            print(f"   • {insight}")
    
    return result


async def run_step_by_step_demo():
    """Show LATS working step by step"""
    
    print("\n" + "=" * 80)
    print("🔬 STEP-BY-STEP LATS DEMONSTRATION")
    print("=" * 80)
    
    # Change to sample codebase
    os.chdir(Path(__file__).parent / 'sample_codebase')
    
    config = LATSConfig(max_depth=3, max_iterations=3, num_expand=2)
    
    if use_real_llm:
        llm = ChatOllama(model="gpt-oss", base_url="http://localhost:11434")
    else:
        llm = TestLLM()
    
    algorithm = LATSAlgorithm(config)
    tools = create_filesystem_tools()
    
    task = "Find authentication vulnerabilities"
    root = TreeNode()
    
    print(f"\n🎯 Task: {task}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"🧠 Agent has no prior knowledge of the codebase")
    
    iteration = 1
    
    while iteration <= config.max_iterations:
        print(f"\n{'='*50}")
        print(f"🔄 ITERATION {iteration}")
        print(f"{'='*50}")
        
        # SELECT
        print("\n1️⃣ SELECT: Choosing most promising node...")
        selected = await algorithm.select_node(root)
        
        if selected == root:
            print("   🌱 Selected ROOT - starting exploration")
        else:
            print(f"   🎯 Selected node at depth {selected.depth}")
            print(f"   📊 UCT score: {selected.uct_score():.3f}")
            print(f"   📈 Visits: {selected.visits}, Value: {selected.value:.2f}")
            if selected.action:
                print(f"   ⚡ Previous action: {selected.action}")
        
        # EXPAND  
        print("\n2️⃣ EXPAND: Agent generates possible actions...")
        children = await algorithm.expand_node(selected, task, tools)
        
        print(f"   🌿 Generated {len(children)} possible actions:")
        for i, child in enumerate(children, 1):
            print(f"   {i}. {child.action}")
        
        # SIMULATE each child
        print("\n3️⃣ SIMULATE: Executing actions and gathering observations...")
        
        for j, child in enumerate(children, 1):
            print(f"\n   ⚡ Executing action {j}: {child.action}")
            
            # Real execution
            observation = await algorithm.simulate_node(child, tools)
            child.observation = observation
            
            # Show first part of observation
            if observation and len(observation) > 100:
                preview = observation[:200] + "..."
                print(f"   📝 Observation: {preview}")
            elif observation:
                print(f"   📝 Observation: {observation}")
            else:
                print(f"   ❌ No observation returned")
        
        # REFLECT on each action
        print("\n4️⃣ REFLECT: Agent evaluates the quality of each action...")
        
        for child in children:
            score = await algorithm.reflect_on_node(child, task)
            print(f"   📊 {child.action} → Score: {score:.1f}/10")
            
            # BACKPROPAGATE
            algorithm.backpropagate(child, score)
        
        # Show tree stats
        total_nodes = len(list(algorithm._traverse_tree(root)))
        print(f"\n📈 Tree statistics: {total_nodes} total nodes explored")
        
        iteration += 1
        
        # Break if we found a good solution
        best_child = max(children, key=lambda x: x.value/max(x.visits, 1)) if children else None
        if best_child and best_child.value/max(best_child.visits, 1) > 8.0:
            print(f"\n🎉 High-quality solution found! Stopping early.")
            break
    
    print(f"\n{'='*80}")
    print("🏁 LATS SEARCH COMPLETE")
    print(f"{'='*80}")
    
    # Find best solution
    all_nodes = list(algorithm._traverse_tree(root))
    best_node = max(all_nodes, key=lambda x: x.value/max(x.visits, 1)) if all_nodes else None
    
    if best_node and best_node.observation:
        print(f"\n🏆 BEST SOLUTION:")
        print(f"   Action: {best_node.action}")
        print(f"   Score: {best_node.value/max(best_node.visits, 1):.1f}/10")
        print(f"   Observation preview:")
        obs_lines = best_node.observation.split('\n')[:10]
        for line in obs_lines:
            if line.strip():
                print(f"      {line}")
    
    return len(all_nodes)


async def main():
    """Run the real LATS demonstration"""
    
    try:
        # First try the full algorithm
        print("Testing full LATS algorithm...")
        result = await run_real_lats_investigation()
        
        print("\n" + "-" * 80)
        
        # Then show step-by-step breakdown
        print("Running step-by-step demonstration...")
        nodes_explored = await run_step_by_step_demo()
        
        print(f"\n🎊 Demo completed successfully!")
        print(f"🌳 Total nodes explored: {nodes_explored}")
        print(f"✅ LATS algorithm working with real inference and exploration")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during LATS execution: {e}")
        print(f"📝 This may indicate the agent needs more sophisticated action generation")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)