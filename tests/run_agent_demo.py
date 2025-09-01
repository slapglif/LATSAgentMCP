#!/usr/bin/env python3
"""
Demo script to run LATS agent against sample codebase
Shows actual agent outputs and tree exploration
"""

import asyncio
import sys
import os
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['LATS_TEST_MODE'] = '1'

from lats_core import LATSAlgorithm, LATSConfig, TreeNode
from filesystem_tools import create_filesystem_tools
from memory_manager import MemoryManager
from test_llm import TestLLM


async def run_investigation_demo():
    """Run a detailed investigation showing agent outputs"""
    
    # Change to sample codebase
    os.chdir(Path(__file__).parent / 'sample_codebase')
    
    # Initialize components
    config = LATSConfig(
        max_depth=5,
        max_iterations=3,
        num_expand=3,
        exploration_constant=1.414
    )
    
    algorithm = LATSAlgorithm(config)
    tools = create_filesystem_tools()
    llm = TestLLM()
    memory = MemoryManager(db_path='demo_memory.db')
    
    # Investigation task
    task = "Find all authentication bugs and security vulnerabilities in the login module"
    
    print("=" * 80)
    print("LATS AGENT INVESTIGATION DEMO")
    print("=" * 80)
    print(f"\nTASK: {task}")
    print("\nStarting investigation of sample codebase...")
    print("-" * 80)
    
    # Create root node
    root = TreeNode()
    print("\n[TREE] Created root node")
    
    # Track exploration
    iteration = 0
    best_solution = None
    best_score = 0.0
    
    while iteration < config.max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{config.max_iterations}")
        print(f"{'='*60}")
        
        # SELECT: Choose most promising node
        print("\n[SELECT] Choosing node to explore...")
        selected = await algorithm.select_node(root)
        
        if selected == root:
            print("  → Selected: ROOT (initial exploration)")
        else:
            print(f"  → Selected: Node at depth {selected.depth}")
            print(f"    UCT Score: {selected.uct_score():.3f}")
            print(f"    Visits: {selected.visits}, Value: {selected.value:.2f}")
        
        # EXPAND: Generate child actions
        print("\n[EXPAND] Generating possible actions...")
        children = await algorithm.expand_node(selected, task, tools)
        
        for i, child in enumerate(children, 1):
            print(f"\n  Action {i}: {child.action}")
            if child.observation:
                reasoning = child.observation[:200] + "..." if len(child.observation) > 200 else child.observation
                print(f"  Reasoning: {reasoning}")
            else:
                print(f"  Reasoning: [Action not yet executed]")
        
        # SIMULATE & EVALUATE each child
        print("\n[SIMULATE] Executing and evaluating actions...")
        
        for child in children:
            print(f"\n  Simulating: {child.action}")
            
            # Execute the action
            result = await algorithm.simulate_node(child, tools)
            child.observation = result
            
            # Show what the agent found
            if "Error:" not in result:
                lines = result.split('\n')[:5]  # First 5 lines
                for line in lines:
                    if line.strip():
                        print(f"    → {line[:100]}...")
            
            # REFLECT: Evaluate the result
            score = await algorithm.reflect_on_node(child, task)
            print(f"    Score: {score:.1f}/10")
            
            # Extract insights from high-scoring actions
            if score > 7.0:
                print(f"    ✓ High-value finding!")
                
                # Store in memory - create proper InvestigationMemory object
                from memory_manager import InvestigationMemory
                investigation_memory = InvestigationMemory(
                    task=task,
                    solution_path=[{'action': child.action, 'score': score}],
                    file_references=[],
                    insights={'finding': result[:200]},
                    score=score,
                    is_complete=True
                )
                memory.store_investigation(investigation_memory)
            
            # BACKPROPAGATE: Update tree statistics
            algorithm.backpropagate(child, score)
            
            # Track best solution
            if score > best_score:
                best_score = score
                best_solution = child
        
        # Show tree statistics
        print(f"\n[TREE STATS]")
        print(f"  Total nodes: {algorithm._count_nodes(root)}")
        print(f"  Max depth: {algorithm._get_max_depth(root)}")
        print(f"  Best score so far: {best_score:.1f}/10")
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    
    if best_solution:
        print(f"\n[BEST SOLUTION]")
        print(f"  Action: {best_solution.action}")
        print(f"  Score: {best_score:.1f}/10")
        print(f"\n  Findings:")
        
        # Parse findings from observation
        obs_lines = best_solution.observation.split('\n')
        for line in obs_lines[:15]:  # Show first 15 lines
            if line.strip():
                print(f"    {line}")
    
    # Extract and show patterns learned
    print("\n[PATTERNS LEARNED]")
    insights = algorithm.extract_insights(root, task)
    
    if insights.get('successful_patterns'):
        print("  Successful approaches:")
        for pattern in insights['successful_patterns'][:3]:
            print(f"    • {pattern}")
    
    if insights.get('failure_patterns'):
        print("  Approaches to avoid:")
        for pattern in insights['failure_patterns'][:3]:
            print(f"    • {pattern}")
    
    # Show memory persistence
    print("\n[MEMORY STORED]")
    similar = memory.search_similar_investigations(task, limit=3)
    print(f"  Stored {len(similar)} investigation memories")
    
    patterns = memory.get_pattern_suggestions(task)
    if patterns:
        print(f"  Discovered {len(patterns)} reusable patterns")
    
    return best_score


def algorithm_count_nodes(algorithm, node):
    """Helper to count nodes in tree"""
    count = 1
    for child in node.children:
        count += algorithm_count_nodes(algorithm, child)
    return count


def algorithm_get_max_depth(algorithm, node, depth=0):
    """Helper to get max depth of tree"""
    if not node.children:
        return depth
    return max(algorithm_get_max_depth(algorithm, child, depth + 1) for child in node.children)


# Monkey-patch the methods
LATSAlgorithm._count_nodes = lambda self, node: algorithm_count_nodes(self, node)
LATSAlgorithm._get_max_depth = lambda self, node: algorithm_get_max_depth(self, node)


async def main():
    """Run the demo"""
    score = await run_investigation_demo()
    
    print("\n" + "=" * 80)
    print(f"Final Investigation Score: {score:.1f}/10")
    
    if score >= 8.0:
        print("✅ Successfully identified critical security issues!")
    elif score >= 6.0:
        print("✓ Found some security issues, partial success")
    else:
        print("⚠ Investigation needs improvement")
    
    return 0 if score >= 6.0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)