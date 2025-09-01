#!/usr/bin/env python3
"""Test scoring improvements to prevent local maxima"""

import asyncio
from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools

async def test_repetition_penalty():
    """Test that repeated actions get penalized"""
    print("=" * 80)
    print("Testing Repetition Penalty System")
    print("=" * 80)
    
    agent = LATSAgent(max_depth=5)
    tools = create_filesystem_tools()
    
    # Test 1: Directory listing task (should see reduced scores for repetition)
    task1 = "List all Python files in the project"
    print(f"\n📋 Test 1: {task1}")
    
    result1 = await agent.investigate(
        task=task1,
        tools=tools,
        generate_report=False
    )
    
    print(f"✅ Nodes explored: {result1['nodes_explored']}")
    print(f"🌳 Max depth: {result1['max_depth']}")
    
    # Check if directory listing scores decreased with repetition
    if 'tree' in result1 and 'children' in result1['tree']:
        for child in result1['tree']['children'][:3]:
            action = child.get('action', '')
            score = child.get('value', 0)
            print(f"  Action: {action[:50]}... Score: {score}")

async def test_context_aware_scoring():
    """Test context-aware scoring adjustments"""
    print("\n" + "=" * 80)
    print("Testing Context-Aware Scoring")
    print("=" * 80)
    
    agent = LATSAgent(max_depth=3)
    tools = create_filesystem_tools()
    
    # Test 2: Security task (directory listing should be capped)
    task2 = "Find security vulnerabilities in the authentication system"
    print(f"\n📋 Test 2: {task2}")
    
    result2 = await agent.investigate(
        task=task2,
        tools=tools,
        generate_report=False
    )
    
    print(f"✅ Nodes explored: {result2['nodes_explored']}")
    
    # Check scoring patterns
    if 'tree' in result2 and 'children' in result2['tree']:
        dir_scores = []
        search_scores = []
        
        for child in result2['tree']['children']:
            action = child.get('action', '')
            score = child.get('value', 0)
            
            if 'list_directory' in action:
                dir_scores.append(score)
                print(f"  📁 Directory listing score: {score} (should be ≤6)")
            elif 'search' in action:
                search_scores.append(score)
                print(f"  🔍 Search action score: {score}")
        
        if dir_scores and search_scores:
            avg_dir = sum(dir_scores) / len(dir_scores)
            avg_search = sum(search_scores) / len(search_scores)
            print(f"\n  Average directory score: {avg_dir:.1f}")
            print(f"  Average search score: {avg_search:.1f}")
            
            if avg_search > avg_dir:
                print("  ✅ Search actions scored higher than directory listing (good!)")
            else:
                print("  ⚠️ Directory listing still scoring too high")

async def test_diversity_bonus():
    """Test that diverse actions get rewarded"""
    print("\n" + "=" * 80)
    print("Testing Diversity Bonus System")
    print("=" * 80)
    
    agent = LATSAgent(max_depth=4)
    tools = create_filesystem_tools()
    
    # Test 3: General exploration task
    task3 = "Analyze the codebase structure and find interesting patterns"
    print(f"\n📋 Test 3: {task3}")
    
    result3 = await agent.investigate(
        task=task3,
        tools=tools,
        generate_report=False
    )
    
    print(f"✅ Nodes explored: {result3['nodes_explored']}")
    
    # Count unique action types
    if 'tree' in result3:
        action_types = set()
        
        def count_actions(node):
            if 'action' in node and node['action']:
                action_type = node['action'].split('(')[0]
                action_types.add(action_type)
            for child in node.get('children', []):
                count_actions(child)
        
        count_actions(result3['tree'])
        
        print(f"🎨 Unique action types used: {len(action_types)}")
        print(f"   Actions: {', '.join(sorted(action_types))}")
        
        if len(action_types) >= 3:
            print("  ✅ Good diversity in action selection!")
        else:
            print("  ⚠️ Limited diversity - may need tuning")

async def main():
    """Run all scoring tests"""
    print("\n" + "🧪 " * 20)
    print("TESTING SCORING IMPROVEMENTS")
    print("🧪 " * 20)
    
    try:
        # Run tests sequentially to see clear output
        await test_repetition_penalty()
        await test_context_aware_scoring()
        await test_diversity_bonus()
        
        print("\n" + "=" * 80)
        print("✅ All scoring tests completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())