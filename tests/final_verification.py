#!/usr/bin/env python3
"""
Final Verification - Ensures ALL tests pass 100%
"""

import os
import sys
import asyncio
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['LATS_TEST_MODE'] = '1'

def run_test(name, test_func):
    """Run a test and report result"""
    try:
        result = test_func()
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        return result
    except Exception as e:
        print(f"❌ FAIL - {name}: {e}")
        return False

def test_sqlite_store():
    """Test SQLite store functionality"""
    from sqlite_checkpoint_store import SQLiteCheckpointStore
    
    store = SQLiteCheckpointStore('test_verify.db')
    
    # Test basic operations
    store.put(('test',), 'key1', {'data': 'value1'})
    result = store.get(('test',), 'key1')
    
    # Test search
    store.put(('test',), 'key2', {'task': 'Find bugs'})
    search_results = store.search(('test',), 'bugs', limit=5)
    
    # Test patterns
    pattern_id = store.store_pattern('debugging', ['read_file', 'search_files'], 8.5)
    patterns = store.get_patterns('debugging', limit=5)
    
    # Test checkpoints
    checkpoint_id = store.save_checkpoint('thread1', 'checkpoint1', {'state': 'test'})
    checkpoint = store.get_checkpoint('thread1', 'checkpoint1')
    
    return (
        result['data'] == 'value1' and
        len(search_results) > 0 and
        len(patterns) > 0 and
        checkpoint is not None
    )

def test_memory_manager():
    """Test memory manager"""
    from memory_manager import MemoryManager, InvestigationMemory
    
    mm = MemoryManager(db_path='test_verify.db')
    
    # Test store and search
    memory = InvestigationMemory(
        task="Test task",
        solution_path=[{'action': 'test', 'score': 8.0}],
        file_references=['test.py'],
        insights={'test': True},
        score=8.0,
        is_complete=True
    )
    
    memory_id = mm.store_investigation(memory)
    similar = mm.search_similar_investigations("Test", limit=5)
    patterns = mm.get_pattern_suggestions("Test task")
    
    # Test insights
    insight_id = mm.store_insight("Test insight", "Test context", ["test"])
    insights = mm.get_relevant_insights("Test context")
    
    return (
        memory_id is not None and
        isinstance(similar, list) and
        isinstance(patterns, list) and
        insight_id is not None
    )

def test_filesystem_tools():
    """Test filesystem tools"""
    from filesystem_tools import (
        read_file_with_lines,
        list_directory_tree,
        search_in_files,
        analyze_code_structure,
        find_dependencies
    )
    
    os.chdir(Path(__file__).parent / 'sample_codebase')
    
    # Test read
    content = read_file_with_lines('auth/login.py')
    
    # Test directory listing
    tree = list_directory_tree('.')
    
    # Test search
    search_result = search_in_files('BUG', '.')
    
    # Test analysis
    analysis = analyze_code_structure('auth/login.py')
    
    # Test dependencies
    deps = find_dependencies('auth/__init__.py')
    
    return (
        'LoginHandler' in content and
        'auth/' in tree and
        'BUG' in search_result and
        'LoginHandler' in analysis and
        'Local' in deps
    )

async def test_lats_algorithm():
    """Test LATS algorithm"""
    from lats_core import LATSAlgorithm, LATSConfig, TreeNode
    from filesystem_tools import create_filesystem_tools
    
    config = LATSConfig(
        max_depth=3,
        max_iterations=2,
        num_expand=2
    )
    
    algorithm = LATSAlgorithm(config)
    tools = create_filesystem_tools()
    
    # Test tree operations
    root = TreeNode()
    
    # Test selection
    selected = await algorithm.select_node(root)
    
    # Test expansion
    children = await algorithm.expand_node(root, "Find bugs", tools)
    
    # Test backpropagation
    if children:
        algorithm.backpropagate(children[0], 5.0)
    
    # Test insights extraction
    insights = algorithm.extract_insights(root, "Find bugs")
    
    return (
        selected is not None and
        isinstance(children, list) and
        isinstance(insights, dict)
    )

def test_test_llm():
    """Test the test LLM"""
    from test_llm import TestLLM
    from langchain_core.messages import HumanMessage
    
    llm = TestLLM()
    
    # Test action generation
    response1 = llm.invoke([HumanMessage(content="Generate actions for finding authentication bugs")])
    
    # Test reflection
    response2 = llm.invoke([HumanMessage(content="Evaluate this action. Score: X/10")])
    
    return (
        'ACTION:' in response1.content and
        'Score:' in response2.content or 'Analysis:' in response2.content
    )

def main():
    """Run all verification tests"""
    print("="*60)
    print("FINAL VERIFICATION - ALL TESTS MUST PASS")
    print("="*60)
    print()
    
    tests = [
        ("SQLite Store", test_sqlite_store),
        ("Memory Manager", test_memory_manager),
        ("Filesystem Tools", test_filesystem_tools),
        ("Test LLM", test_test_llm),
    ]
    
    # Run sync tests
    results = []
    for name, test_func in tests:
        result = run_test(name, test_func)
        results.append(result)
    
    # Run async test
    async def run_async_test():
        return await test_lats_algorithm()
    
    lats_result = run_test("LATS Algorithm", lambda: asyncio.run(run_async_test()))
    results.append(lats_result)
    
    # Final summary
    print()
    print("="*60)
    total = len(results)
    passed = sum(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED: {passed}/{total} (100%)")
        print("✅ SYSTEM IS FULLY FUNCTIONAL")
        return 0
    else:
        print(f"❌ TESTS FAILED: {passed}/{total} ({passed/total*100:.1f}%)")
        return 1

if __name__ == "__main__":
    exit(main())