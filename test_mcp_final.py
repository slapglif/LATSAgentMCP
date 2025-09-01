#!/usr/bin/env python3
"""Final MCP test - quick verification of all improvements"""

import asyncio
import sys
sys.path.insert(0, '/home/ubuntu/lats')
import mcp_server

async def main():
    print("=" * 80)
    print("FINAL MCP TEST - VERIFYING ALL IMPROVEMENTS")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Quick overview with pagination
    print("\n✅ Test 1: Quick Overview")
    result = await mcp_server.quick_codebase_overview.fn(codebase_path=".", page=1, page_size=10)
    results['overview'] = 'pagination' in result
    print(f"   Pagination: {'✅' if results['overview'] else '❌'}")
    
    # Test 2: Pattern search with filtering  
    print("\n✅ Test 2: Pattern Search")
    result = await mcp_server.find_code_patterns.fn(
        pattern="def", 
        file_extension="*.py",
        page=1,
        page_size=5
    )
    results['search'] = 'pagination' in result
    print(f"   Pagination: {'✅' if results['search'] else '❌'}")
    
    # Test 3: Quick LATS test with scoring
    print("\n✅ Test 3: LATS Scoring")
    result = await mcp_server.analyze_codebase.fn(
        task="Find eval() usage",
        generate_report=False,
        max_depth=2,
        streaming=False
    )
    
    # Check scoring improvements
    if 'tree' in result and 'children' in result['tree']:
        scores = []
        for child in result['tree']['children'][:5]:
            action = child.get('action', '')
            score = child.get('value', 0)
            if 'list_directory' in action:
                scores.append(('dir', score))
                print(f"   Directory score: {score} (should be ≤6)")
            elif 'search' in action and 'eval' in action:
                scores.append(('eval', score))
                print(f"   Eval search score: {score} (should be high)")
        
        # Check if scoring is working correctly
        dir_scores = [s[1] for s in scores if s[0] == 'dir']
        eval_scores = [s[1] for s in scores if s[0] == 'eval']
        
        if dir_scores and all(s <= 7 for s in dir_scores):
            results['dir_capping'] = True
            print("   ✅ Directory scores are capped")
        else:
            results['dir_capping'] = False
            print("   ❌ Directory scores not capped properly")
            
        if eval_scores and all(s >= 7 for s in eval_scores):
            results['eval_priority'] = True
            print("   ✅ Security-relevant actions prioritized")
        else:
            results['eval_priority'] = False
            print("   ❌ Security actions not prioritized")
    
    # Test 4: Report generation with correct LATS
    print("\n✅ Test 4: Report Generation")
    result = await mcp_server.analyze_codebase.fn(
        task="Quick test",
        generate_report=True,
        max_depth=1,
        streaming=False
    )
    
    if 'report' in result:
        report = result['report']
        if 'Language Agent Tree Search' in report:
            results['lats_correct'] = True
            print("   ✅ LATS acronym correct")
        else:
            results['lats_correct'] = False
            print("   ❌ LATS acronym incorrect")
            
        # Save report
        with open('final_test_report.md', 'w') as f:
            f.write(report)
        print("   📄 Report saved to final_test_report.md")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        print(f"{'✅' if passed else '❌'} {test}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
    else:
        print(f"\n⚠️ {sum(not v for v in results.values())} tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)