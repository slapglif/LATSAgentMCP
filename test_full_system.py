#!/usr/bin/env python3
"""Test full LATS system with reporting"""

import asyncio
import json
from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools

async def test_with_report():
    """Test LATS with report generation"""
    print("=" * 80)
    print("Testing Full LATS System with Report Generation")
    print("=" * 80)
    
    agent = LATSAgent(
        max_depth=5,
        log_file="full_system_test.log"
    )
    
    tools = create_filesystem_tools()
    
    # Test with a security-focused task
    task = "Find potential security vulnerabilities in the Python files"
    
    print(f"\n📋 Task: {task}")
    print(f"🔧 Tools available: {len(tools)}")
    print(f"🌳 Max depth: {agent.max_depth}")
    print("\nStarting investigation...")
    
    try:
        result = await agent.investigate(
            task=task,
            tools=tools,
            generate_report=True  # Generate full report
        )
        
        print("\n" + "=" * 40)
        print("INVESTIGATION COMPLETE")
        print("=" * 40)
        print(f"✅ Status: {'Completed' if result['completed'] else 'Ongoing'}")
        print(f"📊 Nodes explored: {result['nodes_explored']}")
        print(f"🌳 Max depth reached: {result['max_depth']}")
        print(f"⏱️ Duration: {result['duration_seconds']:.2f}s")
        print(f"📝 Report generated: {'Yes' if 'report' in result else 'No'}")
        
        # Check scoring patterns
        if 'tree' in result:
            print("\n📈 Scoring Analysis:")
            
            def analyze_scores(node, depth=0):
                if 'action' in node and node['action']:
                    action_type = node['action'].split('(')[0]
                    score = node.get('value', 0)
                    
                    if 'list_directory' in action_type:
                        print(f"  {'  '*depth}📁 {action_type}: {score:.1f}")
                    elif 'search' in action_type:
                        print(f"  {'  '*depth}🔍 {action_type}: {score:.1f}")
                    else:
                        print(f"  {'  '*depth}⚡ {action_type}: {score:.1f}")
                
                for child in node.get('children', []):
                    analyze_scores(child, depth + 1)
            
            analyze_scores(result['tree'])
        
        # Save report if generated
        if 'report' in result:
            report_file = f"test_report_{result['session_id']}.md"
            with open(report_file, 'w') as f:
                f.write(result['report'])
            print(f"\n📄 Report saved to: {report_file}")
            
            # Check report content
            print("\n📝 Report Preview:")
            lines = result['report'].split('\n')[:20]
            for line in lines:
                print(f"  {line}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run full system test"""
    result = await test_with_report()
    
    if result:
        print("\n" + "=" * 80)
        print("✅ Full system test completed successfully!")
        print("=" * 80)
        
        # Summary
        print("\n📊 Summary:")
        print(f"  - Nodes explored: {result['nodes_explored']}")
        print(f"  - Max depth: {result['max_depth']}")
        print(f"  - Duration: {result['duration_seconds']:.2f}s")
        
        if result['max_depth'] > 0:
            print("  ✅ Tree expansion working correctly")
        else:
            print("  ⚠️ Tree didn't expand - may need investigation")
    else:
        print("\n❌ Test failed - check logs for details")

if __name__ == "__main__":
    asyncio.run(main())