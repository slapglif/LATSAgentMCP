#!/usr/bin/env python3
"""Test the enhanced MCP server with streaming and adaptive configuration"""

import asyncio
import sys
import os

# Import the raw functions directly
sys.path.append('.')
from mcp_server import _analyze_codebase_characteristics, _determine_optimal_config, _generate_analysis_recommendations

async def test_recommendations():
    """Test the intelligent recommendation system"""
    print("🧪 Testing Analysis Recommendations")
    print("=" * 50)
    
    # Test different types of analysis tasks
    test_tasks = [
        "Understand the LATS implementation and architecture",
        "Find performance bottlenecks in the code",
        "Identify security vulnerabilities", 
        "Debug memory leaks and errors",
        "General code review"
    ]
    
    for task in test_tasks:
        print(f"\n🔍 Task: {task}")
        print("-" * 40)
        
        try:
            # Analyze codebase characteristics
            codebase_info = await _analyze_codebase_characteristics()
            
            # Get optimal configuration  
            config = _determine_optimal_config(task, codebase_info, None, None)
            
            # Generate recommendations
            recommendations = _generate_analysis_recommendations(task, codebase_info, config)
            
            strategy = recommendations["strategy_explanation"]
            duration = recommendations["estimated_duration"]
            considerations = recommendations["key_considerations"]
            
            print(f"📊 Recommended Configuration:")
            print(f"   • Max Depth: {config['max_depth']}")
            print(f"   • Strategy: {config['strategy']}")
            print(f"   • Estimated Duration: {duration}")
            
            print(f"\n💡 Strategy: {strategy}")
            
            if considerations:
                print(f"\n🎯 Key Considerations:")
                for consideration in considerations[:3]:  # Show first 3
                    print(f"   • {consideration}")
                    
            alternatives = recommendations.get("alternative_approaches", [])
            if alternatives:
                print(f"\n🔄 Alternative Approaches:")
                for alt in alternatives:
                    print(f"   • {alt['name']}: {alt['description']}")
                    
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

async def test_adaptive_configuration():
    """Test the adaptive configuration logic"""
    print("\n\n🚀 Testing Adaptive Configuration")
    print("=" * 50)
    
    # Get codebase characteristics first
    codebase_info = await _analyze_codebase_characteristics()
    
    print(f"📁 Codebase Characteristics:")
    print(f"   • Total files: {codebase_info['total_files']}")
    print(f"   • Languages: {codebase_info['languages']}")
    print(f"   • Total size: {codebase_info['total_size_mb']:.2f} MB")
    print(f"   • Complexity indicators: {codebase_info['complexity_indicators']}")
    
    # Test different task configurations
    test_tasks = [
        ("Understand architecture", "architectural analysis"),
        ("Find performance bottlenecks", "performance analysis"), 
        ("Security audit", "security analysis"),
        ("Debug errors", "debugging analysis")
    ]
    
    for task, expected_type in test_tasks:
        print(f"\n🔧 Task: {task}")
        print("-" * 40)
        
        config = _determine_optimal_config(task, codebase_info, None, None)
        recommendations = _generate_analysis_recommendations(task, codebase_info, config)
        
        print(f"   • Recommended max_depth: {config['max_depth']}")
        print(f"   • Strategy: {config['strategy']}")
        print(f"   • Estimated duration: {recommendations['estimated_duration']}")
        print(f"   • Key considerations: {len(recommendations['key_considerations'])}")
        
        # Test urgency adaptations
        high_urgency_config = config.copy()
        high_urgency_config['max_depth'] = min(config['max_depth'], 6)
        
        low_urgency_config = config.copy()
        low_urgency_config['max_depth'] = max(config['max_depth'], 10)
        
        print(f"   • High urgency depth: {high_urgency_config['max_depth']}")
        print(f"   • Low urgency depth: {low_urgency_config['max_depth']}")

async def main():
    """Run all tests"""
    await test_recommendations()
    await test_adaptive_configuration()
    
    print("\n\n🎉 Testing Complete!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())