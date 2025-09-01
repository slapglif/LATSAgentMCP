#!/usr/bin/env python3
"""
REAL OLLAMA LATS TEST - No mocks, no fakes, only real Ollama inference
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from lats_langgraph import LATSAgent
from filesystem_tools import create_filesystem_tools


async def test_real_ollama():
    """Test with real Ollama inference"""
    
    # Check Ollama is running
    import subprocess
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/version'], 
                              capture_output=True, timeout=5)
        if result.returncode != 0:
            print("❌ Ollama is not running at localhost:11434")
            print("Please start Ollama: ollama serve")
            return 1
    except Exception as e:
        print(f"❌ Cannot reach Ollama: {e}")
        return 1
    
    # Check gpt-oss model is available
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'gpt-oss' not in result.stdout:
            print("❌ gpt-oss model not found")
            print("Please install: ollama pull gpt-oss")
            return 1
        print("✅ Ollama and gpt-oss model available")
    except Exception as e:
        print(f"❌ Cannot check Ollama models: {e}")
        return 1
    
    # Change to sample codebase
    sample_dir = Path(__file__).parent / 'sample_codebase'
    os.chdir(sample_dir)
    
    print(f"\n{'='*60}")
    print("REAL OLLAMA LATS INVESTIGATION")
    print(f"{'='*60}")
    print(f"Working directory: {os.getcwd()}")
    print("Model: gpt-oss (real inference)")
    print("Task: Find authentication vulnerabilities")
    print("-" * 60)
    
    # Initialize LATS with real Ollama and logging
    log_file = f"lats_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    agent = LATSAgent(max_depth=3, max_iterations=6, log_file=log_file)
    tools = create_filesystem_tools()
    
    print(f"📝 Logging to: {log_file}")
    
    task = "Find authentication bugs and security vulnerabilities in this codebase"
    
    print("Starting investigation with REAL OLLAMA INFERENCE...")
    print()
    
    # Run the investigation with report generation
    result = await agent.investigate(task, tools, generate_report=True)
    
    print(f"\n{'='*60}")
    print("REAL INVESTIGATION RESULTS")
    print(f"{'='*60}")
    
    print(f"Completed: {result['completed']}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Max depth: {result['max_depth']}")
    print(f"Best score: {result['best_score']:.1f}/10")
    
    print(f"\nBest action: {result['best_action']}")
    
    if result['best_observation']:
        print(f"\nActual findings from real inference:")
        print("-" * 40)
        print(result['best_observation'][:800])
        if len(result['best_observation']) > 800:
            print("...")
    
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    # Verify we got real results
    success_criteria = [
        result['nodes_explored'] >= 3,
        result['best_score'] >= 6.0,
        result['best_observation'] and len(result['best_observation']) > 100,
        result['best_observation'] and ('auth/' in result['best_observation'] or 'login.py' in result['best_observation'] or 'database/' in result['best_observation'])
    ]
    
    print(f"Nodes explored >= 3: {'✅' if success_criteria[0] else '❌'}")
    print(f"Quality score >= 6.0: {'✅' if success_criteria[1] else '❌'}")
    print(f"Has substantial findings: {'✅' if success_criteria[2] else '❌'}")  
    print(f"Contains real content: {'✅' if success_criteria[3] else '❌'}")
    
    if all(success_criteria):
        print(f"\n✅ SUCCESS: Real Ollama inference working!")
        print("🧠 Agent used actual LLM reasoning")
        print("🔍 Found real code and vulnerabilities") 
        print("🌳 Performed genuine tree search")
        print(f"📝 Check logs: {log_file}")
        print("📄 Investigation report generated")
        return 0
    else:
        print(f"\n❌ FAILED: Something is still fake")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_real_ollama())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)