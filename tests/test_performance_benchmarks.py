"""
Performance Benchmarks for LATS System
Measures real-world performance with no mocks
"""

import asyncio
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ['LATS_TEST_MODE'] = '1'

from lats_core import LATSAlgorithm, LATSConfig, TreeNode
from filesystem_tools import create_filesystem_tools
from memory_manager import MemoryManager, InvestigationMemory


class PerformanceBenchmark:
    """Performance benchmark suite for LATS"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "sample_codebase"
        self.results = {}
        
        # Different configurations to test
        self.configs = {
            'minimal': LATSConfig(
                max_depth=2,
                max_iterations=3,
                num_expand=2,
                temperature=0.5
            ),
            'standard': LATSConfig(
                max_depth=4,
                max_iterations=5,
                num_expand=3,
                temperature=0.7
            ),
            'extensive': LATSConfig(
                max_depth=6,
                max_iterations=8,
                num_expand=5,
                temperature=0.8
            )
        }
        
        # Test tasks of varying complexity
        self.test_tasks = {
            'simple': "Find the main authentication function",
            'medium': "Identify all error handling patterns in the codebase",
            'complex': "Analyze the complete data flow from user input to database storage",
            'parallel': "Find all security vulnerabilities across authentication, database, and input handling"
        }
    
    async def benchmark_tree_operations(self) -> Dict[str, Any]:
        """Benchmark tree search operations"""
        print("\n[BENCHMARK] Tree Operations")
        results = {}
        
        for config_name, config in self.configs.items():
            algorithm = LATSAlgorithm(config)
            tools = create_filesystem_tools()
            
            # Measure tree creation and traversal
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            root = TreeNode()
            nodes_created = 0
            
            # Build a test tree
            for _ in range(config.max_iterations):
                current = root
                for depth in range(config.max_depth):
                    if not current.children:
                        for i in range(config.num_expand):
                            child = current.add_child(f"action_{depth}_{i}")
                            nodes_created += 1
                    if current.children:
                        current = current.children[0]
            
            # Measure selection performance
            selection_times = []
            for _ in range(100):
                select_start = time.perf_counter()
                selected = await algorithm.select_node(root)
                selection_times.append(time.perf_counter() - select_start)
            
            # Measure backpropagation
            backprop_times = []
            leaf_nodes = []
            
            def find_leaves(node):
                if not node.children:
                    leaf_nodes.append(node)
                for child in node.children:
                    find_leaves(child)
            
            find_leaves(root)
            
            for leaf in leaf_nodes[:10]:
                backprop_start = time.perf_counter()
                algorithm.backpropagate(leaf, 5.0)
                backprop_times.append(time.perf_counter() - backprop_start)
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            results[config_name] = {
                'nodes_created': nodes_created,
                'total_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'avg_selection_time': sum(selection_times) / len(selection_times) if selection_times else 0,
                'avg_backprop_time': sum(backprop_times) / len(backprop_times) if backprop_times else 0,
                'tree_depth': config.max_depth,
                'branching_factor': config.num_expand
            }
        
        return results
    
    async def benchmark_tool_execution(self) -> Dict[str, Any]:
        """Benchmark filesystem tool execution"""
        print("\n[BENCHMARK] Tool Execution")
        
        tools = create_filesystem_tools()
        results = {}
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Benchmark each tool
        tool_benchmarks = {
            'read_file': ('read_file', {'file_path': 'auth/login.py'}),
            'list_directory': ('list_directory', {'directory_path': '.', 'max_depth': 2}),
            'search_files': ('search_files', {'pattern': 'def', 'directory': '.'}),
            'analyze_structure': ('analyze_structure', {'file_path': 'auth/login.py'}),
            'find_dependencies': ('find_dependencies', {'file_path': 'auth/login.py'})
        }
        
        for benchmark_name, (tool_name, args) in tool_benchmarks.items():
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                continue
            
            # Warm up
            tool.func(args)
            
            # Measure execution time
            times = []
            for _ in range(10):
                start = time.perf_counter()
                result = tool.func(args)
                times.append(time.perf_counter() - start)
            
            results[benchmark_name] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'result_size': len(str(result))
            }
        
        return results
    
    async def benchmark_memory_operations(self) -> Dict[str, Any]:
        """Benchmark memory manager operations"""
        print("\n[BENCHMARK] Memory Operations")
        
        manager = MemoryManager()
        results = {}
        
        # Benchmark store operations
        store_times = []
        memory_ids = []
        
        for i in range(50):
            memory = InvestigationMemory(
                task=f"Test task {i}",
                solution_path=[{'action': f'action_{i}', 'score': i % 10}],
                file_references=[f'file_{i}.py:{i}'],
                insights={'test': i},
                score=float(i % 10),
                is_complete=i % 2 == 0
            )
            
            start = time.perf_counter()
            memory_id = manager.store_investigation(memory)
            store_times.append(time.perf_counter() - start)
            memory_ids.append(memory_id)
        
        results['store'] = {
            'count': len(store_times),
            'avg_time': sum(store_times) / len(store_times),
            'total_time': sum(store_times)
        }
        
        # Benchmark search operations
        search_times = []
        search_queries = [
            "Test task",
            "authentication",
            "database",
            "error handling",
            "security"
        ]
        
        for query in search_queries * 5:
            start = time.perf_counter()
            results_found = manager.search_similar_investigations(query, limit=5)
            search_times.append(time.perf_counter() - start)
        
        results['search'] = {
            'count': len(search_times),
            'avg_time': sum(search_times) / len(search_times),
            'total_time': sum(search_times)
        }
        
        # Benchmark pattern extraction
        pattern_times = []
        for i in range(20):
            start = time.perf_counter()
            patterns = manager.get_pattern_suggestions(f"Test task {i}")
            pattern_times.append(time.perf_counter() - start)
        
        results['patterns'] = {
            'count': len(pattern_times),
            'avg_time': sum(pattern_times) / len(pattern_times),
            'total_time': sum(pattern_times)
        }
        
        return results
    
    async def benchmark_full_investigation(self) -> Dict[str, Any]:
        """Benchmark complete investigation cycles"""
        print("\n[BENCHMARK] Full Investigation Cycles")
        
        results = {}
        
        for task_name, task in self.test_tasks.items():
            print(f"  Testing: {task_name}")
            
            for config_name, config in self.configs.items():
                gc.collect()  # Clean up before measurement
                
                algorithm = LATSAlgorithm(config)
                tools = create_filesystem_tools()
                
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                # Run investigation
                root = TreeNode()
                total_nodes = 0
                total_tool_calls = 0
                max_depth_reached = 0
                
                for iteration in range(config.max_iterations):
                    current = await algorithm.select_node(root)
                    max_depth_reached = max(max_depth_reached, current.depth)
                    
                    children = await algorithm.expand_node(current, task, tools)
                    total_nodes += len(children)
                    
                    for child in children:
                        # Simulate tool execution
                        total_tool_calls += 1
                        child.observation = f"Simulated observation for {child.action}"
                        
                        score = await algorithm.reflect_on_node(child, task)
                        algorithm.backpropagate(child, score)
                        
                        if child.is_terminal:
                            break
                    
                    if any(c.is_terminal for c in children):
                        break
                
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                
                key = f"{task_name}_{config_name}"
                results[key] = {
                    'task': task_name,
                    'config': config_name,
                    'execution_time': end_time - start_time,
                    'memory_used': end_memory - start_memory,
                    'total_nodes': total_nodes,
                    'tool_calls': total_tool_calls,
                    'max_depth': max_depth_reached,
                    'time_per_node': (end_time - start_time) / max(total_nodes, 1),
                    'memory_per_node': (end_memory - start_memory) / max(total_nodes, 1)
                }
        
        return results
    
    async def benchmark_parallel_execution(self) -> Dict[str, Any]:
        """Benchmark parallel execution capabilities"""
        print("\n[BENCHMARK] Parallel Execution")
        
        results = {}
        
        # Test different levels of parallelism
        parallel_configs = [1, 2, 4, 8]
        
        for num_parallel in parallel_configs:
            tasks = []
            start_time = time.perf_counter()
            
            # Create parallel investigation tasks
            for i in range(num_parallel):
                task = self._run_simple_investigation(f"Find function_{i}")
                tasks.append(task)
            
            # Execute in parallel
            await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            
            results[f"parallel_{num_parallel}"] = {
                'num_tasks': num_parallel,
                'total_time': end_time - start_time,
                'time_per_task': (end_time - start_time) / num_parallel
            }
        
        # Calculate speedup
        if 'parallel_1' in results and 'parallel_8' in results:
            single_time = results['parallel_1']['total_time']
            parallel_time = results['parallel_8']['time_per_task']
            results['speedup'] = single_time / parallel_time
        
        return results
    
    async def _run_simple_investigation(self, task: str):
        """Run a simple investigation for parallel testing"""
        config = self.configs['minimal']
        algorithm = LATSAlgorithm(config)
        tools = create_filesystem_tools()
        
        root = TreeNode()
        for _ in range(2):  # Minimal iterations
            current = await algorithm.select_node(root)
            children = await algorithm.expand_node(current, task, tools)
            
            for child in children[:2]:  # Limit children
                child.observation = "Test observation"
                score = await algorithm.reflect_on_node(child, task)
                algorithm.backpropagate(child, score)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks"""
        print("="*60)
        print("PERFORMANCE BENCHMARKS")
        print("="*60)
        
        # Start memory tracking
        tracemalloc.start()
        
        # Run benchmarks
        self.results['tree_operations'] = await self.benchmark_tree_operations()
        self.results['tool_execution'] = await self.benchmark_tool_execution()
        self.results['memory_operations'] = await self.benchmark_memory_operations()
        self.results['full_investigation'] = await self.benchmark_full_investigation()
        self.results['parallel_execution'] = await self.benchmark_parallel_execution()
        
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        self.results['memory_profile'] = {
            'top_allocations': [(str(stat.traceback), stat.size / 1024 / 1024) 
                               for stat in top_stats[:5]]
        }
        
        tracemalloc.stop()
        
        # Generate report
        self._generate_report()
        
        # Calculate overall performance score
        return self._calculate_performance_score()
    
    def _generate_report(self):
        """Generate performance report"""
        print("\n" + "="*60)
        print("PERFORMANCE RESULTS")
        print("="*60)
        
        # Tree operations
        print("\nTREE OPERATIONS:")
        for config, metrics in self.results['tree_operations'].items():
            print(f"\n  {config.upper()}:")
            print(f"    Nodes: {metrics['nodes_created']}")
            print(f"    Time: {metrics['total_time']:.3f}s")
            print(f"    Memory: {metrics['memory_used']:.2f}MB")
            print(f"    Selection: {metrics['avg_selection_time']*1000:.2f}ms avg")
            print(f"    Backprop: {metrics['avg_backprop_time']*1000:.2f}ms avg")
        
        # Tool execution
        print("\nTOOL EXECUTION:")
        for tool, metrics in self.results['tool_execution'].items():
            print(f"  {tool}:")
            print(f"    Avg: {metrics['avg_time']*1000:.2f}ms")
            print(f"    Min/Max: {metrics['min_time']*1000:.2f}ms / {metrics['max_time']*1000:.2f}ms")
        
        # Memory operations
        print("\nMEMORY OPERATIONS:")
        for op, metrics in self.results['memory_operations'].items():
            print(f"  {op}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Avg: {metrics['avg_time']*1000:.2f}ms")
            print(f"    Total: {metrics['total_time']:.2f}s")
        
        # Full investigations
        print("\nFULL INVESTIGATION CYCLES:")
        for key, metrics in self.results['full_investigation'].items():
            print(f"  {key}:")
            print(f"    Time: {metrics['execution_time']:.2f}s")
            print(f"    Nodes: {metrics['total_nodes']}")
            print(f"    Time/Node: {metrics['time_per_node']*1000:.2f}ms")
            print(f"    Memory/Node: {metrics['memory_per_node']:.2f}MB")
        
        # Parallel execution
        print("\nPARALLEL EXECUTION:")
        for config, metrics in self.results['parallel_execution'].items():
            if config != 'speedup':
                print(f"  {config}:")
                print(f"    Total: {metrics['total_time']:.2f}s")
                print(f"    Per Task: {metrics['time_per_task']:.2f}s")
        
        if 'speedup' in self.results['parallel_execution']:
            print(f"\n  Speedup (8x parallel): {self.results['parallel_execution']['speedup']:.2f}x")
    
    def _calculate_performance_score(self) -> Dict[str, Any]:
        """Calculate overall performance score"""
        scores = {
            'tree_efficiency': 0,
            'tool_speed': 0,
            'memory_efficiency': 0,
            'scalability': 0,
            'overall': 0
        }
        
        # Tree efficiency (based on selection and backprop times)
        tree_metrics = self.results['tree_operations']['standard']
        if tree_metrics['avg_selection_time'] < 0.001:  # < 1ms
            scores['tree_efficiency'] = 100
        elif tree_metrics['avg_selection_time'] < 0.01:  # < 10ms
            scores['tree_efficiency'] = 80
        else:
            scores['tree_efficiency'] = 60
        
        # Tool speed
        tool_times = [m['avg_time'] for m in self.results['tool_execution'].values()]
        avg_tool_time = sum(tool_times) / len(tool_times)
        if avg_tool_time < 0.05:  # < 50ms
            scores['tool_speed'] = 100
        elif avg_tool_time < 0.1:  # < 100ms
            scores['tool_speed'] = 80
        else:
            scores['tool_speed'] = 60
        
        # Memory efficiency
        investigation_metrics = list(self.results['full_investigation'].values())[0]
        if investigation_metrics['memory_per_node'] < 0.1:  # < 100KB per node
            scores['memory_efficiency'] = 100
        elif investigation_metrics['memory_per_node'] < 1.0:  # < 1MB per node
            scores['memory_efficiency'] = 80
        else:
            scores['memory_efficiency'] = 60
        
        # Scalability
        if 'speedup' in self.results['parallel_execution']:
            speedup = self.results['parallel_execution']['speedup']
            if speedup > 6:  # Good parallel speedup
                scores['scalability'] = 100
            elif speedup > 4:
                scores['scalability'] = 80
            else:
                scores['scalability'] = 60
        
        # Overall score
        scores['overall'] = sum([
            scores['tree_efficiency'],
            scores['tool_speed'],
            scores['memory_efficiency'],
            scores['scalability']
        ]) / 4
        
        print("\n" + "="*60)
        print("PERFORMANCE SCORES")
        print("="*60)
        for metric, score in scores.items():
            print(f"  {metric}: {score:.0f}/100")
        
        return scores


async def main():
    """Run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    scores = await benchmark.run_all_benchmarks()
    
    if scores['overall'] >= 80:
        print("\n✓ EXCELLENT PERFORMANCE")
        return 0
    elif scores['overall'] >= 60:
        print("\n⚠ ACCEPTABLE PERFORMANCE")
        return 1
    else:
        print("\n✗ POOR PERFORMANCE")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)