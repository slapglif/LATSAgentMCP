#!/usr/bin/env python3
"""
LATS Codebase Analysis MCP Server
Provides deep codebase insights using Language Agent Tree Search
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional, Union
from fastmcp import FastMCP
import re
from lats_langgraph import LATSAgent
from filesystem_tools import (
    read_file_with_lines, 
    list_directory_tree, 
    search_in_files,
    analyze_code_structure,
    find_dependencies
)

# Initialize MCP server
mcp = FastMCP("LATS Codebase Analyzer")

# Utility functions for pagination, filtering, and formatting
def _apply_pagination(data: List[Any], page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    """Apply pagination to data"""
    if not isinstance(data, list):
        return {"data": data, "pagination": None}
    
    total_items = len(data)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_data = data[start_idx:end_idx]
    
    return {
        "data": paginated_data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": (total_items + page_size - 1) // page_size,
            "has_next": end_idx < total_items,
            "has_previous": page > 1
        }
    }

def _apply_filters(data: List[str], filters: Dict[str, Any]) -> List[str]:
    """Apply filters to string data"""
    if not filters or not isinstance(data, list):
        return data
    
    filtered_data = data
    
    # Text filters
    if "contains" in filters:
        pattern = filters["contains"]
        filtered_data = [item for item in filtered_data if pattern.lower() in str(item).lower()]
    
    if "regex" in filters:
        pattern = re.compile(filters["regex"], re.IGNORECASE)
        filtered_data = [item for item in filtered_data if pattern.search(str(item))]
    
    if "exclude" in filters:
        exclude_pattern = filters["exclude"]
        filtered_data = [item for item in filtered_data if exclude_pattern.lower() not in str(item).lower()]
    
    # Length filters
    if "min_length" in filters:
        min_len = filters["min_length"]
        filtered_data = [item for item in filtered_data if len(str(item)) >= min_len]
    
    if "max_length" in filters:
        max_len = filters["max_length"]
        filtered_data = [item for item in filtered_data if len(str(item)) <= max_len]
    
    return filtered_data

def _format_output(data: Any, output_format: str = "json") -> Any:
    """Format output according to specified format"""
    if output_format == "json":
        return data
    elif output_format == "text":
        if isinstance(data, dict):
            return "\n".join([f"{k}: {v}" for k, v in data.items()])
        elif isinstance(data, list):
            return "\n".join([str(item) for item in data])
        else:
            return str(data)
    elif output_format == "summary":
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"Dictionary with {len(keys)} keys: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return f"Data type: {type(data).__name__}"
    elif output_format == "markdown":
        if isinstance(data, dict):
            lines = ["## Data"]
            for k, v in data.items():
                lines.append(f"**{k}**: {v}")
            return "\n".join(lines)
        elif isinstance(data, list):
            return "\n".join([f"- {item}" for item in data[:20]])  # Limit for readability
        else:
            return f"```\n{data}\n```"
    else:
        return data

@mcp.tool()
async def analyze_codebase(
    task: str,
    codebase_path: str = ".",
    generate_report: bool = True,
    max_depth: int = None,
    max_nodes: int = None,
    streaming: bool = True,
    adaptive_mode: bool = True,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Perform deep codebase analysis using LATS (Language Agent Tree Search)
    
    Args:
        task: What to analyze (e.g., "Understand the authentication system", "Find performance bottlenecks", "Analyze data flow")
        codebase_path: Path to the codebase to analyze (default: current directory)
        generate_report: Whether to generate a detailed markdown report (default: True)
        max_depth: Maximum search depth (auto-determined if None)
        max_nodes: Maximum nodes to explore (auto-determined if None) 
        streaming: Enable real-time progress updates (default: True)
        adaptive_mode: Enable intelligent adaptation based on codebase characteristics (default: True)
        output_format: Output format (json, text, summary, markdown)
    
    Returns:
        Comprehensive analysis results with insights, findings, and recommendations
    """
    try:
        # Analyze codebase characteristics for adaptive configuration
        codebase_info = None
        if adaptive_mode:
            try:
                original_dir = os.getcwd()
                if codebase_path != "." and os.path.exists(codebase_path):
                    os.chdir(codebase_path)
                
                codebase_info = await _analyze_codebase_characteristics()
                os.chdir(original_dir)
            except Exception as e:
                print(f"⚠️ Could not analyze codebase characteristics: {e}")
        
        # Determine optimal configuration
        optimal_config = _determine_optimal_config(task, codebase_info, max_depth, max_nodes)
        
        # Setup streaming progress callback
        progress_updates = []
        async def progress_callback(progress_data):
            progress_updates.append(progress_data)
            if streaming:
                # Force immediate output for streaming
                event_type = progress_data['event_type']
                data = progress_data['data']
                
                if event_type == "investigation_started":
                    print(f"🚀 Starting investigation: {data.get('task', '')}")
                    print(f"   Tools available: {data.get('tools_available', 0)}")
                    print(f"   Max depth: {data.get('max_depth', 'N/A')}")
                    
                elif event_type == "node_selected":
                    print(f"🎯 Selected node: {data.get('action', 'N/A')} (depth: {data.get('depth', 0)})")
                    print(f"   UCT score: {data.get('uct_score', 0):.3f}")
                    print(f"   Tree stats: {data.get('total_nodes', 0)} nodes, height: {data.get('tree_height', 0)}")
                    
                elif event_type == "action_executing":
                    print(f"⚡ Executing: {data.get('action', 'Unknown action')}")
                    
                elif event_type == "action_completed":
                    if data.get('success', False):
                        preview = data.get('observation_preview', '')
                        if preview:
                            print(f"✅ Completed: {preview[:100]}{'...' if len(preview) > 100 else ''}")
                        else:
                            print(f"✅ Action completed successfully")
                    else:
                        error = data.get('error', 'Unknown error')
                        print(f"❌ Action failed: {error}")
                        
                elif event_type == "investigation_completed":
                    print(f"🏁 Investigation completed!")
                    print(f"   Duration: {data.get('duration_seconds', 0):.2f}s")
                    print(f"   Nodes explored: {data.get('nodes_explored', 0)}")
                    print(f"   Max depth: {data.get('max_depth', 0)}")
                    print(f"   Best score: {data.get('best_score', 0):.2f}")
                    
                elif event_type == "report_generating":
                    status = data.get('status', '')
                    if status == 'started':
                        print(f"📝 Generating detailed report...")
                    elif status == 'completed':
                        print(f"✅ Report generated successfully")
                
                # Force flush output
                import sys
                sys.stdout.flush()
        
        # Initialize LATS agent with adaptive configuration
        agent = LATSAgent(
            max_depth=optimal_config.get('max_depth', 8),
            progress_callback=progress_callback if streaming else None,
            max_context_tokens=28000
        )
        
        # Available analysis tools
        tools = [
            read_file_with_lines,
            list_directory_tree, 
            search_in_files,
            analyze_code_structure,
            find_dependencies
        ]
        
        print(f"🔍 Starting LATS analysis: {task}")
        print(f"📁 Target codebase: {codebase_path}")
        print(f"⚙️ Configuration: max_depth={optimal_config['max_depth']}, adaptive={adaptive_mode}")
        
        # Change to target directory if specified
        original_dir = os.getcwd()
        if codebase_path != "." and os.path.exists(codebase_path):
            os.chdir(codebase_path)
        
        try:
            # Run deep analysis
            result = await agent.investigate(task, tools, generate_report=generate_report)
            
            # Enhance result with codebase-specific insights
            result["analysis_type"] = "Deep Codebase Analysis"
            result["target_path"] = codebase_path
            result["methodology"] = "LATS (Language Agent Tree Search)"
            result["configuration"] = optimal_config
            result["progress_updates"] = progress_updates if streaming else []
            result["codebase_characteristics"] = codebase_info
            
            return _format_output(result, output_format)
            
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "task": task,
            "codebase_path": codebase_path
        }

@mcp.tool()
async def quick_codebase_overview(
    codebase_path: str = ".",
    page: int = 1,
    page_size: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Get a quick overview of codebase structure and key files
    
    Args:
        codebase_path: Path to the codebase (default: current directory)
        page: Page number for pagination (default: 1)
        page_size: Items per page (default: 50)
        filters: Filter options (contains, regex, exclude, min_length, max_length)
        output_format: Output format (json, text, summary, markdown)
        
    Returns:
        Quick overview including file structure, key files, and basic metrics with pagination
    """
    try:
        import os
        original_dir = os.getcwd()
        if codebase_path != ".":
            os.chdir(codebase_path)
            
        try:
            # Get directory structure
            structure = list_directory_tree(".", max_depth=3)
            
            # Count files by type
            file_counts = {}
            total_files = 0
            
            for line in structure.split('\n'):
                if '.' in line and not line.strip().startswith('├──') and not line.strip().startswith('└──'):
                    continue
                if '(' in line and 'B)' in line:  # File with size
                    total_files += 1
                    if '[' in line and ']' in line:
                        file_type = line.split('[')[1].split(']')[0]
                        file_counts[file_type] = file_counts.get(file_type, 0) + 1
            
            # Prepare data for pagination/filtering
            structure_lines = structure.split('\n')
            if filters:
                structure_lines = _apply_filters(structure_lines, filters)
            
            # Apply pagination to structure lines
            paginated_structure = _apply_pagination(structure_lines, page, page_size)
            
            result = {
                "codebase_path": codebase_path,
                "structure_lines": paginated_structure["data"],
                "structure": "\n".join(paginated_structure["data"]),
                "file_counts": file_counts,
                "total_files": total_files,
                "overview_type": "Quick Structure Analysis",
                "pagination": paginated_structure["pagination"]
            }
            
            return _format_output(result, output_format)
            
        finally:
            os.chdir(original_dir)
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed", 
            "codebase_path": codebase_path
        }

@mcp.tool()
async def find_code_patterns(
    pattern: str,
    codebase_path: str = ".",
    file_extension: str = "*.py",
    page: int = 1,
    page_size: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Search for specific code patterns across the codebase
    
    Args:
        pattern: Pattern to search for (regex supported)
        codebase_path: Path to search in (default: current directory)
        file_extension: File types to search (default: *.py)
        page: Page number for pagination (default: 1)
        page_size: Items per page (default: 50)
        filters: Filter options (contains, regex, exclude, min_length, max_length)
        output_format: Output format (json, text, summary, markdown)
        
    Returns:
        All matches with file locations and context, with pagination and filtering
    """
    try:
        import os
        original_dir = os.getcwd()
        if codebase_path != ".":
            os.chdir(codebase_path)
            
        try:
            results = search_in_files(pattern, ".", file_extension, case_sensitive=False)
            
            # Convert results to lines for pagination/filtering
            result_lines = results.split('\n') if isinstance(results, str) else [str(results)]
            if filters:
                result_lines = _apply_filters(result_lines, filters)
            
            # Apply pagination
            paginated_results = _apply_pagination(result_lines, page, page_size)
            
            result = {
                "pattern": pattern,
                "file_extension": file_extension,
                "codebase_path": codebase_path,
                "results": "\n".join(paginated_results["data"]),
                "result_lines": paginated_results["data"],
                "search_type": "Pattern Search",
                "pagination": paginated_results["pagination"]
            }
            
            return _format_output(result, output_format)
            
        finally:
            os.chdir(original_dir)
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "pattern": pattern,
            "codebase_path": codebase_path
        }

@mcp.tool()
async def analyze_file_structure(
    file_path: str,
    page: int = 1,
    page_size: int = 50,
    filters: Optional[Dict[str, Any]] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Analyze the structure of a specific code file
    
    Args:
        file_path: Path to the file to analyze
        page: Page number for pagination (default: 1)
        page_size: Items per page (default: 50)
        filters: Filter options (contains, regex, exclude, min_length, max_length)
        output_format: Output format (json, text, summary, markdown)
        
    Returns:
        Detailed analysis of classes, functions, imports, and dependencies with pagination
    """
    try:
        # Read the file content
        content = read_file_with_lines(file_path)
        
        # Analyze code structure
        structure = analyze_code_structure(file_path)
        
        # Find dependencies
        dependencies = find_dependencies(file_path)
        
        # Prepare content lines for pagination/filtering
        content_lines = content.split('\n') if isinstance(content, str) else [str(content)]
        if filters:
            content_lines = _apply_filters(content_lines, filters)
        
        # Apply pagination to content
        paginated_content = _apply_pagination(content_lines, page, page_size)
        
        result = {
            "file_path": file_path,
            "content_preview": "\n".join(paginated_content["data"]),
            "content_lines": paginated_content["data"],
            "structure": structure,
            "dependencies": dependencies,
            "analysis_type": "File Structure Analysis",
            "pagination": paginated_content["pagination"]
        }
        
        return _format_output(result, output_format)
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "file_path": file_path
        }

@mcp.tool()
async def get_analysis_recommendations(
    task: str,
    codebase_path: str = ".",
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Get intelligent recommendations for LATS analysis configuration
    
    Args:
        task: What you want to analyze
        codebase_path: Path to the codebase
        output_format: Output format (json, text, summary, markdown)
        
    Returns:
        Recommended configuration and analysis strategy
    """
    try:
        original_dir = os.getcwd()
        if codebase_path != "." and os.path.exists(codebase_path):
            os.chdir(codebase_path)
            
        # Analyze codebase characteristics
        codebase_info = await _analyze_codebase_characteristics()
        
        # Get optimal configuration
        config = _determine_optimal_config(task, codebase_info, None, None)
        
        # Generate intelligent recommendations
        recommendations = _generate_analysis_recommendations(task, codebase_info, config)
        
        os.chdir(original_dir)
        
        result = {
            "task": task,
            "codebase_characteristics": codebase_info,
            "recommended_config": config,
            "recommendations": recommendations,
            "analysis_strategy": recommendations.get("strategy_explanation", ""),
            "estimated_duration": recommendations.get("estimated_duration", "unknown")
        }
        
        return _format_output(result, output_format)
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "task": task
        }

@mcp.tool() 
async def analyze_codebase_adaptive(
    task: str,
    codebase_path: str = ".",
    urgency: str = "normal",  # low, normal, high
    focus: str = "balanced",  # focused, balanced, comprehensive
    streaming: bool = True,
    output_format: str = "json"
) -> Dict[str, Any]:
    """
    Perform adaptive codebase analysis with intelligent configuration
    
    Args:
        task: What to analyze
        codebase_path: Path to the codebase
        urgency: Analysis urgency (low=thorough, normal=balanced, high=quick)
        focus: Analysis focus (focused=narrow&deep, balanced=mixed, comprehensive=broad)
        streaming: Enable real-time progress updates
        output_format: Output format (json, text, summary, markdown)
        
    Returns:
        Analysis results with adaptive configuration applied
    """
    # Get recommendations first
    recommendations = await get_analysis_recommendations(task, codebase_path)
    
    if "error" in recommendations:
        return recommendations
        
    # Adjust configuration based on urgency and focus
    config = recommendations["recommended_config"].copy()
    
    # Urgency adjustments
    if urgency == "high":
        config["max_depth"] = min(config["max_depth"], 6)
        config["strategy"] = "quick_scan"
    elif urgency == "low":
        config["max_depth"] = max(config["max_depth"], 10)
        config["strategy"] = "thorough"
        
    # Focus adjustments
    if focus == "focused":
        config["max_depth"] = min(config["max_depth"], 8)
    elif focus == "comprehensive":
        config["max_depth"] = max(config["max_depth"], 12)
    
    # Run analysis with adaptive configuration
    result = await analyze_codebase(
        task=task,
        codebase_path=codebase_path,
        generate_report=True,
        max_depth=config["max_depth"],
        max_nodes=config.get("max_nodes"),
        streaming=streaming,
        adaptive_mode=True
    )
    
    # Enhance result with recommendation context
    result["adaptive_config"] = {
        "urgency": urgency,
        "focus": focus,
        "applied_config": config,
        "original_recommendations": recommendations["recommendations"]
    }
    
    return _format_output(result, output_format)

async def _analyze_codebase_characteristics() -> Dict[str, Any]:
    """Analyze codebase to determine optimal LATS configuration"""
    characteristics = {
        "total_files": 0,
        "total_size_mb": 0,
        "languages": {},
        "complexity_indicators": {
            "deep_nesting": False,
            "many_dependencies": False,
            "large_files": False,
            "test_coverage": False
        }
    }
    
    try:
        # Quick directory analysis
        structure = list_directory_tree(".", max_depth=3)
        lines = structure.split('\n')
        
        for line in lines:
            if '(' in line and 'B)' in line:
                characteristics["total_files"] += 1
                
                # Extract file size
                if 'KB)' in line:
                    size_str = line.split('(')[1].split('KB)')[0]
                    try:
                        size_kb = float(size_str)
                        characteristics["total_size_mb"] += size_kb / 1024
                        if size_kb > 100:  # Large file indicator
                            characteristics["complexity_indicators"]["large_files"] = True
                    except:
                        pass
                
                # Extract language
                if '[' in line and ']' in line:
                    lang = line.split('[')[1].split(']')[0]
                    characteristics["languages"][lang] = characteristics["languages"].get(lang, 0) + 1
        
        # Check for test files
        test_results = search_in_files("test", ".", "*.py", case_sensitive=False)
        if "test" in test_results.lower() or "spec" in test_results.lower():
            characteristics["complexity_indicators"]["test_coverage"] = True
            
        # Check for complex dependencies
        if characteristics["total_files"] > 50:
            characteristics["complexity_indicators"]["many_dependencies"] = True
            
    except Exception as e:
        print(f"⚠️ Error analyzing codebase: {e}")
    
    return characteristics

def _determine_optimal_config(task: str, codebase_info: Dict[str, Any], max_depth: int, max_nodes: int) -> Dict[str, Any]:
    """Determine optimal LATS configuration based on task and codebase"""
    
    # Default configuration
    config = {
        "max_depth": 8,
        "max_nodes": 100,
        "strategy": "balanced"
    }
    
    # Override with user preferences
    if max_depth is not None:
        config["max_depth"] = max_depth
    if max_nodes is not None:
        config["max_nodes"] = max_nodes
        
    # If no codebase info, return defaults
    if not codebase_info:
        return config
    
    # Adaptive configuration based on codebase characteristics
    total_files = codebase_info.get("total_files", 0)
    complexity = codebase_info.get("complexity_indicators", {})
    
    # Adjust depth based on codebase size
    if total_files < 20:
        config["max_depth"] = min(config["max_depth"], 6)  # Small codebase - less depth needed
        config["strategy"] = "focused"
    elif total_files > 100:
        config["max_depth"] = max(config["max_depth"], 12)  # Large codebase - more depth may be useful
        config["strategy"] = "comprehensive"
    
    # Adjust based on complexity indicators
    if complexity.get("large_files", False):
        config["max_depth"] += 2  # Large files may need deeper analysis
        
    if complexity.get("many_dependencies", False):
        config["max_depth"] += 1  # Complex dependencies need more exploration
        
    # Task-specific adjustments
    task_lower = task.lower()
    if "performance" in task_lower or "optimization" in task_lower:
        config["max_depth"] += 2  # Performance analysis needs deeper investigation
        config["strategy"] = "deep_performance"
    elif "security" in task_lower or "vulnerability" in task_lower:
        config["max_depth"] += 3  # Security analysis needs thorough exploration
        config["strategy"] = "security_focused"
    elif "understand" in task_lower or "architecture" in task_lower:
        config["max_depth"] = max(config["max_depth"], 10)  # Architecture understanding needs breadth
        config["strategy"] = "architectural"
    elif "bug" in task_lower or "error" in task_lower or "fix" in task_lower:
        config["max_depth"] += 1  # Bug hunting needs focused depth
        config["strategy"] = "debugging"
    
    # Cap the maximum depth to reasonable limits
    config["max_depth"] = min(config["max_depth"], 20)
    config["max_depth"] = max(config["max_depth"], 3)
    
    return config

def _generate_analysis_recommendations(task: str, codebase_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate intelligent analysis recommendations for the calling agent"""
    
    recommendations = {
        "strategy_explanation": "",
        "estimated_duration": "2-5 minutes",
        "key_considerations": [],
        "suggested_parameters": config,
        "alternative_approaches": []
    }
    
    # Analysis based on task type
    task_lower = task.lower()
    total_files = codebase_info.get("total_files", 0)
    complexity = codebase_info.get("complexity_indicators", {})
    languages = codebase_info.get("languages", {})
    
    if "performance" in task_lower or "optimization" in task_lower:
        recommendations["strategy_explanation"] = f"Performance analysis requires deep investigation (depth={config['max_depth']}). Will focus on bottlenecks, loops, and resource usage patterns."
        recommendations["estimated_duration"] = "5-10 minutes"
        recommendations["key_considerations"].extend([
            "Will analyze computational complexity in code",
            "Look for memory leaks and resource management",
            "Examine database queries and I/O operations"
        ])
        recommendations["alternative_approaches"].append({
            "name": "Quick Performance Scan",
            "params": {"max_depth": 6, "focus": "focused"},
            "description": "Faster analysis focusing only on obvious bottlenecks"
        })
        
    elif "security" in task_lower or "vulnerability" in task_lower:
        recommendations["strategy_explanation"] = f"Security analysis needs thorough exploration (depth={config['max_depth']}). Will examine input validation, authentication, and data handling."
        recommendations["estimated_duration"] = "8-15 minutes"
        recommendations["key_considerations"].extend([
            "Will check for input validation issues",
            "Examine authentication and authorization",
            "Look for SQL injection and XSS vulnerabilities",
            "Review cryptographic implementations"
        ])
        recommendations["alternative_approaches"].append({
            "name": "Critical Security Only",
            "params": {"max_depth": 8, "focus": "focused"},
            "description": "Focus only on high-risk security patterns"
        })
        
    elif "understand" in task_lower or "architecture" in task_lower:
        recommendations["strategy_explanation"] = f"Architecture analysis uses broad exploration (depth={config['max_depth']}). Will map component relationships and design patterns."
        recommendations["estimated_duration"] = "6-12 minutes"
        recommendations["key_considerations"].extend([
            "Will map component dependencies",
            "Identify design patterns and architectural style",
            "Examine data flow and system boundaries"
        ])
        recommendations["alternative_approaches"].append({
            "name": "High-Level Overview",
            "params": {"max_depth": 5, "focus": "comprehensive"},
            "description": "Quick architectural overview without deep details"
        })
        
    elif "bug" in task_lower or "error" in task_lower or "fix" in task_lower:
        recommendations["strategy_explanation"] = f"Bug hunting uses focused deep search (depth={config['max_depth']}). Will trace error patterns and edge cases."
        recommendations["estimated_duration"] = "4-8 minutes"
        recommendations["key_considerations"].extend([
            "Will trace error propagation paths",
            "Look for unhandled exceptions and edge cases",
            "Examine state management and concurrency issues"
        ])
        
    else:
        recommendations["strategy_explanation"] = f"General analysis using balanced approach (depth={config['max_depth']}). Will provide comprehensive codebase understanding."
        recommendations["estimated_duration"] = "3-7 minutes"
    
    # Adjust based on codebase characteristics
    if total_files > 100:
        recommendations["estimated_duration"] = "8-15 minutes"
        recommendations["key_considerations"].append(f"Large codebase ({total_files} files) - analysis will be thorough but may take longer")
        
    elif total_files < 10:
        recommendations["estimated_duration"] = "2-4 minutes"
        recommendations["key_considerations"].append(f"Small codebase ({total_files} files) - quick analysis possible")
        
    if complexity.get("large_files", False):
        recommendations["key_considerations"].append("Large files detected - will perform detailed analysis of complex modules")
        
    if complexity.get("test_coverage", False):
        recommendations["key_considerations"].append("Test files found - will analyze test patterns and coverage")
        
    # Language-specific considerations
    if "python" in languages:
        recommendations["key_considerations"].append("Python codebase - will check for Pythonic patterns and common issues")
    if "javascript" in languages or "typescript" in languages:
        recommendations["key_considerations"].append("JS/TS codebase - will examine async patterns and DOM manipulation")
        
    return recommendations

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()