#!/usr/bin/env python3
"""
LATS MCP Server
Provides MCP interface for LATS agent with filesystem tools and memory
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from filesystem_tools import create_filesystem_tools
from lats_core import LATSAlgorithm, LATSConfig, TreeNode
from memory_manager import InvestigationMemory, MemoryManager


# MCP Request/Response Models
class InvestigateRequest(BaseModel):
    """Request to investigate a task"""
    task: str = Field(description="Task to investigate")
    max_depth: int = Field(default=5, description="Maximum search depth")
    max_iterations: int = Field(default=10, description="Maximum iterations")
    use_memory: bool = Field(default=True, description="Use memory for context")


class InvestigationStatus(BaseModel):
    """Current investigation status"""
    task: str
    status: str  # 'idle', 'running', 'completed', 'failed'
    progress: Dict[str, Any]
    current_branch: Optional[List[str]] = None
    best_score: float = 0.0


class InsightResponse(BaseModel):
    """Investigation insights"""
    task: str
    solution_path: List[Dict[str, Any]]
    file_references: List[str]
    explored_branches: List[List[str]]
    confidence_score: float
    is_complete: bool
    suggestions: List[str]


# LATS MCP Server
class LATSMCPServer:
    """MCP Server for LATS agent"""
    
    def __init__(self):
        self.mcp = FastMCP("LATS Intelligence Agent")
        self.mcp.description = "Advanced code investigation agent using Language Agent Tree Search"
        
        # Initialize components
        self.config = LATSConfig()
        self.algorithm = LATSAlgorithm(self.config)
        self.memory_manager = MemoryManager()
        self.filesystem_tools = create_filesystem_tools()
        
        # State tracking
        self.current_investigation: Optional[Dict[str, Any]] = None
        self.investigation_history: List[Dict[str, Any]] = []
        
        # Register MCP tools
        self._register_tools()
        
        # Register resources
        self._register_resources()
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.mcp.tool
        async def investigate(request: InvestigateRequest) -> InsightResponse:
            """
            Run LATS investigation on a task.
            Performs parallel exploration of the codebase to find solutions.
            """
            try:
                # Check for similar past investigations
                similar = []
                if request.use_memory:
                    similar = await self.memory_manager.search_similar_investigations(
                        request.task, limit=3
                    )
                
                # Get pattern suggestions
                suggestions = await self.memory_manager.get_pattern_suggestions(request.task)
                
                # Update current investigation
                self.current_investigation = {
                    'task': request.task,
                    'status': 'running',
                    'start_time': datetime.now().isoformat(),
                    'config': {
                        'max_depth': request.max_depth,
                        'max_iterations': request.max_iterations
                    }
                }
                
                # Run LATS algorithm
                root = TreeNode()
                best_score = 0.0
                
                for iteration in range(request.max_iterations):
                    # Select
                    current = await self.algorithm.select_node(root)
                    
                    # Expand
                    children = await self.algorithm.expand_node(
                        current, request.task, self.filesystem_tools
                    )
                    
                    # Simulate and reflect for each child
                    for child in children:
                        # Execute action
                        observation = await self._execute_tool_action(child.action)
                        child.observation = observation
                        
                        # Reflect and score
                        score = await self.algorithm.reflect_on_node(child, request.task)
                        
                        # Backpropagate
                        self.algorithm.backpropagate(child, score)
                        
                        if score > best_score:
                            best_score = score
                        
                        # Check if solution found
                        if child.is_terminal:
                            break
                    
                    # Check termination
                    if any(child.is_terminal for child in children):
                        break
                
                # Extract insights
                insights = self.algorithm.extract_insights(root, request.task)
                
                # Store in memory if successful
                if insights['is_complete'] or insights['statistics']['best_score'] >= 7.0:
                    memory = InvestigationMemory(
                        task=request.task,
                        solution_path=insights['solution_path'],
                        file_references=insights['file_references'],
                        insights=insights['statistics'],
                        score=insights['statistics']['best_score'],
                        is_complete=insights['is_complete']
                    )
                    await self.memory_manager.store_investigation(memory)
                
                # Update status
                self.current_investigation['status'] = 'completed'
                self.current_investigation['end_time'] = datetime.now().isoformat()
                self.investigation_history.append(self.current_investigation)
                
                # Add suggestions from patterns
                all_suggestions = suggestions + [
                    f"Found {len(insights['file_references'])} relevant files",
                    f"Explored {len(insights['explored_branches'])} branches",
                    f"Best confidence: {insights['statistics']['best_score']:.1f}/10"
                ]
                
                # Add insights from similar investigations
                if similar:
                    all_suggestions.append(f"Found {len(similar)} similar past investigations")
                
                return InsightResponse(
                    task=request.task,
                    solution_path=insights['solution_path'],
                    file_references=insights['file_references'],
                    explored_branches=insights['explored_branches'][:5],  # Limit branches
                    confidence_score=insights['statistics']['best_score'],
                    is_complete=insights['is_complete'],
                    suggestions=all_suggestions
                )
                
            except Exception as e:
                self.current_investigation['status'] = 'failed'
                self.current_investigation['error'] = str(e)
                raise e
        
        @self.mcp.tool
        async def get_status() -> InvestigationStatus:
            """Get current investigation status"""
            if not self.current_investigation:
                return InvestigationStatus(
                    task="",
                    status="idle",
                    progress={}
                )
            
            return InvestigationStatus(
                task=self.current_investigation.get('task', ''),
                status=self.current_investigation.get('status', 'unknown'),
                progress=self.current_investigation.get('config', {}),
                best_score=self.current_investigation.get('best_score', 0.0)
            )
        
        @self.mcp.tool
        async def search_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
            """Search memory for relevant past investigations"""
            return await self.memory_manager.search_similar_investigations(query, limit)
        
        @self.mcp.tool
        async def get_insights(context: str) -> List[str]:
            """Get relevant insights for a context"""
            return await self.memory_manager.get_relevant_insights(context)
        
        @self.mcp.tool
        async def store_insight(insight: str, context: str, tags: List[str] = None) -> str:
            """Store a new insight for future reference"""
            return await self.memory_manager.store_insight(insight, context, tags)
        
        @self.mcp.tool
        async def get_history(limit: int = 10) -> List[Dict[str, Any]]:
            """Get investigation history"""
            return await self.memory_manager.get_investigation_history(limit)
        
        @self.mcp.tool
        async def analyze_file(file_path: str) -> Dict[str, Any]:
            """Quick file analysis without full LATS investigation"""
            results = {}
            
            # Read file
            read_tool = next(t for t in self.filesystem_tools if t.name == "read_file")
            content = await self._execute_tool(read_tool, {'file_path': file_path})
            results['content_preview'] = content[:500] if content else ""
            
            # Analyze structure if Python
            if file_path.endswith('.py'):
                analyze_tool = next(t for t in self.filesystem_tools if t.name == "analyze_structure")
                structure = await self._execute_tool(analyze_tool, {'file_path': file_path})
                results['structure'] = structure
            
            # Find dependencies
            deps_tool = next(t for t in self.filesystem_tools if t.name == "find_dependencies")
            deps = await self._execute_tool(deps_tool, {'file_path': file_path})
            results['dependencies'] = deps
            
            return results
        
        @self.mcp.tool
        async def parallel_search(patterns: List[str], directory: str = ".") -> Dict[str, List[str]]:
            """Search for multiple patterns in parallel"""
            search_tool = next(t for t in self.filesystem_tools if t.name == "search_files")
            
            tasks = []
            for pattern in patterns:
                task = self._execute_tool(search_tool, {
                    'pattern': pattern,
                    'directory': directory,
                    'max_results': 20
                })
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            return {
                pattern: self._extract_file_refs(result)
                for pattern, result in zip(patterns, results)
            }
    
    def _register_resources(self):
        """Register MCP resources"""
        
        @self.mcp.resource("config://lats")
        async def get_config() -> Dict[str, Any]:
            """Get LATS configuration"""
            return self.config.model_dump()
        
        @self.mcp.resource("memory://stats")
        async def get_memory_stats() -> Dict[str, Any]:
            """Get memory statistics"""
            history = await self.memory_manager.get_investigation_history(100)
            return {
                'total_investigations': len(history),
                'completed': sum(1 for h in history if h.get('is_complete')),
                'average_score': sum(h.get('score', 0) for h in history) / max(len(history), 1)
            }
    
    async def _execute_tool_action(self, action: str) -> str:
        """Execute a tool action string"""
        try:
            # Parse action format: tool_name(arg1="value1", arg2="value2")
            if '(' not in action:
                return f"Error: Invalid action format: {action}"
            
            tool_name = action.split('(')[0].strip()
            args_str = action[action.index('(')+1:action.rindex(')')].strip()
            
            # Find tool
            tool = next((t for t in self.filesystem_tools if t.name == tool_name), None)
            if not tool:
                return f"Error: Tool {tool_name} not found"
            
            # Parse arguments
            args = {}
            if args_str:
                # Simple argument parsing (can be improved)
                parts = args_str.split(',')
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        args[key] = value
            
            # Execute tool
            return await self._execute_tool(tool, args)
            
        except Exception as e:
            return f"Error executing action: {str(e)}"
    
    async def _execute_tool(self, tool: Any, args: Dict[str, Any]) -> str:
        """Execute a tool with arguments"""
        try:
            # Tools expect sync execution, wrap in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.func, args)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _extract_file_refs(self, search_result: str) -> List[str]:
        """Extract file references from search result"""
        import re
        refs = []
        
        # Look for file:line patterns
        pattern = r'([^\s:]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb)):(\d+)'
        matches = re.findall(pattern, search_result)
        
        for match in matches:
            refs.append(f"{match[0]}:{match[2]}")
        
        return refs
    
    def run(self):
        """Run the MCP server"""
        print("Starting LATS MCP Server...", file=sys.stderr)
        print(f"Model: {self.config.model_name}", file=sys.stderr)
        print(f"Ollama URL: {self.config.base_url}", file=sys.stderr)
        print("Ready for investigations!", file=sys.stderr)
        
        # Run with STDIO transport (default)
        self.mcp.run()


# Main entry point
def main():
    """Main entry point"""
    server = LATSMCPServer()
    server.run()


if __name__ == "__main__":
    main()