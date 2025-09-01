#!/usr/bin/env python3
"""
LATS (Language Agent Tree Search) - Official LangGraph Implementation
Based on LangGraph's StateGraph architecture
"""

import asyncio
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Literal
from uuid import uuid4
from datetime import datetime
from pathlib import Path

from loguru import logger

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

# Import langmem for context management (core dependency)
import langmem


class Node(BaseModel):
    """Node in LATS tree - Pydantic model for serialization"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    action: Optional[str] = None
    observation: Optional[str] = None
    parent_id: Optional[str] = None  # Store parent ID instead of reference
    child_ids: List[str] = Field(default_factory=list)  # Store child IDs
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    is_solved: bool = False
    reflection: Optional[str] = None
    
    # Runtime storage for actual references (not serialized)
    parent_ref: Optional['Node'] = Field(default=None, exclude=True)
    children_ref: List['Node'] = Field(default_factory=list, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def parent(self) -> Optional['Node']:
        return self.parent_ref
    
    @parent.setter
    def parent(self, value: Optional['Node']):
        self.parent_ref = value
        if value:
            self.parent_id = value.id
    
    @property
    def children(self) -> List['Node']:
        return self.children_ref
    
    @property
    def height(self) -> int:
        """Max depth of tree from this node"""
        if not self.children_ref:
            return self.depth
        return max(child.height for child in self.children_ref)
    
    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Tree score for node selection"""
        if self.visits == 0:
            return float('inf')
            
        exploitation = self.value / self.visits
        if self.parent is None:
            return exploitation
            
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self) -> Optional['Node']:
        """Get child with highest UCT score"""
        if not self.children_ref:
            return None
        return max(self.children_ref, key=lambda x: x.uct_score())
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree"""
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(value)
    
    @property 
    def average_value(self) -> float:
        """Get average value (for proper scoring)"""
        return self.value / max(self.visits, 1)
    
    def add_child(self, action: str) -> 'Node':
        """Add child node"""
        child = Node(
            action=action,
            parent_id=self.id,
            depth=self.depth + 1
        )
        child._parent = self
        self.children_ref.append(child)
        self.child_ids.append(child.id)
        return child


class TreeState(TypedDict):
    """State for LangGraph StateGraph"""
    # The full tree
    root: Node
    # The original input
    input: str
    # Tool names (serializable) - actual tools stored on agent
    tool_names: List[str]


class LATSAgent:
    """LATS Agent using LangGraph StateGraph with comprehensive logging"""
    
    def __init__(self, log_file: str = None, checkpoint_db: str = None, max_depth: int = 10, progress_callback=None, max_context_tokens: int = 28000):
        """Initialize LATS Agent with unlimited deep investigation capability"""
        self.session_id = str(uuid4())[:8]
        self.tools = None  # Will be set during investigate
        self.investigation_history = []  # Track what has been investigated
        self.max_depth = max_depth  # Maximum depth for exploration
        self.progress_callback = progress_callback  # Real-time progress updates
        
        # Context management settings
        self.max_context_tokens = max_context_tokens  # Reserve 4k tokens for output
        self.current_context_size = 0
        self.context_summaries = {}  # Store summaries of large content
        
        # Initialize langmem for context management
        self.memory_manager = langmem.create_memory_manager(
            "ollama:gpt-oss",  # Use the same model as the agent
            instructions="Store and summarize code analysis context, preserving key insights and findings.",
            enable_inserts=True,
            enable_updates=True
        )
        
        # Configure logging
        if log_file:
            logger.add(log_file, rotation="100 MB", retention="30 days", 
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        logger.info(f"🚀 Initializing LATS Agent (session: {self.session_id})")
        logger.info(f"📊 Configuration: Deep investigation mode - no artificial limits")
        
        # Setup SQLite checkpointing
        self.checkpointer = None
        self.checkpoint_db = checkpoint_db
        if checkpoint_db:
            logger.info(f"📁 Using SQLite checkpoint: {checkpoint_db}")
        
        self.llm = self._init_llm()
        self.graph = self._build_graph()
        
        logger.success("✅ LATS Agent initialized successfully")
    
    async def _emit_progress(self, event_type: str, data: dict):
        """Emit progress update if callback is available"""
        if self.progress_callback:
            progress_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "data": data
            }
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(progress_data)
                else:
                    self.progress_callback(progress_data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        return len(text.split()) * 1.3  # Rough estimate: ~1.3 tokens per word
    
    async def _manage_context_size(self, new_content: str) -> str:
        """Manage context size and summarize if needed"""
        estimated_tokens = self._estimate_token_count(new_content)
        
        # If content is too large for context, chunk and summarize
        if estimated_tokens > self.max_context_tokens * 0.7:  # Use 70% of available context
            await self._emit_progress("context_management", {
                "action": "chunking_large_content",
                "estimated_tokens": estimated_tokens,
                "max_tokens": self.max_context_tokens
            })
            
            return await self._chunk_and_summarize(new_content)
        
        # Check if adding this would exceed context limit
        if self.current_context_size + estimated_tokens > self.max_context_tokens:
            await self._emit_progress("context_management", {
                "action": "summarizing_to_free_space",
                "current_size": self.current_context_size,
                "new_content_size": estimated_tokens
            })
            
            # Summarize older context using langmem
            await self._summarize_and_store_context()
        
        self.current_context_size += estimated_tokens
        return new_content
    
    async def _chunk_and_summarize(self, content: str, chunk_size: int = 15000) -> str:
        """Chunk large content and create intelligent summaries"""
        if len(content) <= chunk_size:
            return content
            
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        
        summaries = []
        for i, chunk in enumerate(chunks):
            # Use langmem to store and summarize each chunk
            chunk_key = f"chunk_{self.session_id}_{i}_{len(summaries)}"
            
            # Store full chunk in langmem (simplified storage for now)
            # TODO: Implement proper langmem storage integration
            self.context_summaries[chunk_key] = {
                "content": chunk,
                "metadata": {"type": "file_chunk", "chunk_index": i}
            }
            
            # Create summary of chunk
            summary_prompt = f"""Summarize this code/text chunk focusing on key elements:
- Important functions, classes, and methods
- Critical logic and algorithms  
- Security-relevant patterns
- Performance considerations
- Dependencies and imports

Chunk {i+1}/{len(chunks)}:
{chunk}

Provide a concise but comprehensive summary:"""

            try:
                summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                summary = summary_response.content
                summaries.append(f"[Chunk {i+1}]: {summary}")
                
            except Exception as e:
                logger.warning(f"Failed to summarize chunk {i}: {e}")
                # Fallback to basic truncation
                summaries.append(f"[Chunk {i+1}]: {chunk[:500]}...")
        
        combined_summary = "\n".join(summaries)
        
        await self._emit_progress("context_management", {
            "action": "chunking_completed", 
            "original_size": len(content),
            "chunks_created": len(chunks),
            "summary_size": len(combined_summary)
        })
        
        return combined_summary
    
    async def _safe_file_operation(self, operation_func, *args, **kwargs) -> str:
        """Safely execute a file operation with context management"""
        try:
            result = operation_func(*args, **kwargs)
            managed_result = await self._manage_context_size(result)
            return managed_result
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return f"Error executing file operation: {str(e)}"
    
    async def _summarize_and_store_context(self):
        """Summarize current context and store in langmem"""
        if not self.investigation_history:
            return
            
        # Get recent investigation context
        recent_context = self.investigation_history[-10:]  # Last 10 actions
        
        context_text = "\n".join([
            f"Action: {item.get('action', 'N/A')}\nResult: {item.get('observation', 'N/A')[:1000]}"
            for item in recent_context
        ])
        
        # Create context summary
        summary_prompt = f"""Summarize this investigation context, preserving key insights:

{context_text}

Focus on:
- Important findings and discoveries
- Patterns identified
- Security or performance issues found
- Key file locations and structures
- Critical dependencies

Provide a comprehensive summary:"""

        try:
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_response.content
            
            # Store summary in langmem (simplified storage for now)
            # TODO: Implement proper langmem storage integration
            summary_key = f"context_summary_{self.session_id}_{len(self.context_summaries)}"
            self.context_summaries[summary_key] = {
                "content": summary,
                "metadata": {"type": "context_summary", "timestamp": datetime.now().isoformat()}
            }
            
            # Reset current context size
            self.current_context_size = self._estimate_token_count(summary)
            
            await self._emit_progress("context_management", {
                "action": "context_summarized",
                "summary_length": len(summary),
                "new_context_size": self.current_context_size
            })
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {e}")
            # Fallback: just truncate investigation history
            self.investigation_history = self.investigation_history[-5:]
            self.current_context_size = self.current_context_size * 0.5
    
    def _init_llm(self):
        """Initialize real Ollama LLM"""
        return ChatOllama(
            model="gpt-oss",
            base_url="http://localhost:11434",
            temperature=0.7
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for LATS"""
        
        def should_continue(state: TreeState) -> Literal["expand", "__end__"]:
            """Determine whether to continue based on investigation completeness"""
            root = state["root"]
            
            # Stop only if actual vulnerabilities found AND investigation is complete
            if root.is_solved:
                return END
            
            # Ask the LLM scorer if investigation should continue
            # This removes artificial limits and lets AI decide based on findings
            return "expand"
        
        # Build the graph
        builder = StateGraph(TreeState)
        builder.add_node("start", self._generate_initial_response)
        builder.add_node("expand", self._expand_node)
        
        builder.add_edge(START, "start")
        builder.add_conditional_edges("start", should_continue, ["expand", END])
        builder.add_conditional_edges("expand", should_continue, ["expand", END])
        
        # Return uncompiled builder - will compile with checkpointer in investigate method
        return builder
    
    async def _generate_initial_response(self, state: TreeState) -> TreeState:
        """Generate initial response - first step of LATS"""
        task = state['input']
        logger.info(f"🌱 Starting LATS investigation: {task}")
        print(f"🌱 Starting LATS investigation: {task}")
        
        # Initialize root node
        root = state["root"]
        if not root.action:  # First time
            logger.info("🎯 Generating initial actions from root node")
            # Generate initial actions
            actions = await self._generate_actions(root, task, self.tools)
            logger.info(f"🌿 Generated {len(actions)} initial actions: {actions}")
            print(f"🌿 Generated {len(actions)} initial actions")
            
            # Add as children
            for i, action in enumerate(actions):
                child = root.add_child(action)
                logger.debug(f"➕ Added child node {i+1}: {action}")
                # Execute and evaluate
                await self._execute_and_reflect(child, task, self.tools)
                
                # Check if this action found significant insights requiring deep analysis
                insight_info = await self._analyze_for_insights(child, task)
                if insight_info['found']:
                    # Don't stop - trigger deep follow-up analysis
                    logger.success(f"🔍 Insight detected: {insight_info['type']} - Initiating deep analysis")
                    print(f"🔍 Insight found: {insight_info['type']} - Digging deeper...")
                    
                    # Generate follow-up actions for deeper analysis
                    followup_actions = await self._generate_deep_followup_actions(child, insight_info, task)
                    for followup in followup_actions:
                        followup_child = child.add_child(followup)
                        await self._execute_and_reflect(followup_child, task, self.tools)
                        
                    # Mark as solved only after comprehensive analysis
                    if await self._is_investigation_complete(root, task):
                        child.is_solved = True
                        root.is_solved = True
                        logger.success(f"🎉 Comprehensive analysis complete!")
                        print(f"🎉 Deep analysis complete!")
                        break
        
        return state
    
    async def _expand_node(self, state: TreeState) -> TreeState:
        """Expand the most promising node"""
        root = state["root"]
        task = state["input"]
        
        # SELECT: Find most promising leaf node using UCT
        selected = self._select_node(root)
        uct_score = selected.uct_score()
        logger.info(f"🎯 Selected node for expansion: {selected.action} (UCT: {uct_score:.3f}, depth: {selected.depth})")
        print(f"🎯 Selected node: {selected.action} (UCT: {uct_score:.3f})")
        
        # Emit node selection progress
        await self._emit_progress("node_selected", {
            "node_id": selected.id,
            "action": selected.action,
            "depth": selected.depth,
            "uct_score": uct_score,
            "tree_height": root.height,
            "total_nodes": self._count_nodes(root)
        })
        
        # EXPAND: Generate new actions from selected node
        if not selected.is_solved and selected.depth < self.max_depth:
            logger.info(f"🌿 Expanding node at depth {selected.depth}")
            actions = await self._generate_actions(selected, task, self.tools)
            logger.info(f"🌿 Generated {len(actions)} expansion actions: {actions}")
            print(f"🌿 Expanding with {len(actions)} new actions")
            
            # Add children and evaluate them
            for i, action in enumerate(actions):
                child = selected.add_child(action)
                logger.debug(f"➕ Added expansion child {i+1}: {action}")
                await self._execute_and_reflect(child, task, self.tools)
                
                # Check if this action found significant insights requiring deep analysis
                insight_info = await self._analyze_for_insights(child, task)
                if insight_info['found']:
                    # Don't stop - trigger deep follow-up analysis
                    logger.success(f"🔍 Insight detected: {insight_info['type']} - Initiating deep analysis")
                    print(f"🔍 Insight found: {insight_info['type']} - Digging deeper...")
                    
                    # Generate follow-up actions for deeper analysis
                    followup_actions = await self._generate_deep_followup_actions(child, insight_info, task)
                    for followup in followup_actions:
                        followup_child = child.add_child(followup)
                        await self._execute_and_reflect(followup_child, task, self.tools)
                        
                    # Mark as solved only after comprehensive analysis
                    if await self._is_investigation_complete(root, task):
                        child.is_solved = True
                        root.is_solved = True
                        logger.success(f"🎉 Comprehensive analysis complete!")
                        print(f"🎉 Deep analysis complete!")
                        break
        else:
            logger.warning(f"⚠️ Node expansion skipped: solved={selected.is_solved}, depth={selected.depth}/{self.max_depth}")
        
        return state
    
    def _select_node(self, root: Node) -> Node:
        """Select most promising node using UCT"""
        current = root
        
        # Traverse down the tree using UCT until we find a leaf
        while current.children:
            current = current.best_child()
            if current is None:
                break
        
        return current
    
    async def _generate_actions(self, node: Node, task: str, tools: List[Any]) -> List[str]:
        """Generate possible actions from a node"""
        # Create context from node path
        path = []
        current = node
        while current and current.action:
            path.insert(0, f"Action: {current.action}")
            if current.observation:
                path.insert(1, f"Result: {current.observation[:200]}...")
            current = current.parent
        
        context = "\n".join(path) if path else "Starting investigation"
        
        # Generate actions using LLM
        prompt = f"""You are investigating: {task}

Current investigation path:
{context}

Available filesystem tools:
- list_directory(directory_path): List contents of a directory
- search_files(pattern, directory, file_pattern): Search for text patterns in files  
- read_file(file_path, start_line, end_line): Read specific lines from a file
- analyze_structure(file_path): Analyze code structure and functions
- find_dependencies(file_path): Find dependencies in a file

Generate 3-5 specific function calls to continue the investigation. Examples:
- "list_directory('.')" 
- "search_files('password', '.', '*.py')"
- "read_file('config/settings.py', 1, 50)"
- "analyze_structure('src/main.py')"

Focus on security patterns like: auth, login, password, admin, eval, exec, sql, injection

Format as JSON array of function calls: ["function1(args)", "function2(args)", ...]"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f"      OLLAMA OUTPUT: {response.content[:200]}{'...' if len(response.content) > 200 else ''}")
            
            # Parse actions with improved handling of LLM responses
            content = response.content.strip()
            
            # First try JSON parsing
            try:
                actions = json.loads(content)
                if isinstance(actions, list):
                    print(f"      ✅ Successfully parsed {len(actions)} actions as JSON")
                    return actions[:5]  # Limit to 5 actions
            except json.JSONDecodeError:
                # JSON failed, try multiple extraction patterns
                import re
                print(f"      🔧 JSON parsing failed, trying alternative extraction...")
                
                # Try different patterns for action extraction
                patterns = [
                    r'"([^"]*\([^)]*\))"',           # "function_name(args)"
                    r'(\w+\([^)]*\))',               # function_name(args)
                    r'"([^"]*)"',                    # "function_name"
                    r'(\w+)\s*$'                     # function_name at end of line
                ]
                
                actions = []
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        print(f"      ✅ Extracted {len(matches)} actions using pattern: {pattern}")
                        actions.extend(matches)
                        if len(actions) >= 5:  # Got enough actions
                            break
                
                # Filter out duplicates and non-function-like strings
                filtered_actions = []
                for action in actions[:5]:
                    if action and ('(' in action or action.replace('_', '').replace('file', '').isalpha()):
                        filtered_actions.append(action)
                
                return filtered_actions if filtered_actions else []
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            print(f"      ❌ LLM error: {str(e)}")
            return []  # Return empty list on LLM failure
    
    async def _execute_and_reflect(self, node: Node, task: str, tools: List[Any]):
        """Execute action and reflect on results"""
        if not node.action:
            return
        
        # SIMULATE: Execute the action
        logger.info(f"⚡ Executing action: {node.action}")
        print(f"   ⚡ Executing: {node.action}")
        
        # Emit action execution start
        await self._emit_progress("action_executing", {
            "node_id": node.id,
            "action": node.action,
            "depth": node.depth
        })
        
        try:
            observation = await self._execute_action(node.action, tools)
            node.observation = observation
            
            logger.debug(f"📝 Action observation: {observation[:300]}...")
            if observation:
                print(f"   📝 Found: {observation[:100]}...")
                
            # Emit action completion
            await self._emit_progress("action_completed", {
                "node_id": node.id,
                "action": node.action,
                "observation_preview": observation[:200] if observation else None,
                "success": True
            })
        except Exception as e:
            logger.warning(f"⚠️ Action execution failed: {e}")
            print(f"   ⚠️ Execution error: {str(e)}")
            
            # Emit action error
            await self._emit_progress("action_completed", {
                "node_id": node.id,
                "action": node.action,
                "error": str(e),
                "success": False
            })
            
            # Detailed error observation for agent to learn from
            import traceback
            error_details = traceback.format_exc()
            node.observation = f"""Action '{node.action}' failed:
Error: {str(e)}
Type: {type(e).__name__}
This suggests the action format or arguments were incorrect. 
Try alternative approaches or different parameters.
Full traceback: {error_details[-500:]}"""
        
        # REFLECT: Evaluate the action's value
        try:
            score = await self._reflect_on_action(node, task)
            logger.info(f"📊 Action scored: {score:.1f}/10 for action: {node.action}")
            print(f"   📊 Score: {score:.1f}/10")
        except Exception as e:
            logger.warning(f"⚠️ Scoring failed: {e}")
            print(f"   ⚠️ Scoring error: {str(e)}")
            # Use Ollama to score the scoring failure itself
            score = await self._score_error_scenario(node, task, str(e))
        
        # BACKPROPAGATE: Update tree values
        try:
            node.backpropagate(score)
            logger.debug(f"⬆️ Backpropagated score {score:.1f} through tree")
        except Exception as e:
            logger.warning(f"⚠️ Backpropagation failed: {e}")
            print(f"   ⚠️ Backprop error: {str(e)}")
    
    async def _execute_action(self, action: str, tools: List[Any]) -> str:
        """Execute an action using available tools"""
        try:
            # Import filesystem functions for real execution
            from filesystem_tools import (
                read_file_with_lines, 
                list_directory_tree, 
                search_in_files,
                analyze_code_structure,
                find_dependencies
            )
            
            # Handle actions with or without parentheses 
            action_clean = action.strip().strip('"\'').rstrip(',')
            
            # Parse action to extract function and arguments
            if '(' in action_clean and ')' in action_clean:
                func_call = action_clean
                func_name = func_call.split('(')[0].strip()
                args_str = func_call[func_call.find('(')+1:func_call.rfind(')')]
            else:
                # Handle actions without parentheses (simple function names)
                func_name = action_clean
                args_str = ""
                
            # Debug what real Ollama is generating
            print(f"      OLLAMA OUTPUT: func_name='{func_name}', args_str='{args_str}'")
            
            # Handle different formats that real Ollama might generate
            if func_name == "list_directory" and (not args_str or args_str.strip() in [".", "'.'", '"."']):
                result = list_directory_tree(".", max_depth=2)
                # Apply context management
                managed_result = await self._manage_context_size(result)
                return managed_result
            
            elif func_name == "list_directory_tree" and (not args_str or args_str.strip() in [".", "'.'", '"."']):
                result = list_directory_tree(".", max_depth=2)  
                managed_result = await self._manage_context_size(result)
                return managed_result
            
            elif func_name == "search_files":
                # Handle with or without arguments
                if args_str and args_str.strip():
                    # Parse multiple arguments: pattern, directory, file_pattern
                    parts = [p.strip().strip('"\'') for p in args_str.split(',')]
                    pattern = parts[0] if len(parts) > 0 else 'password|login|auth'
                    directory = parts[1] if len(parts) > 1 else '.'
                    file_pattern = parts[2] if len(parts) > 2 else '*.py'
                    return await self._safe_file_operation(search_in_files, pattern, directory, file_pattern, case_sensitive=False)
                # Default search for common security patterns if no args
                return await self._safe_file_operation(search_in_files, 'password|login|auth|admin', '.', '*.py', case_sensitive=False)
                
            # Parse common argument patterns  
            elif func_name == "read_file":
                if args_str and args_str.strip():
                    # Parse arguments: file_path, start_line, end_line
                    parts = [p.strip().strip('"\'') for p in args_str.split(',')]
                    file_path = parts[0] if len(parts) > 0 else None
                    start_line = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                    end_line = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
                    
                    if file_path:
                        if end_line:
                            return await self._safe_file_operation(read_file_with_lines, file_path, start_line, end_line)
                        else:
                            return await self._safe_file_operation(read_file_with_lines, file_path, start_line)
                
                # Default to reading a common config file
                return await self._safe_file_operation(read_file_with_lines, "README.md")
                
            elif func_name == "analyze_structure":
                # Handle analyze_structure with or without arguments
                if args_str and args_str.strip():
                    file_path = args_str.strip().strip('"\'')
                    result = analyze_code_structure(file_path)
                    return result
                else:
                    # Default to analyzing current directory
                    result = analyze_code_structure('.')
                    return result
                    
            elif func_name == "find_dependencies":
                # Handle find_dependencies with or without arguments
                if args_str and args_str.strip():
                    file_path = args_str.strip().strip('"\'')
                    result = find_dependencies(file_path)
                    return result
                else:
                    # Default to finding dependencies in common files
                    for common_file in ['requirements.txt', 'package.json', 'Pipfile', 'setup.py']:
                        try:
                            result = find_dependencies(common_file)
                            return result
                        except:
                            continue
                    return "No dependency files found"
                
            elif func_name == "list_directory" and "directory_path=" in args_str:
                dir_path = args_str.split('directory_path=')[1].strip().strip('"\'')
                result = list_directory_tree(dir_path, max_depth=2)
                return result
                
            elif func_name == "list_directory_tree":
                if "directory=" in args_str:
                    dir_path = args_str.split('directory=')[1].strip().strip('"\'')
                else:
                    dir_path = args_str.strip().strip('"\'') if args_str else '.'
                result = list_directory_tree(dir_path, max_depth=2)
                return result
                
            elif func_name == "search_files" and "pattern=" in args_str:
                # Parse search arguments
                parts = [p.strip() for p in args_str.split(',')]
                pattern = None
                directory = '.'
                    
                for part in parts:
                    if part.startswith('pattern='):
                        pattern = part.split('pattern=')[1].strip().strip('"\'')
                    elif part.startswith('directory='):
                        directory = part.split('directory=')[1].strip().strip('"\'')
                
                if pattern:
                    result = search_in_files(pattern, directory, "*.py", case_sensitive=False)
                    return result
                
            elif func_name == "analyze_structure" and "file_path=" in args_str:
                file_path = args_str.split('file_path=')[1].strip().strip('"\'')
                result = analyze_code_structure(file_path)
                return result
                
            elif func_name == "find_dependencies" and "file_path=" in args_str:
                file_path = args_str.split('file_path=')[1].strip().strip('"\'')
                result = find_dependencies(file_path)
                return result
            
            # If we get here, the action wasn't recognized
            return f"Could not execute action: {action}"
        
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    async def _reflect_on_action(self, node: Node, task: str) -> float:
        """Reflect on action and assign quality score with repetition penalty"""
        if not node.observation:
            return 1.0
        
        # Check for repeated actions in the path to prevent local maxima
        repetition_penalty = self._calculate_repetition_penalty(node)
        
        # Context-aware scoring based on task type
        task_context = self._get_task_context(task)
        
        prompt = f"""Task: {task}

Action taken: {node.action}
Observation: {node.observation}

Rate this action's value for solving the task (1-10 scale):
- How relevant is the observation to {task_context}?
- Does it reveal important information about the codebase?
- Does it help progress toward the goal?
- Consider: Is this action exploring new ground or repeating previous work?

Score (1-10):"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f"      OLLAMA SCORING: {response.content.strip()}")
            
            # Extract score with better parsing for formats like "Score: 9/10" or "9.5"
            content = response.content.strip()
            
            # Try multiple score extraction patterns - order matters for precision
            patterns = [
                r'\*\*(\d+(?:\.\d+)?)\*\*',      # "**10**" or "**9.5**"
                r'Score:\s*(\d+(?:\.\d+)?)/10',  # "Score: 9/10"
                r'Score:\s*(\d+(?:\.\d+)?)',     # "Score: 9.5"
                r'(\d+(?:\.\d+)?)/10',           # "9/10"
                r'Rate.*?(\d+(?:\.\d+)?)',       # "I rate this 8.5"
                r'(\d+(?:\.\d+)?)(?=\s|$)',      # Number followed by space or end of line
            ]
            
            score = None
            for pattern in patterns:
                score_match = re.search(pattern, content)
                if score_match:
                    score = float(score_match.group(1))
                    print(f"      ✅ Extracted score {score} using pattern: {pattern}")
                    break
            
            if score is not None:
                # Apply repetition penalty and context-aware adjustments
                adjusted_score = self._apply_scoring_adjustments(
                    score, node, task, repetition_penalty, task_context
                )
                print(f"      📊 Adjusted score: {score} -> {adjusted_score} (rep penalty: {repetition_penalty})")
                return min(max(adjusted_score, 1.0), 10.0)  # Clamp to 1-10
            else:
                logger.warning(f"⚠️ Could not extract score from: {content[:100]}")
                print(f"      ⚠️ No score found in response, asking LLM to re-evaluate...")
                # Ask Ollama again with a clearer prompt for numeric scoring
                return await self._retry_scoring_with_clear_prompt(node, task)
                
        except Exception as e:
            logger.error(f"❌ Scoring LLM call failed: {e}")
            print(f"      ❌ Scoring error: {str(e)}")
            # Use Ollama to score this error scenario
            return await self._score_error_scenario(node, task, str(e))
    
    def _calculate_repetition_penalty(self, node: Node) -> float:
        """Calculate penalty for repeated actions to prevent local maxima"""
        if not node.action:
            return 0.0
        
        # Extract action type (e.g., "list_directory" from "list_directory('.')")
        action_type = node.action.split('(')[0] if '(' in node.action else node.action
        
        # Count how many times this action type appears in the path
        repetitions = 0
        current = node.parent
        while current:
            if current.action:
                current_type = current.action.split('(')[0] if '(' in current.action else current.action
                if current_type == action_type:
                    repetitions += 1
            current = current.parent
        
        # Also check siblings to prevent breadth-wise repetition
        if node.parent:
            for sibling in node.parent.children:
                if sibling.id != node.id and sibling.action:
                    sibling_type = sibling.action.split('(')[0] if '(' in sibling.action else sibling.action
                    if sibling_type == action_type:
                        repetitions += 0.5  # Half penalty for sibling repetition
        
        # Calculate penalty (increases with each repetition)
        penalty = repetitions * 2.0  # -2 points per repetition
        return min(penalty, 5.0)  # Cap at -5 to avoid over-penalization
    
    def _get_task_context(self, task: str) -> str:
        """Determine task context for context-aware scoring"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ['security', 'vulnerability', 'exploit', 'attack']):
            return "finding security vulnerabilities"
        elif any(word in task_lower for word in ['performance', 'optimize', 'bottleneck', 'slow']):
            return "identifying performance issues"
        elif any(word in task_lower for word in ['architecture', 'structure', 'design', 'pattern']):
            return "understanding system architecture"
        elif any(word in task_lower for word in ['bug', 'error', 'fix', 'issue', 'problem']):
            return "finding and fixing bugs"
        elif any(word in task_lower for word in ['list', 'show', 'display', 'find files']):
            return "exploring the file structure"
        else:
            return "completing the investigation"
    
    def _apply_scoring_adjustments(self, base_score: float, node: Node, task: str, 
                                   repetition_penalty: float, task_context: str) -> float:
        """Apply various scoring adjustments to prevent exploitation"""
        adjusted_score = base_score
        
        # Apply repetition penalty
        adjusted_score -= repetition_penalty
        
        # Context-specific adjustments
        if node.action and 'list_directory' in node.action:
            # Directory listing gets capped based on context
            if "security" in task_context or "bug" in task_context:
                adjusted_score = min(adjusted_score, 6.0)  # Cap at 6 for non-exploration tasks
            elif "exploring" not in task_context:
                adjusted_score = min(adjusted_score, 7.0)  # Cap at 7 for other tasks
        
        # Bonus for deep exploration (encourage going deeper)
        depth_bonus = min(node.depth * 0.5, 2.0)  # Up to +2 for deep exploration
        adjusted_score += depth_bonus
        
        # Diversity bonus - reward trying different action types
        diversity_bonus = self._calculate_diversity_bonus(node)
        adjusted_score += diversity_bonus
        
        return adjusted_score
    
    def _calculate_diversity_bonus(self, node: Node) -> float:
        """Calculate bonus for action diversity"""
        if not node.parent:
            return 0.0
        
        # Get all action types used by siblings
        action_types = set()
        for child in node.parent.children:
            if child.action:
                action_type = child.action.split('(')[0] if '(' in child.action else child.action
                action_types.add(action_type)
        
        # Bonus increases with diversity
        diversity_bonus = len(action_types) * 0.3  # +0.3 per unique action type
        return min(diversity_bonus, 2.0)  # Cap at +2
    
    async def _score_error_scenario(self, node: Node, task: str, error_msg: str) -> float:
        """Use Ollama to score error scenarios - NO hardcoded values"""
        prompt = f"""Task: {task}

Action attempted: {node.action}
Error encountered: {error_msg}
Error details: {node.observation if node.observation else 'No observation available'}

This action failed with an error. Please score how valuable this error information is for the investigation (1-10 scale):

- Does this error reveal important information about the system?
- Does it help us understand what approaches to avoid?
- Does it guide us toward better strategies?
- Is this a useful learning experience?

Even failed actions can be valuable if they provide insight. Score this error scenario:

Score (1-10):"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f"      OLLAMA ERROR SCORING: {response.content.strip()}")
            
            # Extract score
            score_match = re.search(r'(\d+(?:\.\d+)?)', response.content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 1.0), 10.0)
            
            # If we can't extract a score, try one more time with a simpler prompt
            simple_prompt = f"Rate this error from 1-10: {error_msg[:100]}... Score:"
            response2 = await self.llm.ainvoke([HumanMessage(content=simple_prompt)])
            score_match2 = re.search(r'(\d+(?:\.\d+)?)', response2.content)
            if score_match2:
                return min(max(float(score_match2.group(1)), 1.0), 10.0)
            
        except Exception as nested_e:
            logger.error(f"❌ Error scoring failed completely: {nested_e}")
            print(f"      ❌ Could not score error scenario")
        
        # Last resort - but still no hardcoded score
        # Try one final ultra-simple LLM call
        try:
            final_prompt = "Error occurred. Rate 1-10:"
            response3 = await self.llm.ainvoke([HumanMessage(content=final_prompt)])
            score_match3 = re.search(r'(\d)', response3.content)
            if score_match3:
                return float(score_match3.group(1))
        except:
            pass
            
        # Truly last resort - if all LLM attempts fail, log error but continue with minimal score
        logger.error(f"💥 CRITICAL: All LLM scoring attempts failed for action: {node.action}")
        print(f"      💥 CRITICAL SCORING FAILURE - Node: {node.action}, Error: {error_msg}")
        print(f"      🔄 Continuing investigation with minimal score to avoid complete failure")
        
        # Return minimal score but continue investigation
        return 1.0

    async def _retry_scoring_with_clear_prompt(self, node: Node, task: str) -> float:
        """Retry scoring with an ultra-clear prompt for numeric output"""
        prompt = f"""TASK: {task}
ACTION: {node.action}  
RESULT: {node.observation[:200]}...

Rate this action's value from 1 to 10. Respond with ONLY a number.

Score:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f"      OLLAMA RETRY SCORING: {response.content.strip()}")
            
            # Try multiple extraction patterns
            for pattern in [r'(\d+\.?\d*)', r'(\d+)', r'Score:?\s*(\d+)', r'(\d+)/10']:
                score_match = re.search(pattern, response.content)
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score, 1.0), 10.0)
            
            # If still no score found, use error scoring method
            return await self._score_error_scenario(node, task, "Could not parse scoring response")
            
        except Exception as e:
            return await self._score_error_scenario(node, task, f"Retry scoring failed: {str(e)}")
    
    async def _analyze_for_insights(self, node: Node, task: str) -> Dict[str, Any]:
        """Analyze node for significant insights based on the analysis task"""
        if not node.observation:
            return {'found': False, 'type': None, 'details': None}
        
        observation = node.observation.lower()
        task_lower = task.lower()
        
        # Ignore directory listings and basic file operations
        if ("├──" in observation or "└──" in observation or 
            observation.startswith("no matches found") or
            observation.startswith("error:") or
            len(observation.strip()) < 50):
            return {'found': False, 'type': None, 'details': None}
        
        insight_info = {'found': False, 'type': None, 'details': None, 'files': [], 'lines': []}
        
        # Determine analysis focus based on task
        if any(keyword in task_lower for keyword in ['security', 'vulnerability', 'auth', 'sql']):
            return await self._analyze_security_patterns(observation, node)
        elif any(keyword in task_lower for keyword in ['performance', 'bottleneck', 'speed', 'optimization']):
            return await self._analyze_performance_patterns(observation, node)
        elif any(keyword in task_lower for keyword in ['architecture', 'design', 'structure', 'pattern']):
            return await self._analyze_architecture_patterns(observation, node)
        elif any(keyword in task_lower for keyword in ['data', 'flow', 'model', 'database']):
            return await self._analyze_data_patterns(observation, node)
        else:
            # General code analysis - look for any significant patterns
            return await self._analyze_general_patterns(observation, node, task)
    
    async def _analyze_security_patterns(self, observation: str, node: Node) -> Dict[str, Any]:
        """Analyze for security-related patterns"""
        # SQL injection detection
        if (("cursor.execute(" in observation and "%" in observation and "params" not in observation) or
            ("query = f\"" in observation) or ("query = '" in observation) or
            re.search("sql.*injection", observation) or ("unparameterized" in observation)):
            return {
                'found': True,
                'type': 'SQL Injection Risk',
                'details': 'Potential SQL injection vulnerability detected',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        # Authentication issues
        elif (re.search("password.*==", observation) and "plain" in observation) or \
           re.search("plain.*text.*password", observation):
            return {
                'found': True,
                'type': 'Authentication Issue', 
                'details': 'Insecure password handling detected',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        return {'found': False}
    
    async def _analyze_performance_patterns(self, observation: str, node: Node) -> Dict[str, Any]:
        """Analyze for performance-related patterns"""
        # Database query patterns
        if any(pattern in observation for pattern in ['n+1', 'loop.*query', 'select.*in.*loop']):
            return {
                'found': True,
                'type': 'Performance Bottleneck',
                'details': 'Potential N+1 query problem detected',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        # Memory usage patterns
        elif any(pattern in observation for pattern in ['memory.*leak', 'large.*object', 'cache.*unbounded']):
            return {
                'found': True,
                'type': 'Memory Issue',
                'details': 'Potential memory usage problem detected',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        return {'found': False}
    
    async def _analyze_architecture_patterns(self, observation: str, node: Node) -> Dict[str, Any]:
        """Analyze for architectural patterns"""
        # Design patterns
        if any(pattern in observation for pattern in ['class.*singleton', 'factory.*pattern', 'observer.*pattern']):
            return {
                'found': True,
                'type': 'Design Pattern',
                'details': 'Design pattern implementation found',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        # Architectural concerns
        elif any(pattern in observation for pattern in ['tight.*coupling', 'circular.*import', 'god.*class']):
            return {
                'found': True,
                'type': 'Architecture Concern',
                'details': 'Architectural issue detected',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        return {'found': False}
    
    async def _analyze_data_patterns(self, observation: str, node: Node) -> Dict[str, Any]:
        """Analyze for data-related patterns"""
        # Database patterns
        if any(pattern in observation for pattern in ['model.*relationship', 'foreign.*key', 'migration']):
            return {
                'found': True,
                'type': 'Data Model',
                'details': 'Database model or relationship found',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        # Data flow patterns
        elif any(pattern in observation for pattern in ['serializer', 'validator', 'transform']):
            return {
                'found': True,
                'type': 'Data Processing',
                'details': 'Data processing component found',
                'files': self._extract_files_from_observation(node.observation),
                'lines': self._extract_line_numbers_from_observation(node.observation)
            }
            
        return {'found': False}
    
    async def _analyze_general_patterns(self, observation: str, node: Node, task: str) -> Dict[str, Any]:
        """Analyze for general code patterns based on task keywords"""
        # Look for any significant code patterns
        significant_patterns = [
            ('class.*definition', 'Class Definition'),
            ('function.*def', 'Function Definition'),
            ('import.*statement', 'Import Statement'),
            ('api.*endpoint', 'API Endpoint'),
            ('test.*case', 'Test Case'),
            ('configuration', 'Configuration'),
            ('constant', 'Constants'),
            ('exception', 'Exception Handling')
        ]
        
        for pattern, pattern_type in significant_patterns:
            if re.search(pattern, observation):
                return {
                    'found': True,
                    'type': pattern_type,
                    'details': f'{pattern_type} found relevant to analysis task',
                    'files': self._extract_files_from_observation(node.observation),
                    'lines': self._extract_line_numbers_from_observation(node.observation)
                }
                
        return {'found': False}
    
    def _extract_files_from_observation(self, observation: str) -> List[str]:
        """Extract file paths from observation text"""
        files = []
        lines = observation.split('\n')
        for line in lines:
            # Look for file paths in format "/path/to/file.py:"
            if re.match(r'^/.*\.py:', line.strip()):
                file_path = line.strip().split(':')[0]
                if file_path not in files:
                    files.append(file_path)
        return files
    
    def _extract_line_numbers_from_observation(self, observation: str) -> List[int]:
        """Extract line numbers from observation text"""
        line_numbers = []
        lines = observation.split('\n')
        for line in lines:
            # Look for line numbers in format "  123:"
            match = re.match(r'\s*(\d+):', line)
            if match:
                line_num = int(match.group(1))
                if line_num not in line_numbers:
                    line_numbers.append(line_num)
        return sorted(line_numbers)
    
    async def _generate_deep_followup_actions(self, node: Node, insight_info: Dict[str, Any], task: str) -> List[str]:
        """Generate deep follow-up actions using LLM when insights are found"""
        
        prompt = f"""Based on the following insight discovered during codebase analysis, generate 5-10 specific follow-up actions to investigate further:

Task: {task}
Insight Found: {insight_info['type']} - {insight_info['details']}
Files Involved: {', '.join(insight_info['files']) if insight_info['files'] else 'Unknown'}
Line Numbers: {', '.join(map(str, insight_info['lines'])) if insight_info['lines'] else 'Unknown'}

Current Observation: {node.observation[:300]}...

Generate specific follow-up actions to deeply analyze this finding. Each action should be a function call like:
- read_file('path/to/file.py', start_line, num_lines)
- search_files('pattern', 'directory', '*.extension')
- analyze_structure('path/to/file.py')
- find_dependencies('path/to/file.py')
- list_directory('path')

Focus on:
1. Reading relevant files completely
2. Finding related code patterns
3. Understanding dependencies and connections
4. Analyzing code structure for deeper insights

Return actions as JSON list: ["action1", "action2", ...]

Actions:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            print(f"      LLM FOLLOWUP ACTIONS: {response.content[:200]}...")
            
            # Try to parse as JSON first
            try:
                actions = json.loads(response.content.strip())
                if isinstance(actions, list):
                    logger.info(f"🔍 LLM generated {len(actions)} follow-up actions for {insight_info['type']}")
                    return actions[:10]  # Limit for performance
            except json.JSONDecodeError:
                # Extract actions using regex as fallback
                import re
                actions = []
                patterns = [
                    r'"([^"]*\([^)]*\))"',           # "function_name(args)"
                    r'(\w+\([^)]*\))',               # function_name(args) 
                    r'"([^"]*)"',                    # "function_name"
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, response.content, re.MULTILINE)
                    if matches:
                        actions.extend(matches)
                        if len(actions) >= 8:
                            break
                
                if actions:
                    logger.info(f"🔍 LLM generated {len(actions)} follow-up actions via extraction")
                    return actions[:10]
                    
            # Fallback: Generate minimal actions based on files found
            fallback_actions = []
            for file_path in insight_info['files'][:3]:  # Limit files
                fallback_actions.append(f"read_file('{file_path}', 1, 200)")
                fallback_actions.append(f"analyze_structure('{file_path}')")
                
            logger.warning(f"⚠️ Using fallback actions for {insight_info['type']}")
            return fallback_actions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate follow-up actions: {e}")
            # Minimal fallback
            return [f"read_file('{insight_info['files'][0]}', 1, 100)"] if insight_info['files'] else []
    
    async def _is_investigation_complete(self, root: Node, task: str) -> bool:
        """Determine if investigation is comprehensive enough using LLM scoring"""
        # Collect all findings from the investigation tree
        findings = self._collect_all_findings(root)
        
        prompt = f"""Investigation Task: {task}

Findings Summary:
{chr(10).join([f"- {f['action']}: {f['observation'][:200]}..." for f in findings])}

Total Actions Taken: {len(findings)}
Files Analyzed: {len(set(f.get('files', []) for f in findings if f.get('files')))}

Is this investigation comprehensive enough to provide a complete security assessment?
Consider:
1. Have all major vulnerability types been checked (SQL injection, XSS, auth, code injection)?
2. Have vulnerable files been thoroughly analyzed including dependencies?
3. Are there enough details to provide specific remediation recommendations?
4. Have related code patterns been searched across the entire codebase?

Respond with: COMPLETE if investigation is comprehensive, CONTINUE if more analysis needed.
Include a brief reason.

Decision:"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            decision = response.content.strip().upper()
            
            if "COMPLETE" in decision:
                logger.info("🎯 LLM determined investigation is comprehensive")
                return True
            else:
                logger.info("🔄 LLM determined more investigation needed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to get investigation completeness decision: {e}")
            # Default to continue if scorer fails
            return False
    
    def _collect_all_findings(self, node: Node) -> List[Dict[str, Any]]:
        """Recursively collect all findings from investigation tree"""
        findings = []
        
        if node.action and node.observation:
            findings.append({
                'action': node.action,
                'observation': node.observation,
                'score': node.average_value
            })
            
        for child in node.children:
            findings.extend(self._collect_all_findings(child))
            
        return findings

    async def _has_found_vulnerabilities(self, node: Node) -> bool:
        """Check if a node's observation contains actual vulnerability findings"""
        if not node.observation:
            return False
        
        observation = node.observation.lower()
        
        # Ignore directory listings and basic file operations
        if ("├──" in observation or "└──" in observation or 
            observation.startswith("no matches found") or
            observation.startswith("error:") or
            len(observation.strip()) < 50):  # Too short to contain meaningful vulnerability info
            return False
        
        # Look for actual code vulnerabilities, not just file structure
        vuln_found = False
        
        # SQL injection - look for actual vulnerable code patterns
        if (("cursor.execute(" in observation and "%" in observation and "params" not in observation) or
            ("query = f\"" in observation) or ("query = '" in observation) or
            re.search("sql.*injection", observation) or ("unparameterized" in observation)):
            logger.info("🚨 SQL injection vulnerability detected")
            vuln_found = True
            
        # Authentication vulnerabilities - look for actual code issues  
        if (re.search("password.*==", observation) and "plain" in observation) or \
           re.search("plain.*text.*password", observation) or \
           re.search("no.*hash", observation) or \
           re.search("weak.*hash", observation) or \
           (re.search("password.*storage", observation) and "unsafe" in observation):
            logger.info("🚨 Authentication vulnerability detected")
            vuln_found = True
            
        # Code injection - look for dangerous eval/exec usage
        if re.search("eval.*input", observation) or \
           re.search("exec.*input", observation) or \
           re.search("eval.*user", observation) or \
           (re.search("subprocess.*shell=true", observation) and "input" in observation):
            logger.info("🚨 Code injection vulnerability detected") 
            vuln_found = True
            
        # Look for explicit vulnerability mentions in code comments or findings
        explicit_vulns = [
            "vulnerability", "security.*issue", "exploit", "attack.*vector",
            "injection.*flaw", "security.*hole", "unsafe.*code"
        ]
        
        for pattern in explicit_vulns:
            if re.search(pattern, observation):
                logger.info(f"🚨 Explicit vulnerability mention: {pattern}")
                vuln_found = True
                break
                
        return vuln_found
    
    def _count_nodes(self, node: Node) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    async def investigate(self, task: str, tools: List[Any], generate_report: bool = True, thread_id: str = None) -> Dict[str, Any]:
        """Run LATS investigation"""
        start_time = datetime.now()
        logger.info(f"🤖 Starting LATS investigation: {task}")
        print(f"🤖 Starting LATS investigation: {task}")
        
        # Emit investigation start event
        await self._emit_progress("investigation_started", {
            "task": task,
            "tools_available": len(tools) if tools else 0,
            "max_depth": self.max_depth
        })
        
        # Store tools on agent
        self.tools = tools
        
        # Create initial state with tool names only
        initial_state: TreeState = {
            "root": Node(),
            "input": task,
            "tool_names": [tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools] if tools else []
        }
        
        logger.info(f"🚀 Launching LangGraph execution")
        
        try:
            # Run the graph with checkpointing if available
            if self.checkpoint_db and thread_id:
                # Use AsyncSqliteSaver context manager
                async with AsyncSqliteSaver.from_conn_string(self.checkpoint_db) as checkpointer:
                    graph = self.graph.compile(checkpointer=checkpointer)
                    config = {"configurable": {"thread_id": thread_id}}
                    final_state = await graph.ainvoke(initial_state, config)
            else:
                # Compile without checkpointer
                graph = self.graph.compile()
                final_state = await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"❌ LangGraph execution failed: {e}")
            print(f"❌ Investigation failed: {e}")
            # Create a minimal final state for graceful degradation
            root = Node(action="error_recovery")
            root.children_ref = []  # Ensure children list is initialized
            root.observation = f"Investigation failed due to error: {str(e)}"
            root.backpropagate(1.0)  # Minimal score
            final_state = {
                "root": root,
                "input": task,
                "tool_names": [tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in tools] if tools else []
            }
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Extract results
        root = final_state["root"]
        best_node = self._find_best_node(root)
        
        result = {
            "task": task,
            "session_id": self.session_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "completed": root.is_solved,
            "nodes_explored": self._count_nodes(root),
            "max_depth": root.height,
            "best_score": best_node.average_value if best_node else 0.0,
            "best_action": best_node.action if best_node else None,
            "best_observation": best_node.observation if best_node else None,
            "tree": self._serialize_tree(root)
        }
        
        logger.success(f"✅ Investigation completed in {duration.total_seconds():.2f}s")
        logger.info(f"📊 Final stats: {result['nodes_explored']} nodes, depth {result['max_depth']}, score {result['best_score']:.1f}")
        
        # Emit investigation completion
        await self._emit_progress("investigation_completed", {
            "duration_seconds": result['duration_seconds'],
            "nodes_explored": result['nodes_explored'],
            "max_depth": result['max_depth'],
            "best_score": result['best_score'],
            "completed": result['completed']
        })
        
        # Generate report
        if generate_report:
            await self._emit_progress("report_generating", {"status": "started"})
            await self._generate_report(result, root)
            await self._emit_progress("report_generating", {"status": "completed"})
        
        return result
    
    def _find_best_node(self, root: Node) -> Optional[Node]:
        """Find node with highest average value"""
        best = root
        for child in self._traverse_tree(root):
            if child.average_value > best.average_value:
                best = child
        return best
    
    def _traverse_tree(self, node: Node):
        """Traverse all nodes in tree"""
        yield node
        for child in node.children:
            yield from self._traverse_tree(child)
    
    def _serialize_tree(self, node: Node, max_depth: int = 3) -> Dict[str, Any]:
        """Serialize tree for inspection"""
        if node.depth >= max_depth:
            return {"truncated": True}
        
        return {
            "id": node.id,
            "action": node.action,
            "observation": node.observation[:200] if node.observation else None,
            "value": node.average_value,
            "visits": node.visits,
            "children": [self._serialize_tree(child, max_depth) for child in node.children]
        }
    
    async def _generate_report(self, result: Dict[str, Any], root: Node):
        """Generate comprehensive investigation report with detailed analysis and remediation plans"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"lats_investigation_report_{timestamp}.md"
        
        logger.info(f"📝 Generating comprehensive investigation report: {report_file}")
        
        # Collect all findings with detailed analysis
        all_findings = self._collect_all_findings(root)
        vulnerability_findings = []
        file_analysis = {}
        dependency_chains = {}
        
        # Analyze all findings for vulnerabilities and file references
        for finding in all_findings:
            # Extract detailed vulnerability information
            vuln_info = await self._analyze_for_vulnerabilities_for_report(finding)
            if vuln_info['found']:
                vulnerability_findings.append({
                    **vuln_info,
                    'action': finding['action'],
                    'score': finding['score']
                })
            
            # Extract file references and group by file
            files = self._extract_files_from_observation(finding['observation'])
            lines = self._extract_line_numbers_from_observation(finding['observation'])
            
            for file_path in files:
                if file_path not in file_analysis:
                    file_analysis[file_path] = {'findings': [], 'lines': set(), 'issues': []}
                file_analysis[file_path]['findings'].append(finding)
                file_analysis[file_path]['lines'].update(lines)
                
        # Generate comprehensive report
        report_content = await self._generate_comprehensive_report_content(
            result, vulnerability_findings, file_analysis, dependency_chains, all_findings
        )
        
        # Save report
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.success(f"📄 Investigation report saved: {report_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    async def _analyze_for_vulnerabilities_for_report(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a finding for detailed vulnerability information for reporting"""
        observation = finding['observation']
        if not observation:
            return {'found': False}
            
        obs_lower = observation.lower()
        
        # More detailed vulnerability analysis for reporting
        vulnerability_patterns = {
            'SQL Injection': [
                ('cursor.execute(.*query.*)', 'Direct SQL execution without parameterization'),
                ('query.*f["\']', 'String formatting in SQL queries'), 
                ('SELECT.*%s', 'Potential SQL injection via string substitution'),
                ('unparameterized', 'Unparameterized query execution')
            ],
            'Authentication Vulnerability': [
                ('password.*==.*plain', 'Plain text password comparison'),
                ('no.*hash.*password', 'Missing password hashing'),
                ('weak.*hash', 'Weak password hashing algorithm'),
                ('session.*without.*validation', 'Session handling without proper validation')
            ],
            'Code Injection': [
                ('eval\\(.*input', 'Direct evaluation of user input'),
                ('exec\\(.*user', 'Execution of user-controlled code'),
                ('subprocess.*shell=true.*input', 'Shell injection vulnerability')
            ],
            'Input Validation': [
                ('no.*validation', 'Missing input validation'),
                ('unescaped.*input', 'Unescaped user input'),
                ('direct.*file.*access', 'Direct file access without validation')
            ]
        }
        
        for vuln_type, patterns in vulnerability_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, obs_lower):
                    return {
                        'found': True,
                        'type': vuln_type,
                        'description': description,
                        'pattern': pattern,
                        'files': self._extract_files_from_observation(observation),
                        'lines': self._extract_line_numbers_from_observation(observation)
                    }
        
        return {'found': False}
    
    async def _generate_comprehensive_report_content(self, result: Dict[str, Any], 
                                                   vulnerability_findings: List[Dict],
                                                   file_analysis: Dict[str, Dict],
                                                   dependency_chains: Dict[str, List],
                                                   all_findings: List[Dict]) -> str:
        """Generate comprehensive report content with detailed analysis and remediation"""
        
        # Generate executive summary using LLM
        executive_summary = await self._generate_executive_summary(result, vulnerability_findings, file_analysis)
        
        # Generate detailed remediation plans
        remediation_plans = await self._generate_remediation_plans(vulnerability_findings)
        
        report_content = f"""# 🔍 COMPREHENSIVE SECURITY INVESTIGATION REPORT

## 📋 Executive Summary
{executive_summary}

## 📊 Investigation Overview
- **Investigation Task**: {result['task']}
- **Session ID**: `{result['session_id']}`  
- **Duration**: {result['duration_seconds']:.2f} seconds
- **Status**: {'✅ COMPLETED' if result['completed'] else '⚠️ ONGOING'}
- **Total Actions Executed**: {result['nodes_explored']}
- **Investigation Depth**: {result['max_depth']} levels
- **Analysis Mode**: Deep investigation with unlimited exploration

## 🚨 Critical Security Findings

### Vulnerabilities Discovered: {len(vulnerability_findings)}

"""
        
        for i, vuln in enumerate(vulnerability_findings, 1):
            report_content += f"""
#### 🔴 Vulnerability #{i}: {vuln['type']}
- **Severity**: {'HIGH' if vuln['score'] > 8 else 'MEDIUM' if vuln['score'] > 6 else 'LOW'}
- **Confidence Score**: {vuln['score']:.1f}/10
- **Description**: {vuln['description']}
- **Detection Action**: `{vuln['action']}`
- **Affected Files**: {', '.join(f'`{f}`' for f in vuln['files']) if vuln['files'] else 'Not specified'}
- **Line Numbers**: {', '.join(map(str, vuln['lines'])) if vuln['lines'] else 'Not available'}

**Pattern Detected**: `{vuln.get('pattern', 'N/A')}`

"""
        
        # Detailed file analysis
        if file_analysis:
            report_content += f"""
## 📁 Detailed File Analysis

### Files Examined: {len(file_analysis)}

"""
            for file_path, analysis in file_analysis.items():
                lines_analyzed = sorted(list(analysis['lines'])) if analysis['lines'] else []
                report_content += f"""
#### 📄 File: `{file_path}`
- **Investigation Actions**: {len(analysis['findings'])}
- **Lines Analyzed**: {', '.join(map(str, lines_analyzed[:10]))}{'...' if len(lines_analyzed) > 10 else ''}
- **Security Issues Found**: {len([f for f in analysis['findings'] if f['score'] >= 7.0])}

**Key Findings**:
"""
                for finding in analysis['findings'][:3]:  # Top 3 findings per file
                    report_content += f"- Score {finding['score']:.1f}/10: `{finding['action']}`\n"
        
        # Add remediation plans
        report_content += f"""

## 🛠️ Remediation Action Plan

{remediation_plans}

## 📈 Investigation Statistics
- **Total Search Actions**: {len([f for f in all_findings if 'search' in f['action'].lower()])}
- **Files Read**: {len([f for f in all_findings if 'read_file' in f['action'].lower()])}  
- **Structure Analyses**: {len([f for f in all_findings if 'analyze' in f['action'].lower()])}
- **High-Value Discoveries**: {len([f for f in all_findings if f['score'] >= 8.0])}

## 🔧 Technical Details
- **Investigation Engine**: LATS (Language Agent Tree Search)
- **Search Strategy**: Monte Carlo Tree Search with UCT selection
- **Completion Criteria**: LLM-determined investigation completeness
- **No Artificial Limits**: Deep exploration until comprehensive analysis achieved
- **Vulnerability Detection**: Pattern-based + LLM semantic analysis

---
*Report generated by LATS Agent v2.0 - Deep Security Investigation System*
*Session: {result['session_id']} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report_content
    
    async def _generate_executive_summary(self, result: Dict, vulnerability_findings: List, file_analysis: Dict) -> str:
        """Generate executive summary using LLM analysis"""
        try:
            prompt = f"""Based on this security investigation using LATS (Language Agent Tree Search), write a concise executive summary (2-3 paragraphs):

LATS is a Language Agent Tree Search algorithm that uses Monte Carlo Tree Search to systematically explore codebases for security vulnerabilities.

Task: {result['task']}
Duration: {result['duration_seconds']:.1f} seconds
Actions Taken: {result['nodes_explored']}
Vulnerabilities Found: {len(vulnerability_findings)}
Files Analyzed: {len(file_analysis)}

Key Vulnerabilities:
{chr(10).join([f"- {v['type']}: {v['description']}" for v in vulnerability_findings[:5]])}

Write an executive summary that explains:
1. What was investigated and the scope
2. Key security findings and their severity
3. Overall security posture assessment

Keep it professional and concise."""
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Failed to generate executive summary: {e}")
            return f"Security investigation of '{result['task']}' completed in {result['duration_seconds']:.1f} seconds. {len(vulnerability_findings)} vulnerabilities identified across {len(file_analysis)} files analyzed."
    
    async def _generate_remediation_plans(self, vulnerability_findings: List[Dict]) -> str:
        """Generate specific remediation plans for each vulnerability type"""
        if not vulnerability_findings:
            return "No specific vulnerabilities identified. Recommend periodic security assessments."
            
        remediation_content = ""
        vuln_types = {}
        
        # Group vulnerabilities by type
        for vuln in vulnerability_findings:
            vuln_type = vuln['type']
            if vuln_type not in vuln_types:
                vuln_types[vuln_type] = []
            vuln_types[vuln_type].append(vuln)
        
        # Generate remediation plans for each type
        remediation_plans = {
            'SQL Injection': {
                'priority': 'CRITICAL',
                'actions': [
                    'Implement parameterized queries/prepared statements',
                    'Add input validation and sanitization',
                    'Use ORM frameworks where possible',
                    'Implement SQL query allowlists',
                    'Add database user permission restrictions'
                ],
                'timeline': 'Immediate (1-2 days)'
            },
            'Authentication Vulnerability': {
                'priority': 'HIGH', 
                'actions': [
                    'Implement secure password hashing (bcrypt, Argon2)',
                    'Add password strength requirements',
                    'Implement secure session management',
                    'Add multi-factor authentication',
                    'Regular security audits of auth mechanisms'
                ],
                'timeline': '1-2 weeks'
            },
            'Code Injection': {
                'priority': 'CRITICAL',
                'actions': [
                    'Remove eval()/exec() with user input',
                    'Implement input validation and sanitization',
                    'Use safe alternatives to dynamic code execution',
                    'Add code review processes',
                    'Implement runtime application self-protection (RASP)'
                ],
                'timeline': 'Immediate (1-3 days)'
            },
            'Input Validation': {
                'priority': 'MEDIUM',
                'actions': [
                    'Implement comprehensive input validation',
                    'Add output encoding/escaping',
                    'Use validation libraries/frameworks',
                    'Implement allowlist-based validation',
                    'Add logging for validation failures'
                ],
                'timeline': '1-2 weeks'
            }
        }
        
        for vuln_type, vulns in vuln_types.items():
            if vuln_type in remediation_plans:
                plan = remediation_plans[vuln_type]
                remediation_content += f"""
### 🚨 {vuln_type} - Priority: {plan['priority']}
**Instances Found**: {len(vulns)}
**Timeline**: {plan['timeline']}

**Affected Files**:
{chr(10).join([f"- `{'`, `'.join(v['files'])}` (Lines: {', '.join(map(str, v['lines']))})" for v in vulns if v['files']])}

**Remediation Actions**:
{chr(10).join([f"{i+1}. {action}" for i, action in enumerate(plan['actions'])])}

"""
        
        return remediation_content if remediation_content else "No specific remediation plans needed."
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.success(f"📄 Investigation report saved: {report_file}")
            print(f"📄 Report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    def _format_tree_for_report(self, node: Node, indent: int = 0, max_depth: int = 3) -> str:
        """Format tree structure for report"""
        if indent > max_depth:
            return ""
        
        result = ""
        prefix = "  " * indent
        
        if node.action:
            result += f"{prefix}- **{node.action}** (Score: {node.average_value:.1f}, Visits: {node.visits})\n"
            if node.observation and len(node.observation) > 50:
                preview = node.observation[:100].replace('\n', ' ')
                result += f"{prefix}  💡 {preview}...\n"
        else:
            result += f"{prefix}- **ROOT** (Visits: {node.visits})\n"
        
        for child in node.children:
            result += self._format_tree_for_report(child, indent + 1, max_depth)
        
        return result


# For backward compatibility
LATSAlgorithm = LATSAgent