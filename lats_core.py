"""
LATS (Language Agent Tree Search) Core Implementation
Implements Monte Carlo Tree Search for LLM-based code investigation
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


@dataclass
class TreeNode:
    """Node in the LATS search tree"""
    id: str = field(default_factory=lambda: str(uuid4()))
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    action: Optional[str] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None
    reasoning: Optional[str] = None
    value: float = 0.0
    visits: int = 0
    depth: int = 0
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def uct_score(self, c: float = 1.414) -> float:
        """Calculate Upper Confidence Bound for tree search"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration = c * math.sqrt(2 * math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, c: float = 1.414) -> Optional['TreeNode']:
        """Select best child based on UCT score"""
        if not self.children:
            return None
        return max(self.children, key=lambda n: n.uct_score(c))
    
    def add_child(self, action: str) -> 'TreeNode':
        """Add a child node with given action"""
        child = TreeNode(
            parent=self,
            action=action,
            depth=self.depth + 1
        )
        self.children.append(child)
        return child
    
    def backpropagate(self, reward: float):
        """Backpropagate reward up the tree"""
        node = self
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_path_to_root(self) -> List['TreeNode']:
        """Get path from current node to root"""
        path = []
        node = self
        while node:
            path.insert(0, node)
            node = node.parent
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            'id': self.id,
            'action': self.action,
            'observation': self.observation[:500] if self.observation else None,
            'reflection': self.reflection,
            'value': self.value,
            'visits': self.visits,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'num_children': len(self.children)
        }


class LATSState(TypedDict):
    """State for LATS agent"""
    messages: List[BaseMessage]
    root_node: TreeNode
    current_node: TreeNode
    max_depth: int
    max_iterations: int
    num_expansions: int
    task: str
    iteration: int
    best_solution: Optional[Dict[str, Any]]
    explored_branches: List[List[str]]


class LATSConfig(BaseModel):
    """Configuration for LATS agent"""
    model_name: str = Field(default="gpt-oss", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    temperature: float = Field(default=0.7, description="Model temperature")
    max_depth: int = Field(default=5, description="Maximum tree depth")
    max_iterations: int = Field(default=10, description="Maximum iterations")
    num_expand: int = Field(default=5, description="Number of actions to expand")
    c_param: float = Field(default=1.414, description="UCT exploration parameter")
    min_score_threshold: float = Field(default=7.0, description="Minimum score for solution")
    enable_reasoning: bool = Field(default=True, description="Enable reasoning mode")


class LATSAlgorithm:
    """Core LATS algorithm implementation"""
    
    def __init__(self, config: LATSConfig):
        self.config = config
        self.llm = self._init_llm()
    
    def _init_llm(self):
        """Initialize real Ollama LLM with reasoning support"""
        return ChatOllama(
            model=self.config.model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=8192
        )
    
    def parse_reasoning_response(self, response: AIMessage) -> Tuple[str, str]:
        """Extract reasoning and content from response"""
        content = response.content
        reasoning = ""
        
        # Check for reasoning in additional_kwargs
        if hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
            reasoning = response.additional_kwargs['reasoning_content']
        # Fallback to parsing think tags
        elif '<think>' in content:
            think_pattern = r'<think>(.*?)</think>'
            reasoning_match = re.search(think_pattern, content, re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
        
        return reasoning, content
    
    async def select_node(self, root: TreeNode) -> TreeNode:
        """Select best leaf node using UCT"""
        current = root
        
        # Navigate to best leaf node
        while current.children and not current.is_terminal:
            current = current.best_child(self.config.c_param)
        
        return current
    
    async def expand_node(self, node: TreeNode, task: str, tools: List[Any]) -> List[TreeNode]:
        """Expand node with possible actions"""
        if node.is_terminal or node.depth >= self.config.max_depth:
            return []
        
        # Generate actions based on context
        prompt = self._create_expansion_prompt(node, task, tools)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        reasoning, content = self.parse_reasoning_response(response)
        
        # Parse actions from response
        actions = self._parse_actions(content)
        
        # Create child nodes
        children = []
        for action in actions[:self.config.num_expand]:
            child = node.add_child(action)
            child.reasoning = reasoning
            children.append(child)
        
        return children
    
    async def simulate_node(self, node: TreeNode, executor: Any) -> str:
        """Simulate action execution"""
        if not node.action:
            return ""
        
        try:
            result = await executor(node.action)
            node.observation = str(result)[:2000]  # Limit observation size
            return node.observation
        except Exception as e:
            node.observation = f"Error: {str(e)}"
            return node.observation
    
    async def reflect_on_node(self, node: TreeNode, task: str) -> float:
        """Generate reflection and score for node"""
        if not node.observation:
            return 0.0
        
        prompt = self._create_reflection_prompt(node, task)
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        reasoning, reflection = self.parse_reasoning_response(response)
        
        node.reflection = reflection
        score = self._extract_score(reflection)
        node.value = score
        
        # Check if solution found
        if score >= self.config.min_score_threshold:
            node.is_terminal = True
        
        return score
    
    def backpropagate(self, node: TreeNode, reward: float):
        """Backpropagate reward up the tree"""
        node.backpropagate(reward)
    
    def _create_expansion_prompt(self, node: TreeNode, task: str, tools: List[Any]) -> str:
        """Create prompt for action expansion"""
        # Build context from path
        path = node.get_path_to_root()
        context = []
        for n in path[1:]:  # Skip root
            if n.action:
                context.append(f"Action: {n.action}")
            if n.observation:
                context.append(f"Result: {n.observation[:200]}")
        
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
        
        return f"""Task: {task}

Available tools:
{tool_descriptions}

Previous exploration:
{chr(10).join(context) if context else "Starting fresh investigation"}

Generate {self.config.num_expand} diverse actions to investigate the codebase.
Each action should explore a different aspect or approach.

Format each action as:
ACTION: tool_name(arg1="value1", arg2="value2")

Actions:"""
    
    def _create_reflection_prompt(self, node: TreeNode, task: str) -> str:
        """Create prompt for reflection"""
        return f"""Task: {task}

Action taken: {node.action}
Result: {node.observation}

Evaluate how well this action contributes to completing the task.
Consider:
1. Relevance to the task
2. Quality of information gathered
3. Progress toward solution

Provide a score from 0-10 where:
- 0-3: Not helpful or misleading
- 4-6: Somewhat helpful but incomplete
- 7-8: Very helpful, significant progress
- 9-10: Task completed or critical insight found

Response format:
Analysis: [Your analysis]
Score: X/10"""
    
    def _parse_actions(self, text: str) -> List[str]:
        """Parse actions from LLM response"""
        actions = []
        lines = text.split('\n')
        
        for line in lines:
            if line.strip().startswith('ACTION:'):
                action = line.replace('ACTION:', '').strip()
                actions.append(action)
        
        return actions
    
    def _extract_score(self, text: str) -> float:
        """Extract numerical score from reflection"""
        # Look for "Score: X/10" pattern
        match = re.search(r'Score:\s*(\d+(?:\.\d+)?)/10', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # Look for just a number between 0-10
        match = re.search(r'\b([0-9]|10)(?:\.\d+)?\b', text)
        if match:
            score = float(match.group(0))
            if 0 <= score <= 10:
                return score
        
        return 5.0  # Default middle score
    
    def get_best_solution_path(self, root: TreeNode) -> List[Dict[str, Any]]:
        """Get the best solution path from the tree"""
        best_terminal = None
        best_score = 0
        
        def find_best_terminal(node: TreeNode):
            nonlocal best_terminal, best_score
            if node.is_terminal and node.value > best_score:
                best_terminal = node
                best_score = node.value
            for child in node.children:
                find_best_terminal(child)
        
        find_best_terminal(root)
        
        if best_terminal:
            path = best_terminal.get_path_to_root()
            return [n.to_dict() for n in path[1:]]  # Skip root
        
        # If no terminal found, return path to best valued node
        best_node = root
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node.visits > 0 and node.value / node.visits > best_node.value / max(best_node.visits, 1):
                best_node = node
            queue.extend(node.children)
        
        path = best_node.get_path_to_root()
        return [n.to_dict() for n in path[1:]]  # Skip root
    
    def get_explored_branches(self, root: TreeNode) -> List[List[str]]:
        """Get all explored branches in the tree"""
        branches = []
        
        def collect_branch(node: TreeNode, current_branch: List[str]):
            if node.action:
                current_branch = current_branch + [node.action]
            
            if not node.children:
                if current_branch:
                    branches.append(current_branch)
            else:
                for child in node.children:
                    collect_branch(child, current_branch)
        
        collect_branch(root, [])
        return branches
    
    def extract_insights(self, root: TreeNode, task: str) -> Dict[str, Any]:
        """Extract insights from the search tree"""
        solution_path = self.get_best_solution_path(root)
        explored_branches = self.get_explored_branches(root)
        
        # Find file references
        file_refs = set()
        def extract_refs(node: TreeNode):
            if node.observation:
                # Extract file:line patterns
                refs = re.findall(r'([^\s:]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb)):(\d+)', node.observation)
                for ref in refs:
                    file_refs.add(f"{ref[0]}:{ref[2]}")
            for child in node.children:
                extract_refs(child)
        
        extract_refs(root)
        
        # Calculate statistics
        total_nodes = 0
        max_score = 0
        def count_nodes(node: TreeNode):
            nonlocal total_nodes, max_score
            total_nodes += 1
            if node.value > max_score:
                max_score = node.value
            for child in node.children:
                count_nodes(child)
        
        count_nodes(root)
        
        return {
            'task': task,
            'solution_path': solution_path,
            'explored_branches': explored_branches[:10],  # Limit to 10 branches
            'file_references': sorted(list(file_refs)),
            'statistics': {
                'total_nodes': total_nodes,
                'max_depth': max(n['depth'] for n in solution_path) if solution_path else 0,
                'best_score': max_score,
                'num_branches': len(explored_branches)
            },
            'is_complete': any(n['is_terminal'] for n in solution_path) if solution_path else False
        }