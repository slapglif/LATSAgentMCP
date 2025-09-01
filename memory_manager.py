"""
Memory Manager using langmem with Ollama
Provides persistent memory for LATS investigations using langmem's full tooling
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings  
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager, create_manage_memory_tool, create_search_memory_tool


class InvestigationMemory(BaseModel):
    """Memory entry for an investigation"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    task: str
    timestamp: datetime = Field(default_factory=datetime.now)
    solution_path: List[Dict[str, Any]]
    file_references: List[str]
    insights: Dict[str, Any]
    score: float
    is_complete: bool
    tags: List[str] = Field(default_factory=list)


class MemoryManager:
    """Manages long-term memory for LATS investigations using langmem"""
    
    def __init__(self, db_path: str = "lats_memory.db"):
        """Initialize memory manager with langmem and Ollama"""
        self.db_path = db_path
        
        # Initialize Ollama embeddings with Arctic Embed 2
        self.embeddings = OllamaEmbeddings(
            model="snowflake-arctic-embed2",
            base_url="http://localhost:11434"
        )
        
        # Create persistent SQLite store for langmem with Ollama embeddings
        self.store_cm = SqliteStore.from_conn_string(
            db_path,
            index={
                "dims": 1024,  # Arctic Embed 2 dimension  
                "embed": lambda texts: self.embeddings.embed_documents(texts) if isinstance(texts, list) else [self.embeddings.embed_query(texts)]
            }
        )
        self.store = self.store_cm.__enter__()
        
        # Initialize Ollama LLM for memory management
        self.llm = ChatOllama(
            model="gpt-oss",
            base_url="http://localhost:11434"
        )
        
        # Create langmem memory manager
        self.memory_manager = create_memory_store_manager(
            self.llm,
            namespace=("investigations",),
            store=self.store,
            enable_inserts=True,
            enable_deletes=False,
            query_limit=10
        )
        
        # Create langmem tools
        self.manage_tool = create_manage_memory_tool(
            namespace=("investigations",),
            store=self.store
        )
        
        self.search_tool = create_search_memory_tool(
            namespace=("investigations",),
            store=self.store
        )
    
    def store_investigation(self, investigation: InvestigationMemory) -> str:
        """Store investigation memory using langmem"""
        # Format for langmem
        memory_content = {
            'id': investigation.id,
            'task': investigation.task,
            'timestamp': investigation.timestamp.isoformat(),
            'solution_path': investigation.solution_path,
            'file_references': investigation.file_references,
            'insights': investigation.insights,
            'score': investigation.score,
            'is_complete': investigation.is_complete,
            'tags': investigation.tags
        }
        
        # Store using langmem
        self.store.put(
            ("investigations",),
            investigation.id,
            memory_content
        )
        
        return investigation.id
    
    def search_similar_investigations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar investigations using langmem"""
        # Use langmem's search
        results = self.store.search(
            ("investigations",),
            query=query,
            limit=limit
        )
        
        # Format results
        investigations = []
        for item in results:
            if item.value:
                inv = item.value.copy()
                inv['similarity'] = item.score if item.score else 0.0
                investigations.append(inv)
        
        return investigations
    
    def get_insights(self) -> List[str]:
        """Get key insights from investigations"""
        # Search all items (no query returns all)
        items = self.store.search(
            ("investigations",),
            limit=100
        )
        
        insights = []
        for item in items:
            if item.value and isinstance(item.value, dict):
                if 'insights' in item.value and item.value.get('is_complete'):
                    for key, value in item.value['insights'].items():
                        insights.append(f"{key}: {value}")
        
        return insights
    
    def get_pattern_suggestions(self, task: str) -> List[Dict[str, Any]]:
        """Get pattern suggestions for a task using langmem search"""
        similar = self.search_similar_investigations(task, limit=3)
        
        suggestions = []
        for inv in similar:
            if inv.get('is_complete') and inv.get('score', 0) >= 7.0:
                suggestions.append({
                    'task': inv.get('task'),
                    'score': inv.get('score'),
                    'insights': inv.get('insights', {})
                })
        
        return suggestions
    
    def store_error(self, action: str, error: str, context: Dict[str, Any]):
        """Store error for analysis using langmem"""
        error_record = {
            'id': str(uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'error': error,
            'context': context
        }
        
        self.store.put(
            ("errors",),
            error_record['id'],
            error_record
        )
    
    def get_error_patterns(self) -> List[Dict[str, Any]]:
        """Analyze error patterns from langmem"""
        items = self.store.search(
            ("errors",),
            limit=100
        )
        
        # Count occurrences
        action_counts = {}
        for item in items:
            if item.value and 'action' in item.value:
                action = item.value['action']
                action_counts[action] = action_counts.get(action, 0) + 1
        
        # Sort by frequency
        patterns = [
            {'action': action, 'frequency': count}
            for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        ][:10]
        
        return patterns
    
    def __del__(self):
        """Cleanup context manager"""
        if hasattr(self, 'store_cm'):
            try:
                self.store_cm.__exit__(None, None, None)
            except:
                pass