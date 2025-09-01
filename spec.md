# LATS MCP Server - Technical Specification

## Executive Summary

The LATS (Language Agent Tree Search) MCP Server is a sophisticated code investigation agent that combines Monte Carlo Tree Search with LLM reasoning to perform thorough, parallel exploration of codebases. It provides intelligent insights, persistent memory, and pattern recognition capabilities through a Model Context Protocol interface.

## Architecture Overview

### Core Components

1. **LATS Algorithm Core** (`lats_core.py`)
   - Monte Carlo Tree Search implementation
   - UCT (Upper Confidence Bound) node selection
   - Parallel branch exploration
   - Reflection-based scoring system

2. **Filesystem Tools** (`filesystem_tools.py`)
   - File reading with line numbers
   - Directory structure analysis
   - Pattern-based code search
   - Dependency extraction
   - Code structure analysis (AST-based for Python)

3. **Memory Manager** (`memory_manager.py`)
   - langmem integration for persistent memory
   - Investigation history tracking
   - Pattern extraction and recognition
   - Insight storage and retrieval
   - Similarity search for past investigations

4. **MCP Server** (`lats_mcp_server.py`)
   - FastMCP-based server implementation
   - Tool exposure via MCP protocol
   - Async request handling
   - STDIO transport for Claude integration

## Technical Decisions & Rationale

### 1. Python Over Node.js
**Decision**: Use Python with FastMCP instead of Node.js

**Rationale**:
- **Ecosystem Alignment**: LangChain, LangGraph, and langmem are Python-native
- **Scientific Computing**: Better support for tree algorithms and numerical operations
- **AST Analysis**: Native Python AST module for code structure analysis
- **Ollama Integration**: Mature Python bindings via langchain-ollama
- **Simplicity**: Single-language implementation reduces complexity

### 2. LATS Algorithm Implementation
**Decision**: Custom LATS implementation with configurable parameters

**Key Features**:
- **UCT Selection**: Balances exploration vs exploitation with c=1.414
- **Parallel Expansion**: Explores up to 5 branches simultaneously
- **Reflection Scoring**: 0-10 scale with 7.0 threshold for solutions
- **Depth Limiting**: Maximum depth of 5 to prevent infinite exploration
- **Iteration Capping**: Maximum 10 iterations for bounded runtime

**Trade-offs**:
- More compute-intensive than single-shot approaches
- Better solution quality through systematic exploration
- Transparent reasoning process via tree structure

### 3. Memory Architecture with langmem
**Decision**: Use langmem with namespace-separated memories

**Memory Namespaces**:
- `investigations`: Complete investigation records
- `insights`: Extracted knowledge snippets
- `patterns`: Action sequences from successful investigations
- `errors`: Debug and error tracking

**Benefits**:
- Persistent learning across sessions
- Pattern-based suggestions for new tasks
- Similarity search for relevant past investigations
- Gradual improvement through experience

### 4. Ollama + gpt-oss Integration
**Decision**: Use Ollama with gpt-oss model for local reasoning

**Configuration**:
```python
ChatOllama(
    model="gpt-oss",
    reasoning=True,  # Separate reasoning capture
    temperature=0.7,
    num_ctx=8192
)
```

**Reasoning Handling**:
- Extract reasoning from `additional_kwargs['reasoning_content']`
- Fallback to `<think>` tag parsing
- Transparent chain-of-thought for debugging

### 5. Filesystem Tools Design
**Decision**: Comprehensive file analysis toolkit

**Tools Provided**:
- `read_file`: Line-numbered content with range support
- `list_directory`: Tree structure with language detection
- `search_files`: Regex search with context lines
- `analyze_structure`: AST-based code structure (Python)
- `find_dependencies`: Import/dependency extraction

**Design Principles**:
- Safety first (size limits, permission handling)
- Rich context (line numbers, surrounding lines)
- Language awareness (syntax-specific analysis)

### 6. MCP Protocol Implementation
**Decision**: FastMCP with STDIO transport

**Exposed Tools**:
- `investigate`: Full LATS investigation
- `get_status`: Current investigation status
- `search_memory`: Query past investigations
- `get_insights`: Retrieve relevant insights
- `store_insight`: Manual insight storage
- `analyze_file`: Quick single-file analysis
- `parallel_search`: Multi-pattern search

**Benefits**:
- Standard protocol for LLM integration
- Stateless request/response model
- Easy integration with Claude and other LLMs

## Algorithm Details

### LATS Monte Carlo Tree Search

#### Node Selection (UCT)
```python
UCT = value/visits + c * sqrt(2 * ln(parent_visits) / visits)
```
- Balances exploitation (high value) and exploration (low visits)
- c=1.414 provides optimal theoretical balance

#### Expansion Strategy
1. Generate N candidate actions (N=5 default)
2. Create child nodes for each action
3. Parallelize to reduce latency

#### Simulation & Reflection
1. Execute action using filesystem tools
2. Capture observation (limited to 2KB)
3. Generate reflection with score (0-10)
4. Mark terminal if score >= threshold

#### Backpropagation
1. Update visit counts up the tree
2. Accumulate rewards along path
3. Maintain running averages for decision making

### Memory Learning Process

#### Investigation Storage
1. Complete investigation with score >= 7.0
2. Extract solution path and file references
3. Store with timestamp and metadata
4. Index for similarity search

#### Pattern Extraction
1. Identify successful action sequences
2. Classify task type (debugging, implementation, etc.)
3. Store patterns with scores
4. Use for future suggestions

#### Insight Generation
1. Extract key findings from investigations
2. Tag with context and categories
3. Enable semantic search
4. Build knowledge base over time

## Performance Characteristics

### Time Complexity
- **Tree Search**: O(b^d) where b=branching factor, d=depth
- **With UCT**: Focuses on promising branches, practical O(n*d)
- **Memory Search**: O(log n) with indexing

### Space Complexity
- **Tree Storage**: O(b^d) nodes in worst case
- **Memory**: Grows linearly with investigations
- **Practical Limits**: ~1000 nodes per investigation

### Optimization Strategies
1. **Early Termination**: Stop when solution found
2. **Branch Pruning**: Limit to top 5 actions
3. **Observation Truncation**: Cap at 2KB per observation
4. **Async Execution**: Parallel tool execution
5. **Memory Indexing**: Fast similarity search

## Integration Guide

### With Claude (via MCP)
```json
{
  "mcpServers": {
    "lats": {
      "command": "python",
      "args": ["/path/to/lats_mcp_server.py"],
      "transport": "stdio"
    }
  }
}
```

### With Ollama
Ensure Ollama is running with gpt-oss model:
```bash
ollama pull gpt-oss
ollama serve
```

### Python Usage
```python
from lats_core import LATSAlgorithm, LATSConfig

config = LATSConfig(
    model_name="gpt-oss",
    max_depth=5,
    max_iterations=10
)
algorithm = LATSAlgorithm(config)
```

## Error Handling

### Graceful Degradation
1. Tool execution failures logged but don't crash
2. Memory errors stored in error namespace
3. Ollama connection issues handled with retries
4. File access errors return informative messages

### Debugging Support
- Reasoning transparency via thinking tags
- Tree visualization through solution paths
- Error namespace for troubleshooting
- Detailed logging of all operations

## Security Considerations

1. **File Access**: Restricted to readable files
2. **Size Limits**: 1MB file read limit
3. **Depth Limits**: Prevents infinite recursion
4. **No Execution**: Only reads, never executes code
5. **Path Validation**: Resolves to absolute paths

## Future Enhancements

### Planned Improvements
1. **Multi-Model Support**: Beyond gpt-oss
2. **Distributed Search**: Parallel tree exploration
3. **Custom Embeddings**: Local embedding models
4. **Tool Extensions**: Git integration, test running
5. **Visualization**: Interactive tree exploration UI

### Experimental Features
1. **Reinforcement Learning**: Improve UCT parameters
2. **Meta-Learning**: Task-specific strategy selection
3. **Collaborative Memory**: Shared knowledge base
4. **Adaptive Depth**: Dynamic depth based on complexity

## Conclusion

The LATS MCP Server represents a sophisticated approach to code investigation, combining:
- **Systematic Exploration**: Monte Carlo Tree Search
- **Transparent Reasoning**: Ollama with thinking tags
- **Persistent Learning**: langmem memory management
- **Practical Integration**: MCP protocol for LLM usage

This architecture provides a robust foundation for intelligent code analysis that improves over time through experience and pattern recognition.