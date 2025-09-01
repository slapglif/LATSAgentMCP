# LATS MCP Server

A sophisticated code investigation agent that uses Language Agent Tree Search (LATS) with Monte Carlo Tree Search to systematically explore codebases and provide intelligent insights.

## Features

- 🌳 **Monte Carlo Tree Search**: Systematic parallel exploration of solution space
- 🧠 **Reasoning Transparency**: Full chain-of-thought with gpt-oss model
- 💾 **Persistent Memory**: Learn from past investigations using langmem
- 🔍 **Smart Code Analysis**: AST-based structure analysis and dependency extraction
- 🚀 **MCP Integration**: Easy integration with Claude and other LLMs
- 📊 **Pattern Recognition**: Learns successful investigation patterns over time

## Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** with gpt-oss model:
```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the gpt-oss model
ollama pull gpt-oss

# Start Ollama server
ollama serve
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd lats

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Make the server executable
chmod +x lats_mcp_server.py

# Run the MCP server
python lats_mcp_server.py
```

## Integration with Claude

Add to your Claude MCP configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "lats": {
      "command": "python",
      "args": ["/absolute/path/to/lats_mcp_server.py"],
      "transport": "stdio"
    }
  }
}
```

## Usage Examples

### Basic Investigation

```python
# Via MCP in Claude
"Investigate where error handling is implemented in the authentication module"

# Response includes:
# - Solution path with scored steps
# - File references with line numbers
# - Explored branches
# - Confidence score
# - Actionable suggestions
```

### Quick File Analysis

```python
# Analyze a specific file
"Analyze the structure of auth/login.py"

# Returns:
# - File content preview
# - Code structure (classes, functions)
# - Dependencies and imports
```

### Parallel Search

```python
# Search for multiple patterns simultaneously
"Search for 'login', 'authenticate', and 'session' in the codebase"

# Returns matches for each pattern with context
```

## Available MCP Tools

### `investigate`
Full LATS investigation of a task
- **Args**: task (str), max_depth (int), max_iterations (int), use_memory (bool)
- **Returns**: Solution path, file references, confidence score

### `get_status`
Get current investigation status
- **Returns**: Task, status, progress, current branch

### `search_memory`
Search past investigations
- **Args**: query (str), limit (int)
- **Returns**: Similar investigations with solutions

### `get_insights`
Retrieve relevant insights
- **Args**: context (str)
- **Returns**: List of relevant insights

### `analyze_file`
Quick single-file analysis
- **Args**: file_path (str)
- **Returns**: Content, structure, dependencies

### `parallel_search`
Search multiple patterns in parallel
- **Args**: patterns (List[str]), directory (str)
- **Returns**: Matches for each pattern

## How LATS Works

### 1. Tree Search Process
```
Root Node
├── Action 1 (Score: 6.5)
│   ├── Action 1.1 (Score: 7.8) ← Best path
│   └── Action 1.2 (Score: 5.2)
└── Action 2 (Score: 4.3)
    └── Action 2.1 (Score: 3.9)
```

### 2. Node Selection
Uses Upper Confidence Bound (UCT) to balance:
- **Exploitation**: Choose high-scoring paths
- **Exploration**: Try less-visited branches

### 3. Reflection & Scoring
Each action is evaluated on:
- Relevance to task (0-10 scale)
- Information quality
- Progress toward solution

### 4. Memory & Learning
- Stores successful investigations
- Extracts action patterns
- Provides suggestions for similar tasks

## Configuration

Edit `LATSConfig` in `lats_core.py`:

```python
class LATSConfig:
    model_name = "gpt-oss"          # Ollama model
    base_url = "http://localhost:11434"  # Ollama URL
    temperature = 0.7                # Model temperature
    max_depth = 5                    # Max tree depth
    max_iterations = 10              # Max search iterations
    num_expand = 5                   # Actions per expansion
    c_param = 1.414                  # UCT exploration parameter
    min_score_threshold = 7.0        # Solution threshold
```

## Architecture

```
┌─────────────────┐
│   MCP Client    │
│    (Claude)     │
└────────┬────────┘
         │ MCP Protocol
┌────────▼────────┐
│  FastMCP Server │
└────────┬────────┘
         │
┌────────▼────────┐
│  LATS Algorithm │
├─────────────────┤
│ • Tree Search   │
│ • Node Selection│
│ • Reflection    │
└────────┬────────┘
         │
┌────────▼────────────┐
│   Core Components   │
├────────┬────────────┤
│Filesystem│  Memory  │
│  Tools   │ Manager  │
└──────────┴──────────┘
         │
┌────────▼────────┐
│     Ollama      │
│   (gpt-oss)     │
└─────────────────┘
```

## Development

### Running Tests
```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=. tests/
```

### Adding New Tools
1. Add tool function to `filesystem_tools.py`
2. Register in `create_filesystem_tools()`
3. Update MCP server if needed

### Extending Memory
1. Add namespace in `MemoryManager.__init__`
2. Create storage/retrieval methods
3. Integrate with investigation flow

## Troubleshooting

### Ollama Connection Issues
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Verify model is available
ollama list | grep gpt-oss
```

### Memory Store Errors
- Check write permissions in directory
- Verify langmem is properly installed
- Review error namespace for details

### Tool Execution Failures
- Check file permissions
- Verify path existence
- Review size limits (1MB max)

## Performance Tips

1. **Adjust max_depth**: Lower for faster results
2. **Limit iterations**: Reduce for quicker investigations
3. **Use memory**: Leverages past investigations
4. **Parallel search**: Batch multiple queries
5. **Target searches**: Provide specific directories

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- LangChain/LangGraph for agent framework
- Anthropic for MCP protocol
- OpenAI for gpt-oss model
- langmem for memory management