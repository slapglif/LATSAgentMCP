## Codebase Overview

This is a **Language Agent Tree Search (LATS)** implementation with MCP (Model Context Protocol) server integration. Here's what I found:

### Core Components:
- **LATS Implementation**: `lats_langgraph.py` (68KB) - Main implementation using LangGraph
- **MCP Server**: `mcp_server.py` and `lats_mcp_server.py` - MCP server implementations for Claude integration
- **Memory Management**: `memory_manager.py` - Uses LangMem for persistent memory with SQLite
- **Filesystem Tools**: `filesystem_tools.py` - File operations and code analysis utilities

### Key Features:
- Tree search algorithm for complex problem solving
- Integration with LangChain/LangGraph
- MCP server for Claude Code integration
- Persistent memory using SQLite databases
- Comprehensive test suite in `tests/` directory

### Testing Infrastructure:
- Multiple test files covering integration, performance, and verification
- Sample codebase for testing in `tests/sample_codebase/`
- Test databases for memory persistence

### Current State:
- Modified files: `lats_langgraph.py` with pending changes
- Several untracked files including MCP configurations and test outputs
- Active development with recent MCP server additions

The codebase implements a sophisticated AI agent system using tree search techniques for code analysis and problem-solving, with Claude integration through MCP.
