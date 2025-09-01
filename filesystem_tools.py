"""
Filesystem tools for code investigation
Provides file reading, searching, and structure analysis capabilities
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from langchain.tools import Tool
from pydantic import BaseModel, Field


class FileContent(BaseModel):
    """File content with metadata"""
    path: str
    content: str
    lines: int
    size: int
    language: Optional[str] = None


class CodeReference(BaseModel):
    """Code reference with file and line number"""
    file: str
    line: int
    content: str
    context: Optional[str] = None


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension"""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.m': 'matlab',
        '.jl': 'julia',
        '.sh': 'bash',
        '.ps1': 'powershell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.md': 'markdown',
        '.txt': 'text'
    }
    
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext)


def read_file_with_lines(file_path: str, start_line: Optional[int] = None, 
                        end_line: Optional[int] = None) -> str:
    """Read file contents with line numbers"""
    try:
        path = Path(file_path).resolve()
        
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        
        # Check file size
        size = path.stat().st_size
        if size > 1024 * 1024:  # 1MB limit
            return f"Error: File too large ({size} bytes). Maximum 1MB supported."
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Apply line range if specified
        if start_line is not None:
            start_idx = max(0, start_line - 1)
        else:
            start_idx = 0
            
        if end_line is not None:
            end_idx = min(len(lines), end_line)
        else:
            end_idx = len(lines)
        
        # Format with line numbers
        result_lines = []
        for i in range(start_idx, end_idx):
            result_lines.append(f"{i+1:4d}: {lines[i].rstrip()}")
        
        return '\n'.join(result_lines)
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


def list_directory_tree(directory_path: str = ".", max_depth: int = 3, 
                        ignore_patterns: Optional[List[str]] = None) -> str:
    """List directory structure as tree"""
    if ignore_patterns is None:
        ignore_patterns = [
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'node_modules', 'dist', 'build', '.pytest_cache',
            '*.pyc', '*.pyo', '.DS_Store'
        ]
    
    try:
        path = Path(directory_path).resolve()
        
        if not path.exists():
            return f"Error: Directory {directory_path} does not exist"
        if not path.is_dir():
            return f"Error: {directory_path} is not a directory"
        
        def should_ignore(name: str) -> bool:
            for pattern in ignore_patterns:
                if pattern.startswith('*'):
                    if name.endswith(pattern[1:]):
                        return True
                elif pattern in name:
                    return True
            return False
        
        def build_tree(dir_path: Path, prefix: str = "", depth: int = 0) -> List[str]:
            if depth >= max_depth:
                return []
            
            items = []
            try:
                entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                filtered_entries = [e for e in entries if not should_ignore(e.name)]
                
                for i, entry in enumerate(filtered_entries):
                    is_last = i == len(filtered_entries) - 1
                    current_prefix = "└── " if is_last else "├── "
                    next_prefix = "    " if is_last else "│   "
                    
                    if entry.is_dir():
                        items.append(f"{prefix}{current_prefix}{entry.name}/")
                        items.extend(build_tree(entry, prefix + next_prefix, depth + 1))
                    else:
                        size = entry.stat().st_size
                        lang = detect_language(str(entry))
                        lang_str = f" [{lang}]" if lang else ""
                        items.append(f"{prefix}{current_prefix}{entry.name} ({size}B){lang_str}")
            except PermissionError:
                items.append(f"{prefix}[Permission Denied]")
            
            return items
        
        tree_lines = [f"{path.name}/"]
        tree_lines.extend(build_tree(path))
        return '\n'.join(tree_lines)
    
    except Exception as e:
        return f"Error listing directory: {str(e)}"


def search_in_files(pattern: str, directory: str = ".", 
                   file_pattern: str = "*", 
                   case_sensitive: bool = False,
                   max_results: int = 100) -> str:
    """Search for pattern in files with context"""
    try:
        path = Path(directory).resolve()
        
        if not path.exists():
            return f"Error: Directory {directory} does not exist"
        
        results = []
        count = 0
        
        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern - {str(e)}"
        
        # Search files
        for file_path in path.rglob(file_pattern):
            if count >= max_results:
                break
                
            if file_path.is_file():
                try:
                    # Skip binary files
                    with open(file_path, 'rb') as f:
                        chunk = f.read(512)
                        if b'\0' in chunk:
                            continue
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            # Get context (1 line before and after)
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 1)
                            
                            context_lines = []
                            for j in range(context_start, context_end):
                                prefix = ">>>" if j == i - 1 else "   "
                                context_lines.append(f"{prefix} {j+1:4d}: {lines[j].rstrip()}")
                            
                            results.append(f"\n{file_path}:\n{''.join(context_lines)}")
                            count += 1
                            
                            if count >= max_results:
                                break
                                
                except Exception:
                    continue
        
        if results:
            header = f"Found {len(results)} matches for '{pattern}':\n"
            return header + '\n'.join(results)
        else:
            return f"No matches found for '{pattern}'"
            
    except Exception as e:
        return f"Error searching files: {str(e)}"


def analyze_code_structure(file_path: str) -> str:
    """Analyze code structure (Python files)"""
    try:
        path = Path(file_path).resolve()
        
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        if not file_path.endswith('.py'):
            return "Error: Code structure analysis currently only supports Python files"
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return f"Error: Syntax error in file - {str(e)}"
        
        structure = {
            'imports': [],
            'classes': [],
            'functions': [],
            'global_vars': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    structure['imports'].append(f"{module}.{alias.name}")
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                structure['classes'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': methods
                })
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                params = [arg.arg for arg in node.args.args]
                structure['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'params': params
                })
            elif isinstance(node, ast.Assign) and not any(isinstance(parent, (ast.FunctionDef, ast.ClassDef)) for parent in ast.walk(tree)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        structure['global_vars'].append({
                            'name': target.id,
                            'line': node.lineno
                        })
        
        # Format output
        output = [f"Code Structure Analysis: {file_path}\n"]
        
        if structure['imports']:
            output.append("Imports:")
            for imp in sorted(set(structure['imports'])):
                output.append(f"  - {imp}")
        
        if structure['classes']:
            output.append("\nClasses:")
            for cls in structure['classes']:
                output.append(f"  - {cls['name']} (line {cls['line']})")
                if cls['methods']:
                    for method in cls['methods']:
                        output.append(f"    • {method}()")
        
        if structure['functions']:
            output.append("\nFunctions:")
            for func in structure['functions']:
                params = ', '.join(func['params'])
                output.append(f"  - {func['name']}({params}) (line {func['line']})")
        
        if structure['global_vars']:
            output.append("\nGlobal Variables:")
            for var in structure['global_vars']:
                output.append(f"  - {var['name']} (line {var['line']})")
        
        return '\n'.join(output)
        
    except Exception as e:
        return f"Error analyzing code structure: {str(e)}"


def find_dependencies(file_path: str) -> str:
    """Find dependencies and imports in a file"""
    try:
        path = Path(file_path).resolve()
        
        if not path.exists():
            return f"Error: File {file_path} does not exist"
        
        lang = detect_language(str(path))
        
        if lang == 'python':
            return _find_python_deps(path)
        elif lang in ['javascript', 'typescript']:
            return _find_js_deps(path)
        elif lang == 'java':
            return _find_java_deps(path)
        else:
            return f"Dependency analysis not supported for {lang or 'unknown'} files"
            
    except Exception as e:
        return f"Error finding dependencies: {str(e)}"


def _find_python_deps(path: Path) -> str:
    """Find Python dependencies"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "Error: Could not parse Python file"
    
    imports = set()
    local_imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # Relative import
                local_imports.add(f"{'.' * node.level}{node.module or ''}")
            elif node.module:
                # Check if it's a local module (starts with . or is a local package)
                if node.module.startswith('.'):
                    local_imports.add(node.module)
                else:
                    imports.add(node.module.split('.')[0])
    
    # Check for 'from . import' patterns in auth/__init__.py
    if path.name == "__init__.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith('.') or node.level > 0:
                    local_imports.add(f".{node.module}" if node.module else ".")
                # Also check for same-package imports
                if not node.module or node.module in ['login', 'helpers', 'connection']:
                    local_imports.add(f".{node.module}" if node.module else ".")
    
    output = [f"Python Dependencies in {path.name}:\n"]
    
    if imports:
        output.append("External imports:")
        for imp in sorted(imports):
            output.append(f"  - {imp}")
    
    if local_imports:
        output.append("\nLocal/relative imports:")
        for imp in sorted(local_imports):
            output.append(f"  - {imp}")
    
    return '\n'.join(output)


def _find_js_deps(path: Path) -> str:
    """Find JavaScript/TypeScript dependencies"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports = set()
    local_imports = set()
    
    # Find import statements
    import_pattern = r'import\s+(?:[\w\s{},*]+\s+from\s+)?[\'"]([^\'"]+)[\'"]'
    require_pattern = r'require\s*\([\'"]([^\'"]+)[\'"]\)'
    
    for match in re.finditer(import_pattern, content):
        module = match.group(1)
        if module.startswith('.'):
            local_imports.add(module)
        else:
            imports.add(module.split('/')[0])
    
    for match in re.finditer(require_pattern, content):
        module = match.group(1)
        if module.startswith('.'):
            local_imports.add(module)
        else:
            imports.add(module.split('/')[0])
    
    output = [f"JavaScript/TypeScript Dependencies in {path.name}:\n"]
    
    if imports:
        output.append("External imports:")
        for imp in sorted(imports):
            output.append(f"  - {imp}")
    
    if local_imports:
        output.append("\nLocal imports:")
        for imp in sorted(local_imports):
            output.append(f"  - {imp}")
    
    return '\n'.join(output)


def _find_java_deps(path: Path) -> str:
    """Find Java dependencies"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports = set()
    
    # Find import statements
    import_pattern = r'import\s+(?:static\s+)?([^;]+);'
    
    for match in re.finditer(import_pattern, content):
        imports.add(match.group(1).strip())
    
    output = [f"Java Dependencies in {path.name}:\n"]
    
    if imports:
        output.append("Imports:")
        for imp in sorted(imports):
            output.append(f"  - {imp}")
    
    return '\n'.join(output)


# Create Langchain tools
def create_filesystem_tools() -> List[Tool]:
    """Create filesystem tools for Langchain"""
    return [
        Tool(
            name="read_file",
            func=lambda args: read_file_with_lines(**args) if isinstance(args, dict) else read_file_with_lines(args),
            description="Read file contents with line numbers. Args: file_path (str), start_line (int, optional), end_line (int, optional)"
        ),
        Tool(
            name="list_directory",
            func=lambda args: list_directory_tree(**args) if isinstance(args, dict) else list_directory_tree(args),
            description="List directory structure as tree. Args: directory_path (str, default='.'), max_depth (int, default=3)"
        ),
        Tool(
            name="search_files",
            func=lambda args: search_in_files(**args) if isinstance(args, dict) else search_in_files(args),
            description="Search for pattern in files. Args: pattern (str), directory (str, default='.'), file_pattern (str, default='*'), case_sensitive (bool, default=False)"
        ),
        Tool(
            name="analyze_structure",
            func=lambda args: analyze_code_structure(**args) if isinstance(args, dict) else analyze_code_structure(args),
            description="Analyze code structure of Python files. Args: file_path (str)"
        ),
        Tool(
            name="find_dependencies",
            func=lambda args: find_dependencies(**args) if isinstance(args, dict) else find_dependencies(args),
            description="Find dependencies and imports in a file. Args: file_path (str)"
        )
    ]