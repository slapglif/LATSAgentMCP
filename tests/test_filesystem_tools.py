"""
Filesystem Tools Validation Tests
Real file operations with measurable outcomes
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from filesystem_tools import (
    read_file_with_lines,
    list_directory_tree,
    search_in_files,
    analyze_code_structure,
    find_dependencies,
    create_filesystem_tools
)


class FilesystemToolsTester:
    """Test filesystem tools with real operations"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent / "sample_codebase"
        self.results = {}
        self.tools = create_filesystem_tools()
    
    def test_read_file_operations(self) -> Dict[str, Any]:
        """Test file reading with various scenarios"""
        print("\n[TEST] File Reading Operations")
        results = {
            'basic_read': False,
            'line_range': False,
            'line_numbers': False,
            'error_handling': False,
            'large_file': False
        }
        
        # Test basic read
        content = read_file_with_lines(str(self.test_dir / "auth" / "login.py"))
        results['basic_read'] = "class LoginHandler" in content
        
        # Test line range reading
        content_range = read_file_with_lines(
            str(self.test_dir / "auth" / "login.py"),
            start_line=50,
            end_line=60
        )
        lines = content_range.split('\n')
        results['line_range'] = len(lines) <= 11  # 60-50+1
        
        # Test line numbers present
        results['line_numbers'] = all(
            line.strip() == "" or line[0:4].strip().isdigit() 
            for line in content.split('\n')[:10] if line
        )
        
        # Test error handling
        error_content = read_file_with_lines("/nonexistent/file.py")
        results['error_handling'] = "Error:" in error_content
        
        # Test large file handling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create 2MB file (over limit)
            f.write("x" * (2 * 1024 * 1024))
            temp_path = f.name
        
        large_content = read_file_with_lines(temp_path)
        results['large_file'] = "Error:" in large_content and "too large" in large_content
        os.unlink(temp_path)
        
        return results
    
    def test_directory_operations(self) -> Dict[str, Any]:
        """Test directory listing and tree generation"""
        print("\n[TEST] Directory Operations")
        results = {
            'tree_structure': False,
            'language_detection': False,
            'depth_limiting': False,
            'ignore_patterns': False,
            'file_sizes': False
        }
        
        # Test tree structure
        tree = list_directory_tree(str(self.test_dir), max_depth=2)
        results['tree_structure'] = (
            "auth/" in tree and 
            "database/" in tree and
            "utils/" in tree
        )
        
        # Test language detection
        results['language_detection'] = "[python]" in tree
        
        # Test depth limiting
        deep_tree = list_directory_tree(str(self.test_dir), max_depth=1)
        shallow_tree = list_directory_tree(str(self.test_dir), max_depth=0)
        results['depth_limiting'] = len(deep_tree) > len(shallow_tree)
        
        # Test ignore patterns
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            Path(tmpdir, "__pycache__").mkdir()
            Path(tmpdir, "test.py").write_text("test")
            Path(tmpdir, "__pycache__", "test.pyc").write_text("compiled")
            
            tree_with_cache = list_directory_tree(tmpdir, ignore_patterns=[])
            tree_without_cache = list_directory_tree(tmpdir)  # Default ignores __pycache__
            
            results['ignore_patterns'] = (
                "__pycache__" in tree_with_cache and
                "__pycache__" not in tree_without_cache
            )
        
        # Test file sizes shown
        results['file_sizes'] = "bytes" in tree or "B)" in tree
        
        return results
    
    def test_search_operations(self) -> Dict[str, Any]:
        """Test file searching capabilities"""
        print("\n[TEST] Search Operations")
        results = {
            'basic_search': False,
            'regex_search': False,
            'context_lines': False,
            'case_sensitivity': False,
            'file_filtering': False
        }
        
        # Test basic search
        search_result = search_in_files("BUG", str(self.test_dir))
        results['basic_search'] = "BUG:" in search_result
        
        # Test regex search
        regex_result = search_in_files(r"class\s+\w+Error", str(self.test_dir))
        results['regex_search'] = (
            "AuthenticationError" in regex_result or
            "DatabaseError" in regex_result
        )
        
        # Test context lines
        context_result = search_in_files("validate_session", str(self.test_dir))
        lines = context_result.split('\n')
        # Check for line numbers and context markers
        results['context_lines'] = any(">>>" in line for line in lines)
        
        # Test case sensitivity
        case_sensitive = search_in_files("bug", str(self.test_dir), case_sensitive=True)
        case_insensitive = search_in_files("bug", str(self.test_dir), case_sensitive=False)
        results['case_sensitivity'] = (
            "No matches" in case_sensitive or 
            case_insensitive.count("BUG") > case_sensitive.count("bug")
        )
        
        # Test file pattern filtering
        py_only = search_in_files("class", str(self.test_dir), file_pattern="*.py")
        results['file_filtering'] = ".py:" in py_only
        
        return results
    
    def test_code_analysis(self) -> Dict[str, Any]:
        """Test code structure analysis"""
        print("\n[TEST] Code Analysis")
        results = {
            'ast_parsing': False,
            'class_detection': False,
            'function_detection': False,
            'import_detection': False,
            'line_numbers': False
        }
        
        # Analyze Python file
        analysis = analyze_code_structure(str(self.test_dir / "auth" / "login.py"))
        
        results['ast_parsing'] = "Code Structure Analysis:" in analysis
        results['class_detection'] = (
            "LoginHandler" in analysis and
            "SessionManager" in analysis
        )
        results['function_detection'] = (
            "authenticate" in analysis and
            "validate_session" in analysis
        )
        results['import_detection'] = "Imports:" in analysis
        results['line_numbers'] = "line" in analysis.lower()
        
        # Test error handling
        error_analysis = analyze_code_structure(str(self.test_dir / "nonexistent.py"))
        if "Error:" not in error_analysis:
            results['ast_parsing'] = False
        
        return results
    
    def test_dependency_analysis(self) -> Dict[str, Any]:
        """Test dependency extraction"""
        print("\n[TEST] Dependency Analysis")
        results = {
            'python_deps': False,
            'external_imports': False,
            'local_imports': False,
            'stdlib_detection': False
        }
        
        # Test Python dependencies
        py_deps = find_dependencies(str(self.test_dir / "auth" / "login.py"))
        results['python_deps'] = "Python Dependencies" in py_deps
        results['external_imports'] = (
            "hashlib" in py_deps or
            "datetime" in py_deps or
            "json" in py_deps
        )
        results['local_imports'] = "Local" in py_deps or "relative" in py_deps
        results['stdlib_detection'] = (
            "typing" in py_deps or
            "json" in py_deps
        )
        
        return results
    
    def test_tool_integration(self) -> Dict[str, Any]:
        """Test Langchain tool integration"""
        print("\n[TEST] Tool Integration")
        results = {
            'tools_created': False,
            'tool_execution': False,
            'arg_parsing': False,
            'error_handling': False
        }
        
        # Test tools created
        results['tools_created'] = len(self.tools) >= 5
        
        # Test tool execution
        read_tool = next((t for t in self.tools if t.name == "read_file"), None)
        if read_tool:
            # Test with dict args
            result = read_tool.func({'file_path': str(self.test_dir / "auth" / "__init__.py")})
            results['tool_execution'] = "LoginHandler" in result
            
            # Test with string args (backward compatibility)
            result2 = read_tool.func(str(self.test_dir / "auth" / "__init__.py"))
            results['arg_parsing'] = "LoginHandler" in result2
        
        # Test error handling in tools
        search_tool = next((t for t in self.tools if t.name == "search_files"), None)
        if search_tool:
            error_result = search_tool.func({'pattern': '[invalid(regex'})
            results['error_handling'] = "Error" in error_result
        
        return results
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        print("\n[TEST] Performance Tests")
        import time
        
        results = {
            'read_speed': 0,
            'search_speed': 0,
            'tree_speed': 0,
            'analysis_speed': 0
        }
        
        # Test read performance
        start = time.time()
        for _ in range(10):
            read_file_with_lines(str(self.test_dir / "auth" / "login.py"))
        results['read_speed'] = (time.time() - start) / 10
        
        # Test search performance
        start = time.time()
        search_in_files("def", str(self.test_dir), max_results=20)
        results['search_speed'] = time.time() - start
        
        # Test tree performance
        start = time.time()
        list_directory_tree(str(self.test_dir), max_depth=3)
        results['tree_speed'] = time.time() - start
        
        # Test analysis performance
        start = time.time()
        analyze_code_structure(str(self.test_dir / "database" / "connection.py"))
        results['analysis_speed'] = time.time() - start
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all filesystem tool tests"""
        print("="*60)
        print("FILESYSTEM TOOLS VALIDATION TESTS")
        print("="*60)
        
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        try:
            # Run test suites
            self.results['read_operations'] = self.test_read_file_operations()
            self.results['directory_operations'] = self.test_directory_operations()
            self.results['search_operations'] = self.test_search_operations()
            self.results['code_analysis'] = self.test_code_analysis()
            self.results['dependency_analysis'] = self.test_dependency_analysis()
            self.results['tool_integration'] = self.test_tool_integration()
            self.results['performance'] = self.test_performance()
            
            # Generate report
            self._generate_report()
            
            # Calculate overall success
            total_tests = 0
            passed_tests = 0
            
            for suite_name, suite_results in self.results.items():
                if suite_name != 'performance':
                    for test_name, passed in suite_results.items():
                        total_tests += 1
                        if passed:
                            passed_tests += 1
            
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            return {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'success_rate': success_rate,
                'performance': self.results['performance']
            }
            
        finally:
            os.chdir(original_dir)
    
    def _generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        for suite_name, suite_results in self.results.items():
            if suite_name == 'performance':
                continue
                
            print(f"\n{suite_name.upper().replace('_', ' ')}:")
            
            for test_name, result in suite_results.items():
                if isinstance(result, bool):
                    status = "✓" if result else "✗"
                    print(f"  {status} {test_name}")
                else:
                    print(f"  • {test_name}: {result}")
        
        # Performance results
        print("\nPERFORMANCE METRICS:")
        perf = self.results['performance']
        print(f"  Read speed: {perf['read_speed']*1000:.2f}ms avg")
        print(f"  Search speed: {perf['search_speed']*1000:.2f}ms")
        print(f"  Tree speed: {perf['tree_speed']*1000:.2f}ms")
        print(f"  Analysis speed: {perf['analysis_speed']*1000:.2f}ms")
        
        # Identify issues
        print("\nIDENTIFIED ISSUES:")
        issues = self._identify_issues()
        if issues:
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  No significant issues found")
    
    def _identify_issues(self) -> List[str]:
        """Identify issues in filesystem tools"""
        issues = []
        
        # Check for failures
        for suite_name, suite_results in self.results.items():
            if suite_name == 'performance':
                continue
            
            failed = [name for name, result in suite_results.items() if not result]
            if failed:
                issues.append(f"{suite_name}: {', '.join(failed)} failed")
        
        # Check performance
        perf = self.results['performance']
        if perf['read_speed'] > 0.01:  # 10ms threshold
            issues.append("File reading is slow")
        if perf['search_speed'] > 1.0:  # 1 second threshold
            issues.append("Search operation is slow")
        if perf['analysis_speed'] > 0.5:  # 500ms threshold
            issues.append("Code analysis is slow")
        
        return issues


def main():
    """Run filesystem tools tests"""
    tester = FilesystemToolsTester()
    results = tester.run_all_tests()
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {results['passed']}/{results['total_tests']} tests passed")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    
    if results['success_rate'] >= 0.9:
        print("✓ FILESYSTEM TOOLS VALIDATION PASSED")
        return 0
    else:
        print("✗ FILESYSTEM TOOLS VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())