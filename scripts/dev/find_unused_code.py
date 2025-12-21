#!/usr/bin/env python3
"""
Unused Code Finder for MIRA3

Locates abandoned files and functions that aren't used anywhere else.
Intelligently excludes expected standalone files like entry points, tests, and configs.

Usage:
    python find_unused_code.py [--verbose] [--include-tests] [--json]
"""

import ast
import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional


# ============================================================================
# Configuration: Files/patterns we expect to be standalone (not imported)
# ============================================================================

# Entry points and standalone scripts (relative to project root)
EXPECTED_STANDALONE_FILES = {
    # Python entry points
    'python/mira/main.py',          # Backend entry point
    'python/mira/bootstrap.py',     # Self-bootstrapping module
    'python/mira/__init__.py',      # Package init
    'python/mira_backend.py',       # External entry point (called by Node.js)

    # TypeScript entry points
    'src/cli.ts',                   # CLI entry point
    'src/index.ts',                 # Package entry

    # Config and utility scripts
    'find_unused_code.py',          # This script itself
}

# Patterns for files that are expected to be standalone
STANDALONE_PATTERNS = [
    r'__init__\.py$',               # Package inits
    r'__main__\.py$',               # Module entry points
    r'test_.*\.py$',                # Test files
    r'.*_test\.py$',                # Test files (alt naming)
    r'.*\.test\.(ts|js)$',          # JS/TS test files
    r'.*\.spec\.(ts|js)$',          # JS/TS spec files
    r'conftest\.py$',               # Pytest config
    r'setup\.py$',                  # Package setup
    r'migrations?\.py$',            # DB migrations (often standalone)
]

# Directories to skip entirely
SKIP_DIRECTORIES = {
    'node_modules', '.git', '.venv', 'venv', '__pycache__',
    'dist', 'build', '.mira', 'coverage', '.nyc_output',
}

# File extensions to analyze
PYTHON_EXTENSIONS = {'.py'}
TYPESCRIPT_EXTENSIONS = {'.ts', '.tsx', '.js', '.jsx'}


# ============================================================================
# Python Analysis
# ============================================================================

class PythonAnalyzer(ast.NodeVisitor):
    """AST visitor to extract definitions and references from Python code."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.definitions: Dict[str, int] = {}  # name -> line number
        self.references: Set[str] = set()
        self.imports: Set[str] = set()  # module names imported
        self.import_froms: Dict[str, Set[str]] = defaultdict(set)  # module -> names
        self._scope_depth = 0  # Track nesting depth

    def _is_decorated_endpoint(self, node) -> bool:
        """Check if function has API endpoint decorators."""
        if not node.decorator_list:
            return False
        for decorator in node.decorator_list:
            # Check for @app.get, @app.post, @router.get, etc.
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if decorator.func.attr in {'get', 'post', 'put', 'delete', 'patch', 'route', 'api_route'}:
                        return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in {'get', 'post', 'put', 'delete', 'patch', 'route'}:
                    return True
        return False

    def visit_FunctionDef(self, node):
        # Only capture top-level function definitions (not nested functions)
        if self._scope_depth == 0:
            # Skip private/dunder methods
            if not node.name.startswith('_') or node.name.startswith('__') and node.name.endswith('__'):
                # Skip decorated API endpoints
                if not self._is_decorated_endpoint(node):
                    self.definitions[node.name] = node.lineno

        # Track scope for nested functions
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        # Only capture top-level class definitions
        if self._scope_depth == 0:
            self.definitions[node.name] = node.lineno
        # Methods inside classes are not captured (they're called on instances)
        # We increase scope depth so nested functions/classes inside are skipped
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split('.')[0]
            self.imports.add(module)

    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split('.')[0]
            self.imports.add(module)
            for alias in node.names:
                if alias.name != '*':
                    self.import_froms[node.module].add(alias.name)

    def visit_Name(self, node):
        self.references.add(node.id)

    def visit_Attribute(self, node):
        self.references.add(node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.references.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.references.add(node.func.attr)
        self.generic_visit(node)


def analyze_python_file(filepath: Path) -> Optional[PythonAnalyzer]:
    """Parse a Python file and extract definitions/references."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()
        tree = ast.parse(source, filename=str(filepath))
        analyzer = PythonAnalyzer(str(filepath))
        analyzer.visit(tree)
        return analyzer
    except SyntaxError as e:
        print(f"  Warning: Syntax error in {filepath}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return None


# ============================================================================
# TypeScript/JavaScript Analysis
# ============================================================================

def analyze_typescript_file(filepath: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Simple regex-based analysis for TypeScript/JavaScript.
    Returns (exports, imports, references).
    """
    exports = set()
    imports = set()
    references = set()

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return exports, imports, references

    # Find exports
    # export function name
    for match in re.finditer(r'export\s+(?:async\s+)?function\s+(\w+)', content):
        exports.add(match.group(1))
    # export class name
    for match in re.finditer(r'export\s+class\s+(\w+)', content):
        exports.add(match.group(1))
    # export const/let/var name
    for match in re.finditer(r'export\s+(?:const|let|var)\s+(\w+)', content):
        exports.add(match.group(1))
    # export { name1, name2 }
    for match in re.finditer(r'export\s*\{([^}]+)\}', content):
        for name in match.group(1).split(','):
            name = name.strip().split(' as ')[0].strip()
            if name and not name.startswith('type '):
                exports.add(name)
    # export default
    for match in re.finditer(r'export\s+default\s+(?:class|function)?\s*(\w+)?', content):
        if match.group(1):
            exports.add(match.group(1))

    # Find imports
    # import { x, y } from 'module'
    for match in re.finditer(r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", content):
        module = match.group(2)
        if not module.startswith('.'):
            continue  # Skip external packages
        imports.add(module)
        for name in match.group(1).split(','):
            name = name.strip().split(' as ')[0].strip()
            if name and not name.startswith('type '):
                references.add(name)

    # import x from 'module'
    for match in re.finditer(r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]", content):
        module = match.group(2)
        if module.startswith('.'):
            imports.add(module)
            references.add(match.group(1))

    # import * as x from 'module'
    for match in re.finditer(r"import\s*\*\s*as\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]", content):
        module = match.group(2)
        if module.startswith('.'):
            imports.add(module)

    return exports, imports, references


# ============================================================================
# Dependency Graph Building
# ============================================================================

def is_standalone_file(filepath: Path, project_root: Path) -> Tuple[bool, str]:
    """
    Check if a file is expected to be standalone (not imported).
    Returns (is_standalone, reason).
    """
    rel_path = str(filepath.relative_to(project_root))

    # Check explicit standalone files
    if rel_path in EXPECTED_STANDALONE_FILES:
        return True, "configured as entry point"

    # Check patterns
    filename = filepath.name
    for pattern in STANDALONE_PATTERNS:
        if re.search(pattern, filename):
            return True, f"matches pattern: {pattern}"

    # Check if it's in a test directory
    if 'test' in filepath.parts or 'tests' in filepath.parts:
        return True, "in test directory"

    return False, ""


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in the project."""
    files = []
    for path in root.rglob('*.py'):
        if any(skip in path.parts for skip in SKIP_DIRECTORIES):
            continue
        files.append(path)
    return files


def find_typescript_files(root: Path) -> List[Path]:
    """Find all TypeScript/JavaScript files in the project."""
    files = []
    for ext in TYPESCRIPT_EXTENSIONS:
        for path in root.rglob(f'*{ext}'):
            if any(skip in path.parts for skip in SKIP_DIRECTORIES):
                continue
            files.append(path)
    return files


def build_python_import_graph(files: List[Path], project_root: Path) -> Dict[str, Set[str]]:
    """
    Build a graph of which files import which modules.
    Returns {module_name: set of files that import it}
    """
    import_graph = defaultdict(set)

    for filepath in files:
        analyzer = analyze_python_file(filepath)
        if not analyzer:
            continue

        rel_path = str(filepath.relative_to(project_root))

        # Track what this file imports
        for module in analyzer.imports:
            import_graph[module].add(rel_path)

        for module, names in analyzer.import_froms.items():
            import_graph[module].add(rel_path)
            for name in names:
                import_graph[f"{module}.{name}"].add(rel_path)

    return import_graph


def get_module_name(filepath: Path, project_root: Path) -> str:
    """Convert a file path to its Python module name."""
    rel = filepath.relative_to(project_root)
    parts = list(rel.parts)

    # Remove .py extension
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]

    # Handle __init__.py
    if parts[-1] == '__init__':
        parts = parts[:-1]

    return '.'.join(parts)


# ============================================================================
# Unused Code Detection
# ============================================================================

def find_unused_python_files(project_root: Path, verbose: bool = False) -> List[Dict]:
    """Find Python files that aren't imported anywhere."""
    unused = []
    py_files = find_python_files(project_root)
    import_graph = build_python_import_graph(py_files, project_root)

    for filepath in py_files:
        is_standalone, reason = is_standalone_file(filepath, project_root)
        if is_standalone:
            if verbose:
                print(f"  Skipping {filepath.name}: {reason}")
            continue

        module_name = get_module_name(filepath, project_root)
        rel_path = str(filepath.relative_to(project_root))

        # Check various forms of the module name
        module_parts = module_name.split('.')
        short_name = module_parts[-1] if module_parts else module_name

        is_imported = False
        imported_by = set()

        # Check if any form of this module is imported
        for key, importers in import_graph.items():
            key_parts = key.split('.')
            # Match full module path or short name at end
            if (key == module_name or
                key.endswith('.' + short_name) or
                key == short_name or
                (len(key_parts) >= 2 and '.'.join(key_parts[-2:]) == '.'.join(module_parts[-2:]) if len(module_parts) >= 2 else False)):
                is_imported = True
                imported_by.update(importers)

        if not is_imported:
            unused.append({
                'file': rel_path,
                'module': module_name,
                'type': 'file',
                'language': 'python',
            })

    return unused


def find_unused_python_functions(project_root: Path, verbose: bool = False) -> List[Dict]:
    """Find Python functions/classes that aren't referenced anywhere."""
    unused = []
    py_files = find_python_files(project_root)

    # Collect all definitions and all references across the codebase
    all_definitions: Dict[str, List[Dict]] = defaultdict(list)  # name -> [{file, line}]
    all_references: Set[str] = set()

    for filepath in py_files:
        analyzer = analyze_python_file(filepath)
        if not analyzer:
            continue

        rel_path = str(filepath.relative_to(project_root))

        # Skip test files for definition tracking
        if 'test' in rel_path.lower():
            # But still collect references from tests
            all_references.update(analyzer.references)
            continue

        for name, line in analyzer.definitions.items():
            all_definitions[name].append({
                'file': rel_path,
                'line': line,
            })

        all_references.update(analyzer.references)

        # References from imports
        for module, names in analyzer.import_froms.items():
            all_references.update(names)

    # Find definitions that are never referenced
    for name, locations in all_definitions.items():
        if name not in all_references:
            # Skip private functions (single underscore)
            if name.startswith('_') and not name.startswith('__'):
                continue
            # Skip common dunder methods
            if name.startswith('__') and name.endswith('__'):
                continue
            # Skip common patterns that might be called dynamically
            if name in {'main', 'setup', 'teardown', 'run', 'execute', 'handler', 'callback'}:
                continue
            # Skip AST visitor methods (called by ast.NodeVisitor framework)
            if name.startswith('visit_'):
                continue
            # Skip migration functions (called via decorator registry)
            if name.startswith('migrate_v') or name.startswith('migration_'):
                continue
            # Skip watchdog/event handler callbacks
            if name.startswith('on_') and name in {'on_created', 'on_modified', 'on_deleted', 'on_moved', 'on_any_event'}:
                continue
            # Skip FastAPI/Flask lifecycle events
            if name in {'startup', 'shutdown', 'lifespan', 'on_startup', 'on_shutdown'}:
                continue
            # Skip pytest fixtures and hooks
            if name in {'pytest_configure', 'pytest_collection', 'conftest'}:
                continue
            # Skip common CLI/API patterns
            if name in {'cli', 'app', 'api', 'router', 'blueprint'}:
                continue
            # Skip exception classes (often raised but not "called")
            if name.endswith('Error') or name.endswith('Exception'):
                continue
            # Skip Enum-like classes and status classes
            if name.endswith('Status') or name.endswith('State') or name.endswith('Type'):
                continue

            for loc in locations:
                unused.append({
                    'file': loc['file'],
                    'name': name,
                    'line': loc['line'],
                    'type': 'function',
                    'language': 'python',
                })

    return unused


def find_unused_typescript_files(project_root: Path, verbose: bool = False) -> List[Dict]:
    """Find TypeScript/JavaScript files that aren't imported anywhere."""
    unused = []
    ts_files = find_typescript_files(project_root)

    # Build import graph - normalize all imports by removing extensions
    all_imports: Set[str] = set()

    for filepath in ts_files:
        _, imports, _ = analyze_typescript_file(filepath)
        for imp in imports:
            # Normalize: remove .js, .ts, .tsx extensions from imports
            normalized = re.sub(r'\.(js|ts|tsx|jsx)$', '', imp)
            all_imports.add(normalized)

    for filepath in ts_files:
        is_standalone, reason = is_standalone_file(filepath, project_root)
        if is_standalone:
            if verbose:
                print(f"  Skipping {filepath.name}: {reason}")
            continue

        rel_path = str(filepath.relative_to(project_root))

        # Generate normalized path forms that could match imports
        # Remove extension from the file path
        file_stem = re.sub(r'\.(ts|tsx|js|jsx)$', '', rel_path)

        # Check if this file is imported using various relative path forms
        is_imported = False

        for imp in all_imports:
            # Normalize the import path
            imp_normalized = imp.lstrip('./')

            # Check if the import path ends with our file path (handles various relative paths)
            if file_stem.endswith(imp_normalized) or imp_normalized.endswith(file_stem.split('/')[-1]):
                is_imported = True
                break

            # Handle src-relative paths
            if 'src/' in file_stem:
                src_relative = file_stem.split('src/')[-1]
                if imp_normalized.endswith(src_relative) or src_relative.endswith(imp_normalized.lstrip('../')):
                    is_imported = True
                    break

        if not is_imported:
            unused.append({
                'file': rel_path,
                'type': 'file',
                'language': 'typescript',
            })

    return unused


def find_unused_typescript_exports(project_root: Path, verbose: bool = False) -> List[Dict]:
    """Find TypeScript/JavaScript exports that aren't imported anywhere."""
    unused = []
    ts_files = find_typescript_files(project_root)

    # Collect all exports and all imported names
    all_exports: Dict[str, List[Dict]] = defaultdict(list)
    all_imported_names: Set[str] = set()

    for filepath in ts_files:
        exports, _, references = analyze_typescript_file(filepath)
        rel_path = str(filepath.relative_to(project_root))

        # Skip test files
        if 'test' in rel_path.lower() or 'spec' in rel_path.lower():
            all_imported_names.update(references)
            continue

        for name in exports:
            all_exports[name].append({'file': rel_path})

        all_imported_names.update(references)

    # Find exports never imported
    for name, locations in all_exports.items():
        if name not in all_imported_names:
            # Skip default exports and common patterns
            if name in {'default', 'handler', 'main', 'setup'}:
                continue

            for loc in locations:
                unused.append({
                    'file': loc['file'],
                    'name': name,
                    'type': 'export',
                    'language': 'typescript',
                })

    return unused


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Find unused code in the MIRA3 codebase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python find_unused_code.py              # Basic scan
    python find_unused_code.py --verbose    # Show skipped files
    python find_unused_code.py --json       # Output as JSON
    python find_unused_code.py --functions  # Include unused functions
        """
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show files being skipped and why')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--include-tests', action='store_true',
                        help='Include test files in analysis')
    parser.add_argument('--functions', '-f', action='store_true',
                        help='Also find unused functions (slower)')
    parser.add_argument('--path', type=Path, default=Path('.'),
                        help='Project root path (default: current directory)')

    args = parser.parse_args()
    project_root = args.path.resolve()

    if not args.json:
        print(f"Scanning for unused code in: {project_root}\n")

    results = {
        'unused_files': [],
        'unused_functions': [],
        'summary': {},
    }

    # Find unused Python files
    if not args.json:
        print("Analyzing Python files...")
    unused_py_files = find_unused_python_files(project_root, args.verbose)
    results['unused_files'].extend(unused_py_files)

    # Find unused TypeScript files
    if not args.json:
        print("Analyzing TypeScript/JavaScript files...")
    unused_ts_files = find_unused_typescript_files(project_root, args.verbose)
    results['unused_files'].extend(unused_ts_files)

    # Optionally find unused functions
    if args.functions:
        if not args.json:
            print("Analyzing function usage...")
        unused_py_funcs = find_unused_python_functions(project_root, args.verbose)
        unused_ts_exports = find_unused_typescript_exports(project_root, args.verbose)
        results['unused_functions'].extend(unused_py_funcs)
        results['unused_functions'].extend(unused_ts_exports)

    # Summary
    results['summary'] = {
        'unused_files': len(results['unused_files']),
        'unused_functions': len(results['unused_functions']),
    }

    # Output
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("UNUSED CODE REPORT")
        print("=" * 60)

        if results['unused_files']:
            print(f"\n## Potentially Unused Files ({len(results['unused_files'])})")
            print("-" * 40)

            # Group by language
            by_lang = defaultdict(list)
            for item in results['unused_files']:
                by_lang[item['language']].append(item)

            for lang, items in sorted(by_lang.items()):
                print(f"\n### {lang.title()}")
                for item in sorted(items, key=lambda x: x['file']):
                    print(f"  - {item['file']}")
                    if 'module' in item:
                        print(f"    (module: {item['module']})")
        else:
            print("\nNo unused files detected.")

        if args.functions:
            if results['unused_functions']:
                print(f"\n## Potentially Unused Functions/Exports ({len(results['unused_functions'])})")
                print("-" * 40)

                # Group by file
                by_file = defaultdict(list)
                for item in results['unused_functions']:
                    by_file[item['file']].append(item)

                for filepath, items in sorted(by_file.items()):
                    print(f"\n  {filepath}:")
                    for item in sorted(items, key=lambda x: x.get('line', 0)):
                        line_info = f":{item['line']}" if 'line' in item else ""
                        print(f"    - {item['name']}{line_info}")
            else:
                print("\nNo unused functions detected.")

        print("\n" + "=" * 60)
        print(f"Summary: {results['summary']['unused_files']} unused files, "
              f"{results['summary']['unused_functions']} unused functions")
        print("=" * 60)

        if results['unused_files'] or results['unused_functions']:
            print("\nNote: Review these results carefully. Some items may be:")
            print("  - Entry points called externally")
            print("  - Dynamically imported/called")
            print("  - Part of a public API")
            print("  - Used via string references")


if __name__ == '__main__':
    main()
