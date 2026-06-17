import os
import ast
import json
import logging
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.table import Table

# Configure Logging
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s"
)

def map_and_extract_repo(repo_path):
    logging.info(f"Starting codebase extraction for path: {repo_path}")
    repo_manifest = {
        "documentation": [],
        "code_modules": []
    }
    
    if not os.path.exists(repo_path):
        logging.error(f"Target repository path does not exist: {repo_path}")
        return repo_manifest

    for root, _, files in os.walk(repo_path):
        # Split root path to inspect segments
        root_segments = root.split(os.sep)
        
        # Skip directories like .git, .github, tests, and venv (common virtualenv directory)
        # Check if any segment starts with '.' (and is not '.' or '..') or is 'tests' or 'venv'
        if any((seg.startswith('.') and seg not in ('.', '..')) or seg in ('tests', 'venv') for seg in root_segments):
            logging.debug(f"Skipping walking directory: {root}")
            continue

        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_path)
            
            # Additional path-based exclusion (just to be safe)
            path_segments = relative_path.split(os.sep)
            if any((seg.startswith('.') and seg not in ('.', '..')) or seg in ('tests', 'venv') for seg in path_segments):
                logging.debug(f"Skipping file: {relative_path}")
                continue

            if file.endswith('.md'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        repo_manifest["documentation"].append({
                            "file_path": relative_path,
                            "content": content
                        })
                        logging.info(f"Mapped documentation: {relative_path} ({len(content)} bytes)")
                except Exception as e:
                    logging.error(f"Failed to read markdown file {relative_path}: {e}")
                    
            elif file.endswith('.py'):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code_content = f.read()
                except Exception as e:
                    logging.error(f"Failed to read python file {relative_path}: {e}")
                    continue

                try:
                    tree = ast.parse(code_content, filename=file_path)
                except SyntaxError as e:
                    logging.warning(f"Syntax error parsing {relative_path}: {e}")
                    continue
                
                code_lines = code_content.splitlines()
                num_lines = len(code_lines)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        start_line = max(1, node.lineno)
                        end_line = min(num_lines, getattr(node, 'end_lineno', start_line + 5))
                        
                        raw_lines = code_lines[start_line-1:end_line]
                        
                        called_functions = []
                        for child_node in ast.walk(node):
                            if isinstance(child_node, ast.Call) and hasattr(child_node.func, 'id'):
                                called_functions.append(child_node.func.id)
                        
                        repo_manifest["code_modules"].append({
                            "file_path": relative_path,
                            "type": type(node).__name__,
                            "name": node.name,
                            "start_line": start_line,
                            "end_line": end_line,
                            "calls": called_functions,
                            "raw_code": "\n".join(raw_lines)
                        })
                        logging.info(
                            f"Extracted module: {relative_path} -> {type(node).__name__} '{node.name}' "
                            f"(lines {start_line}-{end_line})"
                        )
                            
            else:
                # Exclude lock files or typical compiled/binary files
                ignored_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.ico', '.pyc', '.pyd', '.pyo', '.db', '.sqlite', '.woff', '.woff2', '.ttf', '.eot')
                if file.endswith(ignored_extensions) or file in ('poetry.lock', 'package-lock.json', 'yarn.lock'):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                        content = f.read()
                        if '\x00' in content:  # Binary file check
                            continue
                        
                        # Add as a whole-file module
                        line_count = len(content.splitlines())
                        
                        # Determine file type category
                        file_type = "File"
                        if file.endswith(('.sh', '.bat', '.ps1')) or 'scripts' in root_segments:
                            file_type = "Script"
                        elif file.endswith(('.toml', '.yml', '.yaml', '.json', '.ini', '.cfg')) or file == 'requirements.txt':
                            file_type = "Config"
                            
                        repo_manifest["code_modules"].append({
                            "file_path": relative_path,
                            "type": file_type,
                            "name": file,
                            "start_line": 1,
                            "end_line": line_count,
                            "calls": [],
                            "raw_code": content
                        })
                        logging.info(
                            f"Extracted other text file: {relative_path} as {file_type} "
                            f"(lines 1-{line_count})"
                        )
                except UnicodeDecodeError:
                    # Skip binary or non-UTF8 files gracefully
                    continue
                except Exception as e:
                    logging.error(f"Failed to read other file {relative_path}: {e}")
    logging.info(
        f"Extraction finished. Mapped {len(repo_manifest['documentation'])} docs and "
        f"{len(repo_manifest['code_modules'])} code modules."
    )
    return repo_manifest

if __name__ == "__main__":
    console = Console()
    console.print("[bold blue]Starting codebase extraction pipeline...[/bold blue]")
    
    manifest = map_and_extract_repo("./target_repo")
    
    # Save manifest
    with open("repo_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    console.print("[bold green]Success: Generated 'repo_manifest.json'.[/bold green]")
    
    # Build summary table
    table = Table(title="Extraction Process Completed", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=25)
    table.add_column("Count", style="green", justify="right", width=10)
    table.add_column("Description", style="yellow")
    
    table.add_row("Documentation (.md)", str(len(manifest['documentation'])), "Total markdown documents mapped")
    table.add_row("Code Modules (.py)", str(len(manifest['code_modules'])), "Total Python classes and functions extracted")
    
    console.print(table)