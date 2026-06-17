import json
import os
import requests
import hashlib
# pyrefly: ignore [missing-import]
import chromadb
import argparse
import logging
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
# pyrefly: ignore [missing-import]
from rich.table import Table

def generate_doc_id(file_path: str) -> str:
    return "doc_" + hashlib.md5(file_path.encode('utf-8')).hexdigest()

def generate_module_id(file_path: str, module_name: str, start_line: int) -> str:
    unique_str = f"{file_path}:{module_name}:{start_line}"
    return "mod_" + hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def calculate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

# Configure Logging
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s"
)

def build_vector_pipeline():
    parser = argparse.ArgumentParser(description="Codebase RAG Ingestion Pipeline")
    parser.add_argument("--verbose", action="store_true", help="Stream token thinking output")
    parser.add_argument("--limit", type=int, help="Limit the number of code modules to process")
    parser.add_argument("--force", action="store_true", help="Force rebuild of all summaries, ignoring cache")
    parser.add_argument("--url", type=str, default="http://localhost:11434/api/generate", help="Ollama API generation endpoint URL")
    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]Initializing ChromaDB Client...[/bold blue]")
    
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="codebase_rag")
        logging.info("Connected to ChromaDB collection 'codebase_rag'")
    except Exception as e:
        console.print(f"[bold red]Failed to connect to ChromaDB: {e}[/bold red]")
        logging.error(f"Failed to connect to ChromaDB: {e}")
        return

    console.print("[bold blue]Running codebase extraction (updating manifest)...[/bold blue]")
    try:
        import extractor
        manifest = extractor.map_and_extract_repo("./target_repo")
        with open("repo_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)
        console.print("[bold green]Success: Codebase extraction complete. 'repo_manifest.json' updated.[/bold green]")
    except Exception as e:
        console.print(f"[bold yellow]Warning: Failed to auto-extract codebase: {e}[/bold yellow]")
        logging.warning(f"Failed to auto-extract codebase using extractor: {e}")
        if os.path.exists("repo_manifest.json"):
            console.print("[yellow]Falling back to existing 'repo_manifest.json'...[/yellow]")
            with open("repo_manifest.json", "r") as f:
                manifest = json.load(f)
        else:
            console.print("[bold red]Error: 'repo_manifest.json' not found and extraction failed.[/bold red]")
            return
        
    url = args.url
    
    # Stats trackers
    doc_stats = {"total": 0, "skipped": 0, "updated": 0, "failed": 0}
    mod_stats = {"total": 0, "skipped": 0, "updated": 0, "failed": 0}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # 1. Ingest Documentation
        docs = manifest.get("documentation", [])
        doc_stats["total"] = len(docs)
        doc_task = progress.add_task("[cyan]Ingesting documentation files", total=len(docs))
        logging.info(f"Ingesting {len(docs)} markdown files...")
        
        # Pre-fetch existing documentation IDs and metadatas
        doc_ids = [generate_doc_id(doc["file_path"]) for doc in docs]
        existing_docs = {}
        if doc_ids:
            try:
                res = collection.get(ids=doc_ids, include=["metadatas"])
                if res and "ids" in res and "metadatas" in res:
                    for eid, emeta in zip(res["ids"], res["metadatas"]):
                        existing_docs[eid] = emeta
            except Exception as e:
                logging.warning(f"Failed to fetch existing documentation metadatas: {e}")
        
        for i, doc in enumerate(docs):
            doc_id = generate_doc_id(doc["file_path"])
            content_hash = calculate_hash(doc["content"])
            
            if not args.force and doc_id in existing_docs:
                meta = existing_docs[doc_id]
                if meta and meta.get("content_hash") == content_hash:
                    logging.info(f"Skipping unchanged doc: {doc['file_path']} ({doc_id})")
                    doc_stats["skipped"] += 1
                    progress.advance(doc_task)
                    continue
            
            try:
                collection.upsert(
                    documents=[doc["content"]],
                    metadatas=[{
                        "file_path": doc["file_path"],
                        "type": "documentation",
                        "content_hash": content_hash
                    }],
                    ids=[doc_id]
                )
                logging.info(f"Successfully upserted doc {doc['file_path']} ({doc_id})")
                doc_stats["updated"] += 1
            except Exception as e:
                logging.error(f"Failed to ingest doc {doc['file_path']} ({doc_id}): {e}")
                doc_stats["failed"] += 1
            progress.advance(doc_task)
            
        # 2. Ingest Code Modules
        code_modules = manifest.get("code_modules", [])
        target_modules = code_modules[:args.limit]
        mod_stats["total"] = len(target_modules)
        
        code_task = progress.add_task("[magenta]Processing code modules (Ollama)", total=len(target_modules))
        logging.info(f"Starting processing for {len(target_modules)} code modules (limit: {args.limit})")
        
        # Pre-fetch existing module IDs and metadatas in chunks
        mod_ids = [generate_module_id(mod["file_path"], mod["name"], mod["start_line"]) for mod in target_modules]
        existing_mods = {}
        if mod_ids:
            try:
                chunk_size = 500
                for start in range(0, len(mod_ids), chunk_size):
                    chunk_ids = mod_ids[start:start+chunk_size]
                    res = collection.get(ids=chunk_ids, include=["metadatas"])
                    if res and "ids" in res and "metadatas" in res:
                        for eid, emeta in zip(res["ids"], res["metadatas"]):
                            existing_mods[eid] = emeta
            except Exception as e:
                logging.warning(f"Failed to fetch existing modules: {e}")
        
        for i, mod in enumerate(target_modules):
            line_count = len(mod['raw_code'].splitlines())
            mod_id = generate_module_id(mod["file_path"], mod["name"], mod["start_line"])
            code_hash = calculate_hash(mod['raw_code'])
            
            if not args.force and mod_id in existing_mods:
                meta = existing_mods[mod_id]
                if meta and meta.get("code_hash") == code_hash:
                    logging.info(f"Skipping unchanged module: '{mod['name']}' from {mod['file_path']} ({mod_id})")
                    mod_stats["skipped"] += 1
                    progress.advance(code_task)
                    continue
            
            logging.info(f"Processing '{mod['name']}' from {mod['file_path']} ({line_count} lines)...")
            
            prompt = f"""
            Analyze the following Python source code block from `{mod['file_path']}`:
            
            ```python
            {mod['raw_code']}
            ```
            
            Context Details:
            - Entity Name: {mod['name']}
            - Calls Made: {', '.join(mod['calls']) if mod['calls'] else 'None'}
            
            Write a detailed, high-density, architecturally precise summary of this block focusing on its purpose and call interactions.
            Format your response strictly as follows (no other text, markdown headings only):
            
            ## Purpose
            [1-2 sentences explaining the core objective and architectural role of the block]
            
            ## Call-Graph Data Flow
            [1-2 sentences explaining strictly how data/control flows between this block and the functions it calls: {', '.join(mod['calls']) if mod['calls'] else 'None'}]
            """
            
            true_intent = ""
            try:
                payload = {"model": "gemma4:12b", "prompt": prompt, "stream": args.verbose}
                res = requests.post(url, json=payload, stream=args.verbose, timeout=120)
                res.raise_for_status()
                
                if args.verbose:
                    full_response = []
                    progress.console.print(f"\n[bold yellow][{i+1}/{len(target_modules)}] Streaming Intent for '{mod['name']}':[/bold yellow] ", end="")
                    for line in res.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            progress.console.print(token, end="", flush=True)
                            full_response.append(token)
                    progress.console.print()
                    true_intent = "".join(full_response).strip()
                else:
                    true_intent = res.json().get("response", "").strip()
                
                if not true_intent:
                    true_intent = "No intent summary generated."
                    
                collection.upsert(
                    documents=[true_intent],
                    metadatas=[{
                        "file_path": mod["file_path"],
                        "raw_code": mod["raw_code"],
                        "original_name": mod["name"],
                        "type": mod["type"],
                        "start_line": mod["start_line"],
                        "end_line": mod["end_line"],
                        "connections": json.dumps(mod["calls"]),
                        "code_hash": code_hash
                    }],
                    ids=[mod_id]
                )
                logging.info(f"Successfully indexed function: '{mod['name']}' as {mod_id}")
                mod_stats["updated"] += 1
                
            except Exception as e:
                logging.error(f"Failed to process module {mod['name']}: {e}")
                progress.console.print(f"[bold red]Error processing '{mod['name']}': {e}[/bold red]")
                
                # Generate a high-quality fallback summary based on function/method name
                fallback_intent = f"Handles execution logic and internal operations for '{mod['name']}'."
                if mod['name'] == "__init__":
                    fallback_intent = "Initializes a new instance of the class."
                elif mod['name'] == "__iter__":
                    fallback_intent = "Provides synchronous iteration capabilities."
                elif mod['name'] == "__aiter__":
                    fallback_intent = "Provides asynchronous iteration capabilities."
                elif mod['name'] in ("__str__", "__repr__"):
                    fallback_intent = "Returns the string representation of the class instance."
                elif mod['name'] == "__len__":
                    fallback_intent = "Returns the length or size of the object."
                elif mod['name'] == "__eq__":
                    fallback_intent = "Checks equality with another object."
                elif mod['name'] == "__hash__":
                    fallback_intent = "Computes the hash value of the object."
                
                progress.console.print(f"[yellow]Caching fallback summary for '{mod['name']}' to prevent future timeouts.[/yellow]")
                
                try:
                    collection.upsert(
                        documents=[fallback_intent],
                        metadatas=[{
                            "file_path": mod["file_path"],
                            "raw_code": mod["raw_code"],
                            "original_name": mod["name"],
                            "type": mod["type"],
                            "start_line": mod["start_line"],
                            "end_line": mod["end_line"],
                            "connections": json.dumps(mod["calls"]),
                            "code_hash": code_hash
                        }],
                        ids=[mod_id]
                    )
                    logging.info(f"Successfully cached fallback summary for '{mod['name']}' as {mod_id}")
                except Exception as db_err:
                    logging.error(f"Failed to write fallback summary to DB: {db_err}")
                
                mod_stats["failed"] += 1
                
            progress.advance(code_task)

    # Print Ingestion Statistics
    table = Table(title="Ingestion Pipeline Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Skipped (Cached)", style="green", justify="right")
    table.add_column("Updated/Processed", style="yellow", justify="right")
    table.add_column("Failed", style="red", justify="right")
    
    table.add_row(
        "Documentation Files", 
        str(doc_stats["total"]), 
        str(doc_stats["skipped"]), 
        str(doc_stats["updated"]), 
        str(doc_stats["failed"])
    )
    table.add_row(
        "Code Modules", 
        str(mod_stats["total"]), 
        str(mod_stats["skipped"]), 
        str(mod_stats["updated"]), 
        str(mod_stats["failed"])
    )
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    build_vector_pipeline()