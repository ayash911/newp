import hashlib
import json
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.table import Table

def generate_doc_id(file_path: str) -> str:
    return "doc_" + hashlib.md5(file_path.encode('utf-8')).hexdigest()

def generate_module_id(file_path: str, module_name: str, start_line: int) -> str:
    unique_str = f"{file_path}:{module_name}:{start_line}"
    return "mod_" + hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def calculate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()

def run_migration():
    console = Console()
    console.print("[bold blue]Starting ChromaDB Database Migration...[/bold blue]")
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="codebase_rag")
    except Exception as e:
        console.print(f"[bold red]Failed to connect to ChromaDB: {e}[/bold red]")
        return

    # Fetch all records
    console.print("[yellow]Fetching all existing records from ChromaDB...[/yellow]")
    data = collection.get(include=["documents", "metadatas"])
    total_records = len(data["ids"])
    console.print(f"Found [cyan]{total_records}[/cyan] total records in the collection.")
    
    ids_to_delete = []
    migrated_records = {}
    
    docs_migrated = 0
    mods_migrated = 0
    already_migrated = 0
    duplicates_removed = 0
    
    for doc_id, doc_text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        parts = doc_id.split("_")
        is_old_format = len(parts) == 2 and parts[1].isdigit()
        is_doc = meta.get("type") == "documentation" or doc_id.startswith("doc_")
        
        if not is_old_format:
            # Already in new format, ensure hash exists
            already_migrated += 1
            if is_doc:
                if "content_hash" not in meta:
                    meta["content_hash"] = calculate_hash(doc_text)
                    collection.update(ids=[doc_id], metadatas=[meta])
            else:
                if "code_hash" not in meta:
                    raw_code = meta.get("raw_code", "")
                    meta["code_hash"] = calculate_hash(raw_code)
                    collection.update(ids=[doc_id], metadatas=[meta])
            continue
            
        # Migrate old format
        if is_doc:
            file_path = meta.get("file_path", f"unknown_{parts[1]}")
            new_id = generate_doc_id(file_path)
            
            if new_id in migrated_records:
                duplicates_removed += 1
                ids_to_delete.append(doc_id)
                continue
                
            # Compute hash
            meta["content_hash"] = calculate_hash(doc_text)
            migrated_records[new_id] = (doc_text, meta)
            ids_to_delete.append(doc_id)
            docs_migrated += 1
        else:
            file_path = meta.get("file_path", "Unknown")
            module_name = meta.get("original_name", f"unknown_{parts[1]}")
            start_line = meta.get("start_line", 0)
            new_id = generate_module_id(file_path, module_name, start_line)
            
            if new_id in migrated_records:
                duplicates_removed += 1
                ids_to_delete.append(doc_id)
                continue
                
            # Compute hash
            raw_code = meta.get("raw_code", "")
            meta["code_hash"] = calculate_hash(raw_code)
            migrated_records[new_id] = (doc_text, meta)
            ids_to_delete.append(doc_id)
            mods_migrated += 1
            
    # Perform upserts and deletions
    if migrated_records:
        new_ids = list(migrated_records.keys())
        new_documents = [migrated_records[nid][0] for nid in new_ids]
        new_metadatas = [migrated_records[nid][1] for nid in new_ids]
        
        console.print(f"[yellow]Migrating {len(new_ids)} unique records to the new hashed/deterministic format...[/yellow]")
        collection.upsert(ids=new_ids, documents=new_documents, metadatas=new_metadatas)
        
    if ids_to_delete:
        console.print(f"[yellow]Cleaning up {len(ids_to_delete)} old index-based IDs...[/yellow]")
        collection.delete(ids=ids_to_delete)
        
    # Build summary table
    table = Table(title="Database Migration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Status / Details", style="yellow")
    
    table.add_row("Total DB Records Scanned", str(total_records), "Total pre-existing items")
    table.add_row("Documentation Migrated", str(docs_migrated), "Converted doc_X -> doc_<hash>")
    table.add_row("Code Modules Migrated", str(mods_migrated), "Converted mod_X -> mod_<hash>")
    table.add_row("Duplicate Records Removed", str(duplicates_removed), "Removed redundant duplicates")
    table.add_row("Already in New Format", str(already_migrated), "Unchanged / already deterministic")
    table.add_row("Old Records Cleaned Up", str(len(ids_to_delete)), "Successfully deleted from DB")
    
    console.print(table)
    console.print("[bold green]Success: Database migration complete. Your 8+ hour generated summaries are preserved![/bold green]")

if __name__ == "__main__":
    run_migration()
