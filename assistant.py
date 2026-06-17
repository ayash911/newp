import os
import re
import json
import logging
import requests
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from rich.console import Console
# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.table import Table
# pyrefly: ignore [missing-import]
from rich.markdown import Markdown
# pyrefly: ignore [missing-import]
from rich.syntax import Syntax

# Configure Logging
logging.basicConfig(
    filename="rag_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s"
)

MAX_TOKENS_BUDGET = 8000

def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string based on character count (1 token ≈ 4 characters)."""
    return len(text) // 4

def extract_and_validate_json_trace(text: str):
    """
    Finds the markdown JSON code block under ## 1. EXECUTION LOGGING TRACE,
    parses it, and returns the parsed dict or validation error.
    """
    # Regex to find ## 1. EXECUTION LOGGING TRACE followed by a json code block
    pattern = r"## 1\.\s*EXECUTION LOGGING TRACE\s*\n\s*```(?:json)?\s*\n(.*?)\n\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        # Fallback to search any ```json ... ``` or ``` ... ``` block
        fallback_pattern = r"```(?!json\b)(?:json)?\s*\n(.*?)\n\s*```"
        # Wait, simple: r"```(?:json)?\s*\n(.*?)\n\s*```"
        fallback_pattern = r"```(?:json)?\s*\n(.*?)\n\s*```"
        match = re.search(fallback_pattern, text, re.DOTALL | re.IGNORECASE)
        
    if match:
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            logging.info(f"Response execution trace validated successfully: {json.dumps(parsed)}")
            return parsed, None
        except json.JSONDecodeError as e:
            logging.warning(f"Response execution trace block found but failed to parse: {e}. Raw content: {json_str}")
            return None, f"JSON Decode Error: {e}"
    else:
        logging.warning("No EXECUTION LOGGING TRACE JSON code block found in response.")
        return None, "No JSON code block found matching target schema"

def extract_markdown_content(text: str) -> str:
    """Extracts everything from ## 2. ARCHITECTURAL ANALYSIS & SOLUTION onwards."""
    match = re.search(r"(## 2\..*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def build_context(results, max_tokens=6000):
    """
    Iterates through retrieved results, filters by token budget,
    and constructs the RAG context string.
    """
    context_list = []
    current_tokens = 0
    used_sources = []
    
    if not results or not results['documents'] or not results['documents'][0]:
        return "", []

    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results.get('distances', [[]])[0] if 'distances' in results else [0.0] * len(docs)
    
    for doc, meta, dist in zip(docs, metadatas, distances):
        file_path = meta.get('file_path', 'Unknown')
        dtype = meta.get('type', 'documentation')
        
        if dtype == 'documentation':
            formatted_chunk = f"--- Documentation ({file_path}) [Distance: {dist:.4f}] ---\n{doc}"
        else:
            raw_code = meta.get('raw_code', '')
            connections = meta.get('connections', '[]')
            formatted_chunk = (
                f"--- Code Module ({file_path}) [Distance: {dist:.4f}] ---\n"
                f"Type: {meta.get('type')}\n"
                f"Function/Class: {meta.get('original_name')}\n"
                f"Lines: {meta.get('start_line')}-{meta.get('end_line')}\n"
                f"Intent Summary: {doc}\n"
                f"Connected Calls: {connections}\n"
                f"Code:\n{raw_code}"
            )
            
        chunk_tokens = estimate_tokens(formatted_chunk)
        if current_tokens + chunk_tokens <= max_tokens:
            context_list.append(formatted_chunk)
            current_tokens += chunk_tokens
            used_sources.append({
                "file_path": file_path,
                "type": dtype,
                "name": meta.get('original_name', 'Doc') if dtype != 'documentation' else 'Doc',
                "line_range": f"{meta.get('start_line', '')}-{meta.get('end_line', '')}" if dtype != 'documentation' else "N/A",
                "distance": dist
            })
        else:
            logging.info(f"Context budget reached. Skipping chunk from {file_path} (estimated {chunk_tokens} tokens)")
            
    return "\n\n".join(context_list), used_sources

def run_assistant():
    console = Console()
    
    # Render welcome banner
    console.print(Panel(
        "[bold green]GraphRAG Codebase Architect AI Agent[/bold green]\n"
        "[dim]Powered by ChromaDB, Gemma 4 (12B), and Ollama[/dim]\n\n"
        "Ask queries about your repository architectures and connections. Type 'exit' to quit.",
        title="[bold blue]Welcome[/bold blue]",
        border_style="blue",
        expand=False
    ))
    
    logging.info("Initializing ChromaDB Client in Assistant stage...")
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="codebase_rag")
        logging.info("Connected to ChromaDB collection 'codebase_rag' successfully")
    except Exception as e:
        console.print(Panel(
            "[bold red]Error: ChromaDB database or collection 'codebase_rag' not found.[/bold red]\n"
            "[yellow]Please verify you ran 'extractor.py' and 'injestion.py' first.[/yellow]",
            title="[bold red]Database Connection Failed[/bold red]",
            border_style="red"
        ))
        logging.error(f"Failed to find ChromaDB collection: {e}")
        return
        
    url = "http://localhost:11434/api/generate"
    
    while True:
        try:
            user_query = console.input("\n[bold green]Query[/bold green] > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold blue]Exiting. Goodbye![/bold blue]")
            break
            
        if not user_query:
            continue
            
        if user_query.lower() in ['exit', 'quit']:
            console.print("[bold blue]Exiting. Goodbye![/bold blue]")
            break
            
        logging.info(f"User Query: {user_query}")
        
        # Retrieve relevant code blocks and markdown documentation
        console.print("[yellow]Retrieving and sorting relevant code blocks & documentation...[/yellow]")
        try:
            # Retrieve a larger pool of results so we can fit them dynamically
            results = collection.query(query_texts=[user_query], n_results=12)
        except Exception as e:
            console.print(f"[bold red]ChromaDB query failed: {e}[/bold red]")
            logging.error(f"ChromaDB query failed: {e}")
            continue
            
        # Build context within token sliding window budget
        context, sources = build_context(results, max_tokens=MAX_TOKENS_BUDGET)
        
        # Show retrieval sources using Table
        if sources:
            table = Table(title=f"Retrieved References (Within {MAX_TOKENS_BUDGET} Token Budget)", show_header=True, header_style="bold magenta")
            table.add_column("Type", style="cyan")
            table.add_column("File Path", style="yellow")
            table.add_column("Entity / Range", style="blue")
            table.add_column("Distance", style="green", justify="right")
            
            for src in sources:
                entity = f"{src['name']} (L:{src['line_range']})" if src['type'] != 'documentation' else "Doc"
                table.add_row(src['type'].capitalize(), src['file_path'], entity, f"{src['distance']:.4f}")
            console.print(table)
        else:
            console.print("[bold red]No references found in database.[/bold red]")
            logging.warning("Query returned zero context references.")
            continue
            
        # Setup Master Prompt
        prompt = f"""# ROLE AND CORE OBJECTIVE
You are an advanced Senior Codebase Architect and a deterministic GraphRAG Reasoning Engine. Your core objective is to analyze complex repository structures, track multi-file execution paths, and provide high-fidelity solutions strictly based on the retrieved context provided to you (Markdown documents, raw code modules, AST call graphs, and line numbers).

---

# CONTEXT VALIDATION & COGNITIVE BOUNDARIES
1. STRICT TRUTH TO CONTEXT: Rely *only* on the provided context chunks. Do not hallucinate external architectural patterns, missing functions, or undocumented dependencies.
2. AST TRUTH OVER FILENAMES: Disregard vague, misleading, or legacy function/class names. Analyze the raw abstract syntax tree logic, parameters, and return types to evaluate true functional intent.
3. CONTEXT DEFICIT PROTOCOL: If the retrieved data is insufficient to completely resolve the query, or if an execution line path breaks midway due to missing files, you must halt execution and explicitly generate a "Structural Context Deficit Log" identifying exactly what code link or file is missing.

---

# STEP-BY-STEP REASONING PIPELINE (CHAIN OF THOUGHT)
You must systematically execute these analytical steps before formulating your final architectural response:
1. Parse Request: Deconstruct the user's inquiry into its core technical requirements and target assets.
2. Trace Execution Graph: Inspect the `Connections` array in the metadata. Map out the recursive caller-to-callee dependencies related to the target modules.
3. Cross-Reference Documentation: Correlate the mapped code entities with any extracted `.md` architectural guidelines or conceptual overviews.
4. Detect File Boundaries: Note exact file paths, line ranges (`start_line` to `end_line`), and types (`FunctionDef`, `ClassDef`).
5. Synthesize Impact: Determine how the targeted code blocks intersect and impact the wider execution loop.

---

# MANDATORY LOGGING & OUTPUT FORMAT
Your entire response must strictly follow this structural markdown schema. No free-form prose is permitted outside these defined blocks:

## 1. EXECUTION LOGGING TRACE
```json
{{
  "status": "PROCESSING_SUCCESSFUL | CONTEXT_DEFICIT_DETECTED",
  "analyzed_components": [
    {{
      "file_path": "string",
      "type": "FunctionDef | ClassDef | Documentation",
      "entity_name": "string",
      "line_range": "start-end"
    }}
  ],
  "traversed_call_links": [
    "caller_function_name() -> callee_function_name()"
  ],
  "confidence_score": 0.00
}}
```

## 2. ARCHITECTURAL ANALYSIS & SOLUTION

[Provide a step-by-step, technically rigorous breakdown of how the codebase functions regarding the query. Reference exact file paths and line numbers.]

## 3. IMPLEMENTATION / RESOLUTION REFERENCE

[Provide precise code snippets, structural patches, or configuration changes directly derived from the analyzed context logic.]

---

# RETRIEVED CONTEXT
{context}

---

# USER QUERY
Query: {user_query}

# DETERMINISTIC RESPONSE:
"""
        
        # Connect and Stream response from Ollama
        full_response_text = ""
        try:
            with console.status("[yellow]Querying Ollama (gemma4:12b) & analyzing codebase...[/yellow]", spinner="dots") as status:
                res = requests.post(
                    url, 
                    json={
                        "model": "gemma4:12b", 
                        "prompt": prompt, 
                        "stream": True,
                        "options": {
                            "num_ctx": 16384
                        }
                    }, 
                    stream=True, 
                    timeout=600
                )
                res.raise_for_status()
                
                # Stream tokens into buffer
                for line in res.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        token = chunk.get("response", "")
                        full_response_text += token
            
            # Post-processing: Extract and validate JSON execution logging trace
            parsed_trace, validation_error = extract_and_validate_json_trace(full_response_text)
            
            # Output validation state
            if parsed_trace:
                status_val = parsed_trace.get("status", "PROCESSING_SUCCESSFUL")
                trace_color = "green" if status_val == "PROCESSING_SUCCESSFUL" else "yellow"
                console.print(Panel(
                    Syntax(json.dumps(parsed_trace, indent=2), "json", theme="monokai"),
                    title=f"[bold {trace_color}]Execution Logging Trace (Validated - Status: {status_val})[/bold {trace_color}]",
                    border_style=trace_color
                ))
            else:
                logging.warning(f"Validation failed. Raw response was:\n{full_response_text}")
                console.print(Panel(
                    f"[bold red]Response trace failed validation or is missing.[/bold red]\n[yellow]Error: {validation_error}[/yellow]",
                    title="[bold red]Execution Logging Trace (Failed Validation)[/bold red]",
                    border_style="red"
                ))
            
            # Render and print Section 2 & 3 as premium Markdown
            markdown_content = extract_markdown_content(full_response_text)
            console.print(Markdown(markdown_content))
                
        except Exception as e:
            console.print(f"\n[bold red]Error communicating with Ollama: {e}[/bold red]")
            logging.error(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    run_assistant()
