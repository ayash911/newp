import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="codebase_rag")
data = collection.get(include=["documents", "metadatas"])

for i in range(len(data["ids"])):
    print(f"ID: {data['ids'][i]}")
    print(f"Type: {data['metadatas'][i].get('type', 'Unknown')}")
    print(f"File: {data['metadatas'][i]['file_path']}")
    
    if "original_name" in data['metadatas'][i]:
        print(f"Original Name: {data['metadatas'][i]['original_name']}")
        print(f"Lines: {data['metadatas'][i]['start_line']}-{data['metadatas'][i]['end_line']}")
        connections = json.loads(data['metadatas'][i].get('connections', '[]'))
        print(f"Call Connections: {connections}")
        
    print(f"Content Summary:\n{data['documents'][i]}")
    print("-" * 50)