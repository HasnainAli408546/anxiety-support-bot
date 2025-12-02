import json
import os
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
import os


# Mapping from each supported 'type' to the ChromaDB collection name
COLLECTION_MAP = {
    "technique": "techniques",
    "tool": "techniques",
    "exercise": "techniques",
    "education": "education",
    "psychoeducation": "education",
    "theory": "education",
    "reflection": "education",  # or make your own collection e.g. "reflections"
    "reassurance": "reassurance",
    "support": "reassurance",   # or make your own "support"
    "affirmation": "reassurance",  # or make your own "affirmations"
    "reminder": "reassurance",     # or "reminders"
    "resource": "resources",
    "directory": "resources",
    "immediate": "resources",
    "reference": "resources",   # or "references"
    # Add more custom mappings as you discover content types!
}



def flatten_metadata(metadata):
    """
    Convert any list-type metadata values to comma-separated strings.
    Keeps str, int, float, bool, None unchanged.
    """
    return {
        k: (", ".join(v) if isinstance(v, list) else v)
        for k, v in metadata.items()
    }

def ingest_jsonl(jsonl_path, collections, encoder):
    """
    Ingest each chunk in the .jsonl file into the correct ChromaDB collection by type,
    flattening any list in metadata to a string.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    added = {k: 0 for k in COLLECTION_MAP.values()}
    for idx, line in enumerate(tqdm(lines, desc=f"Processing {os.path.basename(jsonl_path)}")):
        doc = json.loads(line)
        content_type = doc.get('type', None)
        collection_name = COLLECTION_MAP.get(content_type, None)
        content = doc.get('content')
        metadata = flatten_metadata(doc.copy())  # Flatten all metadata fields
        if not content or not collection_name or collection_name not in collections:
            print(f"Skipping line {idx+1}: missing content or unknown type '{content_type}'")
            continue

        embedding = encoder.encode(content).tolist()
        content_id = doc.get('id', f"{os.path.basename(jsonl_path).replace('.jsonl','')}_{idx+1}")

        collections[collection_name].add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[content_id]
        )
        added[collection_name] += 1

    for name, count in added.items():
        if count:
            print(f"✅ Ingested {count} items into '{name}' from {jsonl_path}")

def main():
    DB_PATH = os.getenv("CHROMADB_PATH", "therapeutic_kb")
    JSONL_DIR = "."  # Modify as needed to match your data folder
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Initializing ChromaDB at path:", DB_PATH)

    client = chromadb.PersistentClient(path=DB_PATH)
    # Ensure all needed collections exist
    collections = {name: client.get_or_create_collection(name) for name in COLLECTION_MAP.values()}

    jsonl_files = [fname for fname in os.listdir(JSONL_DIR) if fname.endswith(".jsonl")]
    if jsonl_files:
        for fname in jsonl_files:
            fpath = os.path.join(JSONL_DIR, fname)
            print(f"\n➡️ Ingesting: {fname}")
            ingest_jsonl(fpath, collections, encoder)
        print("\n🎉 All ready! You can now retrieve chunks from ChromaDB by semantic search.")
    else:
        print(f"⚠️ No JSONL files found in '{JSONL_DIR}' for ingestion.")

if __name__ == "__main__":
    main()
