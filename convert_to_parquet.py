import os
import json
import glob
from datasets import Dataset

def parse_md_files(directory, source_name):
    """Walk through directory and yield text content from all .md files."""
    print(f"Scanning {directory} for .md files...")
    count = 0
    # Walk directory recursively
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if text and len(text.strip()) > 0:
                            yield {"text": text, "source": source_name}
                            count += 1
                            if count % 10000 == 0:
                                print(f"  Processed {count} files...")
                except Exception:
                    pass
    print(f"Finished scanning {source_name}. Total files read: {count}")

def parse_social_jsonl_files(directory):
    """Walk through social directory and parse .jsonl files line by line."""
    print(f"Scanning {directory} for social .jsonl files...")
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            
                            # Extract title and body_text
                            title = data.get("title", "")
                            text = data.get("body_text", data.get("text", ""))
                            
                            # Combine title and body if both exist
                            if title and text:
                                full_text = f"{title}\n{text}"
                            elif text:
                                full_text = text
                            else:
                                full_text = title
                                
                            if full_text and len(full_text.strip()) > 0:
                                yield {"text": full_text, "source": "social"}
                                count += 1
                                if count % 20000 == 0:
                                    print(f"  Processed {count} social records...")
                except Exception:
                    pass
    print(f"Finished scanning social. Total records read: {count}")

def main():
    # External SSD source paths
    src_base = "/Volumes/Extreme SSD/Project/ThaiLLMRepo"
    
    # Local destination path
    dest_base = "/Users/howlinglight/foundation-f1/data/ThaiLLMRepo_parquet"
    os.makedirs(dest_base, exist_ok=True)
    
    print("=== Starting Direct Conversion of ThaiLLMRepo (from external SSD) to Local Parquet ===")
    
    # 1. Convert TOR-Law (.md files)
    law_src = os.path.join(src_base, "TOR-Law")
    law_dst = os.path.join(dest_base, "law.parquet")
    if os.path.exists(law_src):
        print("\nProcessing TOR-Law (Markdown) from external SSD...")
        law_dataset = Dataset.from_generator(lambda: parse_md_files(law_src, "TOR-Law"))
        print(f"Saving law data to parquet ({law_dst})...")
        law_dataset.to_parquet(law_dst)
        print("TOR-Law conversion completed!")
    else:
        print(f"Error: TOR-Law directory not found at {law_src}")

    # 2. Convert social (.jsonl files)
    social_src = os.path.join(src_base, "social")
    social_dst = os.path.join(dest_base, "social.parquet")
    if os.path.exists(social_src):
        print("\nProcessing social data (JSONL) from external SSD...")
        social_dataset = Dataset.from_generator(lambda: parse_social_jsonl_files(social_src))
        print(f"Saving social data to parquet ({social_dst})...")
        social_dataset.to_parquet(social_dst)
        print("Social data conversion completed!")
    else:
        print(f"Error: social directory not found at {social_src}")

    # 3. Convert Nation News JSONL
    nation_src = os.path.join(src_base, "nation_data", "reformatted_nation-dolma-v1.jsonl")
    nation_dst = os.path.join(dest_base, "nation.parquet")
    if os.path.exists(nation_src):
        print(f"\nProcessing Nation News JSONL ({nation_src}) from external SSD...")
        # Load JSONL using HF Dataset directly for fast multi-threaded reading
        nation_dataset = Dataset.from_json(nation_src)
        
        # Add metadata source
        def add_metadata(example):
            example["source"] = "nation_data"
            return example
        print("Adding metadata and saving to parquet...")
        nation_dataset = nation_dataset.map(add_metadata, desc="Adding metadata")
        
        # Keep only text and source columns for schema compatibility
        columns_to_keep = ["text", "source"]
        columns_to_remove = [col for col in nation_dataset.column_names if col not in columns_to_keep]
        nation_dataset = nation_dataset.remove_columns(columns_to_remove)
        nation_dataset.to_parquet(nation_dst)
        print("Nation News conversion completed!")
    else:
        print(f"Error: Nation News JSONL not found at {nation_src}")

    print("\n=== All ThaiLLMRepo datasets successfully converted to Local Parquet! ===")
    print(f"Output files stored in: {dest_base}")

if __name__ == "__main__":
    main()
