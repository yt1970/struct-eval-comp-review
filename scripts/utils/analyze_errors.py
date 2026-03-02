import json
import re
import pandas as pd
from io import StringIO

PUBLIC_DATA_PATH = "data/public_150.json"
INFERENCE_DATA_PATH = "inference_SFT_0_again（0.75191 ）.json"

def load_data():
    with open(PUBLIC_DATA_PATH, "r") as f:
        public_data = json.load(f)
    # Convert to dict for easy lookup
    public_map = {item["task_id"]: item for item in public_data}
    
    with open(INFERENCE_DATA_PATH, "r") as f:
        inference_data = json.load(f)
    
    return public_map, inference_data

def extract_json(tensor_str):
    # Try to find JSON block
    match = re.search(r"```json\s*(.*?)\s*```", tensor_str, re.DOTALL)
    if match:
        return match.group(1), True # Found markdown
    
    # If no markdown, try to find first { or [ and last } or ]
    match_braces = re.search(r"(\{.*\}|\[.*\])", tensor_str, re.DOTALL)
    if match_braces:
        return match_braces.group(1), False
        
    return tensor_str, False

def extract_required_keys_from_prompt(query):
    # Pattern 1: "The field 'a.b.c'..."
    keys = re.findall(r"The field '([^']+)'", query)
    if keys:
        return set(keys)
    
    # Pattern 2: CSV headers
    if "CSV code" in query:
        # Extract CSV block
        code_match = re.search(r"<code>(.*?)</code>", query, re.DOTALL)
        if code_match:
            try:
                csv_content = code_match.group(1).strip()
                # Read first line as header
                header = csv_content.split('\n')[0]
                cols = [c.strip() for c in header.split(',')]
                # Convert dot notation to expected json structure?
                # Usually CSV to JSON tasks require mapping "a.b" -> nested dict or keeping as is?
                # Based on inference file, "system.star_name" -> {"system": {"star_name": ...}}
                return set(cols)
            except:
                pass
    return set()

def flatten_keys(data, parent_key='', sep='.'):
    keys = set()
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            keys.add(new_key)
            keys.update(flatten_keys(v, new_key, sep=sep))
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            # Assumption: all items in list have same schema, check first one
            # But specific tasks might map rows to list.
            # Let's check keys of the first object
            keys.update(flatten_keys(data[0], parent_key, sep=sep))
            # Also add list index keys? No, usually we care about the schema fields.
    return keys

def match_keys(expected_keys, actual_keys):
    # Logic: "a.b" in expected should match "a.b" in actual flattened keys.
    # However, sometimes CSV "a.b" becomes {"a": {"b": ...}} which flattens to "a.b".
    # Sometimes CSV "a.b" stays "a.b" key.
    
    missing = []
    for k in expected_keys:
        if k not in actual_keys:
            # Try handling array notation like "materials[0].type" -> matches "materials.type" ??
            # Or regex match?
            # "materials[0].type" -> check if "materials" in actual and "type" in materials?
            
            # Simple normalization for array indices in keys: Remove [0] etc.
            k_norm = re.sub(r"\[\d+\]", "", k) # materials.type
            
            # Check if any actual key matches this normalized key
            # Actual keys from flatten don't typically have [0] unless logic adds it. my logic doesn't.
            if k_norm not in actual_keys:
                 missing.append(k)
                 
    return missing

def main():
    public_map, inference_data = load_data()
    
    stats = {
        "total": 0,
        "chatty": 0,    # Contains "Sure", "Here is", etc outside JSON
        "markdown": 0,  # Uses ```json
        "parse_error": 0,
        "missing_keys_count": 0,
        "missing_keys_details": []
    }
    
    print(f"Analyzing {len(inference_data)} inference results...")
    
    for item in inference_data:
        stats["total"] += 1
        task_id = item["task_id"]
        generation = item["generation"]
        
        # Check Chatty
        # Heuristic: if it doesn't start with { or [ or ```
        if not generation.strip().startswith(("{", "[", "```")):
            stats["chatty"] += 1
            
        # Check Markdown
        json_str, has_markdown = extract_json(generation)
        if has_markdown:
            stats["markdown"] += 1
            
        # Parse JSON
        try:
            generated_json = json.loads(json_str)
        except json.JSONDecodeError:
            stats["parse_error"] += 1
            continue
            
        # Check Schema
        if task_id in public_map:
            query = public_map[task_id]["query"]
            required_keys = extract_required_keys_from_prompt(query)
            
            if required_keys:
                actual_keys = flatten_keys(generated_json)
                missing = match_keys(required_keys, actual_keys)
                
                if missing:
                    stats["missing_keys_count"] += 1
                    stats["missing_keys_details"].append({
                        "task_id": task_id,
                        "missing": missing,
                        "required_count": len(required_keys),
                        "missing_count": len(missing)
                    })

    print("-" * 30)
    print("📊 Analysis Results")
    print("-" * 30)
    print(f"Total Tasks: {stats['total']}")
    print(f"Conversational/Filler Text Detected: {stats['chatty']} ({stats['chatty']/stats['total']:.2%})")
    print(f"Markdown Blocks Used: {stats['markdown']} ({stats['markdown']/stats['total']:.2%})")
    print(f"JSON Parse Errors: {stats['parse_error']} ({stats['parse_error']/stats['total']:.2%})")
    print(f"Possible Schema Violations (Missing Keys): {stats['missing_keys_count']} ({stats['missing_keys_count']/stats['total']:.2%})")
    
    if stats["missing_keys_details"]:
        print("\n🔍 Top Missing Keys Patterns (First 5 examples):")
        for detail in stats["missing_keys_details"][:5]:
            print(f"Task {detail['task_id']}: Missing {detail['missing_count']}/{detail['required_count']} keys -> {detail['missing']}")

if __name__ == "__main__":
    main()
