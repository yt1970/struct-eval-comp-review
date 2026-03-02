import json
with open('data/public_150.json', 'r') as f:
    data = json.load(f)

print(f"Total tasks: {len(data)}")
missing = []
for t in data:
    query = t.get('query', '')
    # CSV, XML, TOML, YAMLなどは通常 <code> に囲まれている
    if '<code>' not in query and '```' not in query:
        missing.append(f"{t['task_id']} ({t['task_name']})")

print(f"Tasks lacking data blocks: {len(missing)}")
for m in missing:
    print(f"  - {m}")
