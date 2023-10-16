import json

def load_jsonl(file_path: Text) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(file_path, data):
    with open(file_path, 'w') as pf:
        for item in data:
            obj = json.dumps(item, ensure_ascii=False)
            pf.write(obj + '\n')
