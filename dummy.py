from tqdm import tqdm
from glob import glob
from saturn.utils.io import load_jsonl, write_jsonl

data = load_jsonl(file_path="data_test.jsonl")
for id in tqdm(range(len(data))):
    data[id]['document_negative'] = " ".join(data[id]['document_negative']).strip()

print(data[:3])
write_jsonl(file_path="data/train/data_test.jsonl", data=data)