import re
import json
import torch
import chromadb
from pyvi.ViTokenizer import tokenize
from transformers import AutoModel, AutoTokenizer

vowel = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]

vowel_to_idx = {}
for i in range(len(vowel)):
    for j in range(len(vowel[i]) - 1):
        vowel_to_idx[vowel[i][j]] = (i, j)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def preprocessing(text):
    text = text.replace('tui', 'tôi')
    text = text.replace('lệ phí', 'học phí')
    punctuation = "[\".?!#$%&'()*+,:;<=>@[\\]^`{|}~]“”"
    text = text.translate(str.maketrans(' ', ' ', punctuation))
    text = text.lower()

    # remove duplicate space
    text = re.sub(r"[\s]+", " ", text)
    return text


def load_data(file_path, col_meta):
    data = load_jsonl(file_path)
    results = []
    for sentence in data:
        results.append(sentence[col_meta])

    return results


class Chromadb:
    def __init__(self):
        self.questions = load_data(file_path="/u01/os_callbot/hoaf13/namnp/QA_PTIT/data/data.jsonl", col_meta='query')
        self.documents = load_data(file_path="/u01/os_callbot/hoaf13/namnp/QA_PTIT/data/data.jsonl",
                                   col_meta='document')

        self.tokenizer = AutoTokenizer.from_pretrained("/u01/os_callbot/hoaf13/namnp/QA_PTIT/models")
        self.model = AutoModel.from_pretrained("/u01/os_callbot/hoaf13/namnp/QA_PTIT/models")

        self.client = chromadb.Client()
        self.client = self.client.get_or_create_collection(name="chatbot")

    def add_document(self):
        questions = [tokenize(preprocessing(query)) for query in self.questions]
        inputs = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            self.embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        self.embeddings = [list(embedding) for embedding in self.embeddings]
        self.embeddings = [list(map(float, embedding)) for embedding in self.embeddings]

        self.client.add(
            embeddings=self.embeddings,
            documents=self.documents,
            ids=[str(i) for i in range(len(self.documents))]
        )

    def search(self, query):
        query_embeddings = self.tokenizer(preprocessing(query), padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            query_embeddings = self.model(**query_embeddings, output_hidden_states=True, return_dict=True).pooler_output

        query_embeddings = list(query_embeddings)
        query_embeddings = [list(embedding) for embedding in query_embeddings]
        query_embeddings = [list(map(float, embedding)) for embedding in query_embeddings]

        results = self.client.query(
            query_embeddings=query_embeddings,
            n_results=1
        )
        return results['documents'][0][0]
