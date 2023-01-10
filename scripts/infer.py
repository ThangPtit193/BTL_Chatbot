from sentence_transformers import SentenceTransformer, util
from comet.components.embeddings.embedding_models import BertEmbedder
import pprint
import time
# model = SentenceTransformer('./models')
# embedder = BertEmbedder(pretrained_name_or_abspath='timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0')
embedder = BertEmbedder(pretrained_name_or_abspath='models_nli')
# embedder = BertEmbedder(pretrained_name_or_abspath='models_test/final_model')
# Two lists of sentences
queries = ["hi", "bye"]
corpus = [
    "good morning",
    "morning",
    "hello",
    "hi there",
    'i hate you',
    'I am a student',
    "vhgb bn  bn",
    "it's so beautiful",
    "see you lator",
    "it's so ugly",
    "a",
    "bla bla,"
    "tôi đi chươi"
]

from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
t0 = time.time()
from sentence_transformers import CrossEncoder
model = CrossEncoder('models_nli')
scores = model.predict([('hi', 'hello'), ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')])

#Convert scores to labels
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
print(labels)
# for query in queries:
#     print(f"Query: {query}")
#     t0 = time.time()
#     similarities = embedder.find_similarity([query], corpus)
#     print(f"Time: {time.time() - t0}")
#     pprint.pprint(similarities)

