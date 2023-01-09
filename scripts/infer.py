from sentence_transformers import SentenceTransformer, util
from comet.components.embeddings.embedding_models import BertEmbedder
import pprint
import time
# model = SentenceTransformer('./models')
# embedder = BertEmbedder(pretrained_name_or_abspath='timi-idol-keepitreal-vn-sbert-faq-9M-v1.0.0')
embedder = BertEmbedder(pretrained_name_or_abspath='models')
# Two lists of sentences
query = "it's so ugly"
corpus = [
    "hi there",
    'i hate you',
    'I am a student',
    "vhgb bn  bn",
    "it's so beautiful",
    "it's so ugly"
]
t0 = time.time()
similarities = embedder.find_similarity([query], corpus)
print(f"Time: {time.time() - t0}")
pprint.pprint(similarities)

