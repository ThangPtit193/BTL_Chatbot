from sentence_transformers import SentenceTransformer, util
from comet.components.embeddings.embedding_models import BertEmbedder
import pprint

# model = SentenceTransformer('./models')
# embedder = BertEmbedder(pretrained_name_or_abspath='models/epoch-286')
embedder = BertEmbedder(pretrained_name_or_abspath='models/final_model')
# Two lists of sentences
query = "Tôi rất thích con mèo"
corpus = [
    "Tôi yêu con mèo rất nhiều",
    "Tôi rất ghét con mèo",
    "Tôi rất thương con chó",
    "Tôi thương con mèo rất nhiều"


]
similarities = embedder.find_similarity([query], corpus)
pprint.pprint(similarities)
