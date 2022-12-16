from sentence_transformers import SentenceTransformer, util
from comet.components.embeddings.embedding_models import BertEmbedder
import pprint

# model = SentenceTransformer('./models')
embedder = BertEmbedder(pretrained_name_or_abspath='timi-idol-phobert-32M-embedder-bm25-e10')
# Two lists of sentences
query = "hôm nay không biết thời tiết ra sao nữa"
corpus = [
    "hôm nay không biết thời tiết ra sao nữa",
    "hôm nay trời như nào nhỉ",
    "thời tiết ở Đà Nẵng ntn vậy timi",
    "không biết bây giờ Hà Nội có mưa không nhỉ",
    "cho mình hỏi thời tiết tại Hải Phòng với",
    "sắp đến đi du lịch mà ko biết thời tiết ở Đà Lạt có đỡ mưa ko ớ",
    "mình cần biết Hàn Quốc đang có nhiệt độ ra sao",
    "trời hôm nay nóng vãi",
    "thời tiết chán ghia bot",
    "trời mưa hoài dị trời",
    "trời ơi là trời, sao mùa hè mà mưa quoài dậy chời",
    "má ơi, nóng chết đi đc",
]
similarities = embedder.find_similarity([query], corpus)
pprint.pprint(similarities)
