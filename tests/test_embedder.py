from comet.components.embeddings.embedding_models import BertEmbedder
import time


def test_encodes():
    embedder = BertEmbedder(
        pretrained_name_or_abspath="timi-keepitreal-H768-faq-2M-v1.1.3",
        batch_size=32,
        device="cuda:1",
    )
    start = time.time()
    text_list = ["hello" + str(i) for i in range(1000)]
    res = embedder.get_encodings(text_list, add_to_cache=False)
    print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
    test_encodes()
