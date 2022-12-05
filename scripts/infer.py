from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('./models')
model = SentenceTransformer('khanhpd2/sbert_phobert_large_cosine_sim')

# Two lists of sentences
sentences1 = [
    "Tôi rất thích con mèo",
    "Tôi rất thích con mèo",
    '"Tôi rất thích con chó',
    'Tôi rất thích con mèo',
    'Tôi rất thích con chó',
    'hôm nay tôi đi học',
    'hôm nay tôi đi học',
]
sentences2 = [
    "Tôi không thích con mèo",
    'Tôi yêu con mèo rất nhiều',
    'Tôi thương con chó rất nhiều',
    'Tôi rất ghét con mèo',
    'Tôi rất ghét con chó',
    'hôm nay tôi đến trường',
    'hôm nay tôi đi chơi',
]

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
