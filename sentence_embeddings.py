#My model is shitty and I am low on time.
#I will implement a pre-trained sentence transformer model
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The audio quality is great.",
    "The sound is very clear",
    "I drove my truck to the lake.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)


# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
