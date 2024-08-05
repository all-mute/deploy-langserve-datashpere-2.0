# debug_embeddings.py
from yandex_chain import YandexEmbeddings

# Initialize the embeddings instance
embedding_instance = YandexEmbeddings(
    folder_id="___",
    api_key="___"
)

# Test the embedding generation
test_text = "This is a test text for generating embeddings."
test_embedding = embedding_instance.embed_query(test_text)

# Check if the embedding is generated correctly
if not test_embedding:
    raise ValueError("Failed to generate embedding.")
else:
    print("Embedding generated successfully.")
    print(f"Test text: {test_text}")
    print(f"Embedding: {test_embedding}")
