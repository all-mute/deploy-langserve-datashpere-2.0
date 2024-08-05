# debug_loader.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the document loader
loader = WebBaseLoader(
    web_paths=(
        "https://cloud.yandex.ru/ru/docs/datasphere/concepts/",
        "https://cloud.yandex.ru/ru/docs/datasphere/concepts/resource-model"
    )
)

# Load the documents
docs = loader.load()

# Check if documents are loaded correctly
if not docs:
    raise ValueError("No documents loaded.")
else:
    print(f"Loaded {len(docs)} documents.")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the documents
splits = text_splitter.split_documents(docs)

# Check if documents are split correctly
if not splits:
    raise ValueError("No document splits created.")
else:
    print(f"Created {len(splits)} document splits.")

# Print details of the splits
for i, split in enumerate(splits):
    print(f"Split {i+1}:")
    print(f"Content: {split.page_content[:200]}...")  # Print the first 200 characters
    print(f"Metadata: {split.metadata}")
    print("="*50)
