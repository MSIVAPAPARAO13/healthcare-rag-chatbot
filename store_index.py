from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 🔹 Step 1: Load PDF data
extracted_data = load_pdf_file(data='data/')

# 🔹 Step 2: Clean metadata
filtered_data = filter_to_minimal_docs(extracted_data)

# 🔹 Step 3: Split into chunks
text_chunks = text_split(filtered_data)

print(f"Total chunks created: {len(text_chunks)}")

# 🔹 Step 4: Load embedding model
embeddings = download_hugging_face_embeddings()

# 🔹 Step 5: Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# 🔥 FIXED (new Pinecone version)
if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )
else:
    print("Index already exists!")

# Connect to index
index = pc.Index(index_name)

# 🔹 Step 6: Store embeddings in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("✅ Data successfully stored in Pinecone!")