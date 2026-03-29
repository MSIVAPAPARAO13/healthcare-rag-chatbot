# ================================
# IMPORTS
# ================================
from typing import List
from langchain.schema import Document

# ✅ Updated imports (no deprecation warnings)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# ================================
# 1. LOAD PDF FILES
# ================================
def load_pdf_file(data: str) -> List[Document]:
    """
    Load all PDF files from a directory.
    Each page becomes a Document.
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs")

    return documents


# ================================
# 2. FILTER METADATA
# ================================
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only page_content and source metadata.
    """
    minimal_docs = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": source}
            )
        )

    print(f"✅ Filtered {len(minimal_docs)} documents")
    return minimal_docs


# ================================
# 3. SPLIT TEXT INTO CHUNKS
# ================================
def text_split(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(docs)

    print(f"✅ Created {len(chunks)} text chunks")
    return chunks


# ================================
# 4. LOAD EMBEDDINGS
# ================================
def download_hugging_face_embeddings():
    """
    Load HuggingFace embedding model.
    Output dimension = 384
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    print("✅ Embedding model loaded")

    return embeddings


# ================================
# 5. FULL PIPELINE (OPTIONAL)
# ================================
def run_pipeline(data_path: str):
    """
    Complete pipeline:
    PDF → Clean → Chunk → Embedding
    """
    documents = load_pdf_file(data_path)
    minimal_docs = filter_to_minimal_docs(documents)
    chunks = text_split(minimal_docs)
    embeddings = download_hugging_face_embeddings()

    return chunks, embeddings


# ================================
# 6. TEST RUN
# ================================
if __name__ == "__main__":
    DATA_PATH = "data"

    chunks, embeddings = run_pipeline(DATA_PATH)

    # Test embedding
    test_query = "What is acne?"
    vector = embeddings.embed_query(test_query)

    print("\n🧪 Sample Query:", test_query)
    print("🔢 Vector length:", len(vector))