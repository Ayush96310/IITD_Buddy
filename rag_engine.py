import os
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# --- CONFIGURATION ---
DATA_FILE = "iitd_data.txt"
DB_DIR = "chroma_db"

def build_vector_db():
    print(">>> 1. Loading Knowledge Base...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please create it.")
        return None

    loader = TextLoader(DATA_FILE)
    documents = loader.load()
    
    print(">>> 2. Splitting Text into Chunks...")
    # Updated class usage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"   [+] Created {len(chunks)} chunks.")
    
    print(">>> 3. Embedding & Indexing (ChromaDB)...")
    # Updated embedding class
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and persist database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print("   [+] Database Saved!")
    return vector_db

def search_iitd(query):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load from disk
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    
    print(f"\n>>> User Query: '{query}'")
    print(">>> Searching Knowledge Base...")
    
    results = vector_db.similarity_search(query, k=3)
    
    print("\n--- RETRIEVED FACTS ---")
    for i, res in enumerate(results):
        print(f"SOURCE {i+1}: {res.page_content}")
    print("-----------------------")

if __name__ == "__main__":
    # Create DB if it doesn't exist
    if not os.path.exists(DB_DIR):
        build_vector_db()
    
    # Loop for user questions
    while True:
        q = input("\nAsk IITD Buddy (or 'exit'): ")
        if q.lower() == 'exit': break
        search_iitd(q)