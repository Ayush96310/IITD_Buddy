import os
import warnings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
GROQ_API_KEY = "ENTER_YOUR_GROQ_API_KEY_HERE"
DB_DIR = "chroma_db"

# Silence warnings
warnings.filterwarnings("ignore")

def generate_answer(query):
    # 1. SETUP RETRIEVAL
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(DB_DIR):
        print("Error: Database not found. Run rag_engine.py first to build it.")
        return

    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    
    # 2. RETRIEVE CONTEXT
    print(f"\n[Thinking] Retrieving facts for: '{query}'...")
    results = vector_db.similarity_search(query, k=3)
    
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # 3. SETUP LLM
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=GROQ_API_KEY
    )

    # 4. CREATE PROMPT
    prompt_template = ChatPromptTemplate.from_template("""
    You are an official assistant for IIT Delhi.
    Answer the question based ONLY on the following context. 
    If the answer is not in the context, say "I don't have that information in my official records."
    Do not make up rules.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    # 5. GENERATE ANSWER
    chain = prompt_template | llm
    response = chain.invoke({"context": context_text, "question": query})
    
    # 6. PRINT RESULT
    print("\n" + "="*40)
    print("IITD Buddy says:")
    print(response.content)
    print("="*40)

if __name__ == "__main__":
    print(">>> IITD Buddy (Llama 3 Edition) Initialized.")
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == 'exit': break
        
        try:
            generate_answer(q)
        except Exception as e:
            print(f"Error: {e}")
            print("Check your API Key and internet connection.")