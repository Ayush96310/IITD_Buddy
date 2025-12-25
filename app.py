import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IITD Buddy Pro", page_icon="🎓", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stButton>button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎓 IITD Buddy Pro")
    st.markdown("### ⚙️ Settings")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Rules/Papers (PDF)", type="pdf")
    reset_db = st.button("🔄 Reset Database")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- PROCESSING ENGINE ---
def process_pdf(uploaded_file):
    try:
        with st.spinner("🧠 Reading & Indexing Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # --- UPGRADE 1: Larger Chunks for Rulebooks ---
            # 1000 chars keeps paragraphs together. Overlap 200 prevents cut-off sentences.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            # Filter empty pages
            chunks = [c for c in chunks if c.page_content.strip()]
            
            if not chunks:
                st.error("❌ Error: No text found. PDF might be an image.")
                os.remove(tmp_path)
                return None

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings,
                persist_directory="chroma_db_pro"
            )
            os.remove(tmp_path)
            return vector_db

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- LOGIC FLOW ---
if reset_db:
    st.session_state.vector_db = None
    st.session_state.messages = []
    st.rerun()

if uploaded_file and st.session_state.vector_db is None:
    db = process_pdf(uploaded_file)
    if db:
        st.session_state.vector_db = db
        st.success("✅ Knowledge Base Ready!")

st.title("💬 IITD Rules Assistant")

if not uploaded_file:
    st.info("👈 Upload 'CoS 2024__UG Programme Rules.pdf' to start.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: What is the attendance requirement?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_db and groq_api_key:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    # --- UPGRADE 2: Retrieve MORE context (k=5) ---
                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                    context_docs = retriever.invoke(prompt)
                    
                    # Create a clean context block with Page Numbers
                    context_text = ""
                    for doc in context_docs:
                        context_text += f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}\n\n"

                    llm = ChatGroq(
                        temperature=0.1, # Slight creativity allowed for smooth sentences
                        model_name="llama-3.3-70b-versatile",
                        api_key=groq_api_key
                    )
                    
                    # --- UPGRADE 3: Strict "Persona" Prompt ---
                    final_prompt = f"""
                    You are an expert academic advisor at IIT Delhi.
                    Answer the question based strictly on the official rules provided below.
                    
                    Guidelines:
                    1. Quote the specific rule or value (e.g., "75% attendance").
                    2. Mention the Page Number if available in the context.
                    3. If the context does not have the answer, say "I cannot find this in the uploaded document."
                    
                    <official_rules>
                    {context_text}
                    </official_rules>
                    
                    Question: {prompt}
                    """
                    
                    response = llm.invoke(final_prompt)
                    message_placeholder.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})

                    # --- UPGRADE 4: Debug View (The "Resume Flex") ---
                    with st.expander("🔍 View Retrieved Source Context (Debug)"):
                        st.markdown(context_text)
                    
                except Exception as e:
                    message_placeholder.error(f"Error: {e}")