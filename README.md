# IITD Buddy Pro: RAG-Powered Knowledge Assistant

**A Full-Stack RAG Application that allows users to chat with institutional documents (PDFs) via a "WhatsApp-style" interface.**

## 🚀 Features
* **📄 Dynamic PDF Ingestion:** Drag-and-drop support for any PDF (Rules, Research Papers, Syllabi).
* **🧠 Conversation Memory:** Remembers context from previous turns (e.g., "What is the fine?" -> "Who do I pay **it** to?").
* **🛡️ Privacy-First:** Vector embeddings and storage are handled 100% locally using **ChromaDB**.
* **🤖 Llama 3 Intelligence:** Uses Groq's LPU for sub-second responses grounded in strict facts.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain (2025 Stack)
* **Vector DB:** ChromaDB (Local Persistence)
* **Model:** Llama-3-70b via Groq API
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

## ⚡ How to Run locally
### 1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Ayush96310/IITD_Buddy.git](https://github.com/Ayush96310/IITD_Buddy.git)
   cd IITD_Buddy

```
### 2. **Install Dependencies**
```bash
pip install -r requirements.txt

```
### 3. **Launch the App**
```bash
streamlit run app.py
```
### 4. **Usage:**

    Enter your Groq API Key in the Sidebar.

    Upload a PDF.

    Start chatting!

