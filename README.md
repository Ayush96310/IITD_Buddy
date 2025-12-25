# IITD Buddy: Privacy-First RAG Chatbot

**A local Retrieval-Augmented Generation (RAG) system that answers queries about institutional policies (Hostel Rules, Academics) using strict ground-truth documents.**

## 🚀 Overview
Standard LLMs hallucinate specific details about private institutions. IITD Buddy solves this by using a **Local RAG Pipeline** to retrieve exact clauses from a secure knowledge base before generating an answer.

* **Privacy:** Vector embeddings and storage are handled 100% locally using ChromaDB.
* **Accuracy:** Answers are generated using **Llama 3** (via Groq) with a strict `temperature=0` setting to prevent hallucination.

## 🛠️ Tech Stack
* **Orchestration:** LangChain (2025 Updated Stack)
* **Vector DB:** ChromaDB (Persistent Local Storage)
* **Inference:** Llama-3-70b via Groq API
* **Embeddings:** HuggingFace / Sentence-Transformers

## ⚡ How to Run
### 1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

```
### 2. **Setup Credentials**
Open `rag_chat_v2.py` and replace `ENTER_YOUR_GROQ_API_KEY_HERE` with your free Groq API key.

### 3. **Run the Chatbot:**
```bash
python rag_chat_v2.py
