# IITD Buddy: Privacy-First RAG Search Engine

**A local Retrieval-Augmented Generation (RAG) system designed to query internal institutional documents (Hostel Rules, Academic Policies) without sending private data to cloud LLMs.**

![RAG Architecture](https://miro.medium.com/v2/resize:fit:1400/1*dHQqI2d_i_Pty0jK9Qx8-Q.png)
*(Standard RAG Architecture implemented locally)*

## 🚀 The Problem
General LLMs (like ChatGPT) hallucinate when asked about specific, private institutional rules.
* **Example:** "What is the fine for late entry at Aravali Hostel?" -> ChatGPT guesses.
* **Constraint:** Institutional data (student details, exact fines) cannot be uploaded to public API endpoints due to privacy concerns.

## 💡 The Solution
IITD Buddy is a **Local RAG Pipeline** that runs 100% on-device:
1.  **Ingestion:** Loads raw policy documents (`.txt`).
2.  **Vectorization:** Chunks text and creates embeddings using **Sentence-BERT (`all-MiniLM-L6-v2`)**.
3.  **Storage:** Indexes vectors in **ChromaDB** (Persistent Local Vector Store).
4.  **Retrieval:** Performs Semantic Search to retrieve the exact clause (e.g., "Fine is Rs. 500") rather than generating a probabilistic guess.

## 🛠️ Tech Stack
* **Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace / SBERT
* **Language:** Python 3.10+

## ⚡ How to Run
### 1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

```
### 2. **Add Data**
Update `iitd_data.txt` with your institutional rules and policies.

### 3. **Run the Engine**
```bash
python rag_engine.py
