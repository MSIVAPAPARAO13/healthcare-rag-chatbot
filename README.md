# 🏥 Medical Chatbot using LLMs, LangChain, Pinecone & Flask

A production-ready **AI-powered Medical Chatbot** that leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, context-aware medical responses using custom PDF data.

---

## 🚀 Project Overview

This project implements an end-to-end **Medical Question Answering System** using:

* 📄 Custom medical documents (PDFs)
* 🧠 Embeddings + Vector Database (Pinecone)
* 🤖 Large Language Models (LLMs via Groq/OpenAI)
* 🔗 LangChain for orchestration
* 🌐 Flask for backend API
* 🐳 Docker + ☁️ AWS for deployment

---

## 🧠 Architecture

```
User Query
   ↓
Flask API
   ↓
Retriever (Pinecone)
   ↓
Relevant Chunks (PDF Data)
   ↓
LLM (Groq / OpenAI)
   ↓
Final Answer (Context-aware)
```

---

## ✨ Features

* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Custom PDF-based knowledge system
* ✅ Fast semantic search using Pinecone
* ✅ Scalable backend using Flask
* ✅ Dockerized for production
* ✅ AWS deployment ready (ECR + EC2)
* ✅ CI/CD with GitHub Actions

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **LLM:** Groq / OpenAI
* **Framework:** LangChain
* **Vector DB:** Pinecone
* **Embeddings:** Sentence Transformers
* **Deployment:** Docker, AWS (EC2, ECR)
* **CI/CD:** GitHub Actions

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd healthcare-rag-chatbot
```

---

### 2️⃣ Create Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables

Create `.env` file:

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

---

### 5️⃣ Ingest Data into Pinecone

```bash
python store_index.py
```

---

### 6️⃣ Run Application

```bash
python app.py
```

---

### 🌐 Access App

```
http://localhost:8080
```

---

## 🧪 Sample Queries

* What is Acne?
* What are symptoms of diabetes?
* How is hypertension treated?
* Causes of fever

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t medical-chatbot .
```

### Run Container

```bash
docker run -d -p 8080:8080 \
-e PINECONE_API_KEY=xxx \
-e GROQ_API_KEY=xxx \
medical-chatbot
```

---

## ☁️ AWS Deployment (Production)

### 🔹 Services Used

* EC2 (Compute)
* ECR (Container Registry)
* IAM (Access Control)

---

### 🔹 Deployment Steps

1. Build Docker image
2. Push image to AWS ECR
3. Launch EC2 instance
4. Install Docker on EC2
5. Pull image from ECR
6. Run container

---

### 🔹 IAM Policies

* AmazonEC2FullAccess
* AmazonEC2ContainerRegistryFullAccess

---

### 🔹 GitHub Actions (CI/CD)

* Automates build & deployment
* Uses secrets:

  * AWS_ACCESS_KEY_ID
  * AWS_SECRET_ACCESS_KEY
  * ECR_REPO
  * PINECONE_API_KEY
  * GROQ_API_KEY

---

## 📂 Project Structure

```
medical-chatbot/
│
├── app.py
├── store_index.py
├── requirements.txt
├── Dockerfile
├── .env
│
├── src/
│   ├── helper.py
│   ├── prompt.py
│
├── templates/
├── static/
└── data/
```

---

## 🔥 Key Highlights (For Interview)

* Built **end-to-end RAG system**
* Integrated **LLM + Vector DB + Backend**
* Solved **hallucination using context retrieval**
* Implemented **scalable deployment using Docker & AWS**
* Designed **real-world healthcare assistant**

---

## 🚧 Future Improvements

* Add chat memory
* Streaming responses
* Voice input/output
* Multi-language support
* Fine-tuned medical models

---

## 👨‍💻 Author

**Siva Paparao Medisetti**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
