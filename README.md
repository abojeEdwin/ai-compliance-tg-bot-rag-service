# AI Compliance Telegram Bot - RAG Service

A highly optimized Retrieval-Augmented Generation (RAG) backend service for an AI Compliance Telegram Bot. This service is designed to provide contextually grounded, technical answers to user queries. For its knowledge base, it is configured to crawl Youverify's developer documentation, process and chunk the text into a vector database, and use Cohere's advanced LLMs to generate highly accurate compliance and integration answers.

## 🌟 Features
* **Automated Documentation Crawling:** Uses Firecrawl to actively scrape and index `doc.youverify.co`.
* **Parent-Child Chunking Strategy:** Implements advanced chunking logic—retrieving broader parent contexts based on precise child-vector similarities for highly accurate RAG.
* **Telemetry Filtering:** Intelligent chunk filtering drops client-side browser fingerprinting fragments (like WebGL/Canvas telemetry) to maintain high-quality context and prevent hallucinations.
* **Vector Database:** Uses PostgreSQL with `pgvector` for blazingly fast cosine distance search via `asyncpg`.
* **Zero-Hallucination Answers:** Powered by Cohere's `command-r7b-12-2024` with a deterministic temperature (`0.0`) and grounded context logic.

## 🏗️ Architecture

1. **Scraping (`crawl_and_seed.py`)**: Crawls API/SDK documentation via Firecrawl. The markdown is divided into parent blocks and smaller overlapping child chunks.
2. **Ingestion Endpoint (`/api/ingest`)**: The crawler sends structured data here. It filters out junk telemetry and generates embeddings via Cohere's `embed-english-v3.0`.
3. **Database (`pgvector`)**: Stores `parent_documents` (raw text) and `child_vectors` (searchable vector points mapped to parents).
4. **Query Pipeline (`/api/rag/ask`)**: Called by the Node.js Telegram bot orchestrator. Embeds the user prompt, performs vector search against child chunks, retrieves the corresponding parent texts, and asks the LLM to generate a grounded answer.

## 🛠️ Tech Stack
* **Python 3.12+**
* **Framework:** FastAPI / Uvicorn
* **Database:** PostgreSQL (with `pgvector`), `asyncpg`
* **AI Models:** Cohere (`embed-english-v3.0` & `command-r7b-12-2024`)
* **Scraping:** Firecrawl API

## 🚀 Local Setup & Installation

### 1. Requirements
Ensure you have Python 3.12 and PostgreSQL (with the `pgvector` extension) installed.

### 2. Environment Setup
Clone the repository and install dependencies:
```bash
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create an `app/.env` file with the following variables:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/youverify_rag
FIRECRAWL_API_KEY=your_firecrawl_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 4. Initialize the Database
Run the database initialization script to create the tables:
```bash
python scripts/init_db.py
```

### 5. Start the API Server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Run the Crawler
In a separate terminal, trigger the Firecrawl data pipeline to seed the vector database:
```bash
python crawl_and_seed.py
```

## ☁️ Deployment on Render

This service is optimized for Render deployments. 
* **Root Directory:** Must be set to `app` in the Render dashboard so the system detects the `requirements.txt`.
* **Python Version:** Make sure to set `PYTHON_VERSION` to `3.12.0` (or another stable version) in your Render environment variables to ensure `pydantic-core` and other native dependencies build correctly without hitting read-only filesystem errors.
* **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`