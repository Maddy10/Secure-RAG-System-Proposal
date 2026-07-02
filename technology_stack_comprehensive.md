# Technology Stack for RAG Framework
## ISRO VSSC - RES-VSSC-2025-003
### Secure and Accurate Retrieval-Augmented Generation Framework

---

## 📋 TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Core Components Stack](#core-components-stack)
3. [Embedding Models](#embedding-models)
4. [Large Language Models (LLMs)](#large-language-models-llms)
5. [Vector Databases](#vector-databases)
6. [Reranking Models](#reranking-models)
7. [Document Processing](#document-processing)
8. [RAG Frameworks & Orchestration](#rag-frameworks--orchestration)
9. [Development & Programming](#development--programming)
10. [Infrastructure & Deployment](#infrastructure--deployment)
11. [Security & Monitoring](#security--monitoring)
12. [Testing & Evaluation](#testing--evaluation)
13. [Complete Stack Summary](#complete-stack-summary)
14. [Implementation Roadmap](#implementation-roadmap)

---

## 🎯 EXECUTIVE SUMMARY

This document provides a comprehensive technology stack recommendation for building a production-grade, **offline-capable RAG system** for ISRO VSSC. All recommendations prioritize:

- ✅ **Offline/Air-gapped deployment capability**
- ✅ **Open-source models** (no dependency on external APIs)
- ✅ **High accuracy and reliability**
- ✅ **Security and data privacy**
- ✅ **Active community support**
- ✅ **Performance optimization for Indian context**

**Key Philosophy:** Use battle-tested, open-source technologies that can run completely offline on local infrastructure.

---

## 🧠 CORE COMPONENTS STACK

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│         Gradio / Streamlit / FastAPI + React            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              RAG ORCHESTRATION LAYER                     │
│        LangChain / LlamaIndex / Custom Python           │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                     ↓
┌──────────────────┐              ┌──────────────────────┐
│   RETRIEVAL      │              │    GENERATION        │
│                  │              │                      │
│ • Embeddings:    │              │ • LLM:               │
│   BGE-M3 / E5    │              │   Llama 3.1 / Qwen   │
│                  │              │                      │
│ • Vector DB:     │              │ • Inference:         │
│   FAISS / Chroma │              │   vLLM / TGI         │
│                  │              │                      │
│ • Lexical:       │              │ • Framework:         │
│   BM25 / Elastic │              │   Transformers       │
│                  │              │                      │
│ • Reranker:      │              │                      │
│   BGE-reranker   │              │                      │
└──────────────────┘              └──────────────────────┘
        ↓                                     ↑
┌─────────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING LAYER                   │
│   PyMuPDF / Unstructured / DocTR / Tesseract           │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│                  DATA STORAGE LAYER                      │
│        PostgreSQL + pgvector / File System              │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 EMBEDDING MODELS

### Recommended: BGE-M3 (BAAI General Embedding - Multilingual)

**Model:** `BAAI/bge-m3`

**Why BGE-M3:**
- ✅ **State-of-the-art performance** on retrieval benchmarks
- ✅ **Multilingual support** (100+ languages, includes Indian languages)
- ✅ **Multi-functionality:** Dense, sparse, and multi-vector representations
- ✅ **8192 token context** - handles long documents
- ✅ **Open-source** (MIT license)
- ✅ **Optimized for Indian context** - works well with technical English

**Technical Specifications:**
```yaml
Model Name: BAAI/bge-m3
Size: 568M parameters
Embedding Dimension: 1024
Max Sequence Length: 8192 tokens
Languages: 100+ (including Hindi, Tamil, Telugu)
License: MIT
Hardware: 2GB GPU VRAM (inference)
Performance: 
  - MTEB Score: 66.1 (average across tasks)
  - BEIR Score: 54.9
  - Retrieval Recall@10: ~85% on technical docs
```

**Usage Example:**
```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('BAAI/bge-m3')

# Generate embeddings
documents = [
    "PSLV C51 launched successfully from Sriharikota",
    "Telemetry data shows nominal performance"
]

embeddings = model.encode(documents, normalize_embeddings=True)
# Shape: (2, 1024)
```

**Alternative Options:**

| Model | Embedding Dim | Max Tokens | Size | Best For |
|-------|--------------|------------|------|----------|
| **E5-Mistral-7B** | 4096 | 32768 | 7B | Highest accuracy, needs more GPU |
| **BGE-Large-EN-v1.5** | 1024 | 512 | 335M | English-only, faster inference |
| **Jina-Embeddings-v2** | 768 | 8192 | 137M | Lightweight, good baseline |
| **UAE-Large-v1** | 1024 | 512 | 335M | Arabic/English, if needed |

**Recommendation:** 
- **Primary:** BGE-M3 (best balance of performance and resource efficiency)
- **Fallback:** E5-Mistral-7B (if you have ample GPU resources)

---

## 🤖 LARGE LANGUAGE MODELS (LLMs)

### Recommended: Meta Llama 3.1 (8B Instruct)

**Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`

**Why Llama 3.1:**
- ✅ **Excellent instruction-following** for RAG tasks
- ✅ **128K context window** - can handle large retrieved contexts
- ✅ **Strong reasoning capabilities**
- ✅ **Open-source** (Llama 3 Community License)
- ✅ **Quantization-friendly** - can run on 16GB GPU
- ✅ **Tool use support** - for advanced RAG workflows

**Technical Specifications:**
```yaml
Model Name: Meta-Llama-3.1-8B-Instruct
Parameters: 8 billion
Context Length: 128K tokens
Quantization: 4-bit, 8-bit support (via bitsandbytes)
Hardware Requirements:
  - Full precision (FP16): 16GB GPU VRAM
  - 8-bit quantization: 10GB GPU VRAM
  - 4-bit quantization: 6GB GPU VRAM
License: Llama 3 Community License (permissive for research)
Performance:
  - MMLU: 68.4
  - HumanEval: 62.2
  - GSM8K: 79.6
```

**Usage Example:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model with quantization
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # 4-bit quantization for efficiency
)

# RAG prompt template
prompt = f"""
You are a helpful AI assistant for ISRO. Answer based ONLY on the provided context.

Context:
{retrieved_context}

Question: {user_query}

Answer:"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.1,  # Low temperature for factual accuracy
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Alternative LLM Options:**

### Option 2: Qwen 2.5 (7B Instruct)

**Model:** `Qwen/Qwen2.5-7B-Instruct`

**Why Qwen 2.5:**
- ✅ **Excellent for technical/mathematical reasoning**
- ✅ **128K context window**
- ✅ **Strong performance on code and structured data**
- ✅ **Open-source** (Apache 2.0 license)
- ✅ **Multilingual** (including Chinese, if needed for collaboration)

**Technical Specifications:**
```yaml
Model Name: Qwen/Qwen2.5-7B-Instruct
Parameters: 7.6 billion
Context Length: 128K tokens
Hardware Requirements: 14GB GPU VRAM (FP16)
License: Apache 2.0
Performance:
  - MMLU: 70.3
  - HumanEval: 61.0
  - GSM8K: 82.5
  - C-Eval (Chinese): 83.1
```

### Option 3: Mistral 7B v0.3 (Instruct)

**Model:** `mistralai/Mistral-7B-Instruct-v0.3`

**Why Mistral:**
- ✅ **High quality outputs**
- ✅ **32K context window**
- ✅ **Efficient architecture** (Sliding Window Attention)
- ✅ **Apache 2.0 license**

**Technical Specifications:**
```yaml
Model Name: mistralai/Mistral-7B-Instruct-v0.3
Parameters: 7.3 billion
Context Length: 32K tokens
Hardware Requirements: 14GB GPU VRAM (FP16)
License: Apache 2.0
Performance:
  - MMLU: 62.5
  - HumanEval: 40.2
  - GSM8K: 52.2
```

### LLM Comparison Table

| Model | Params | Context | MMLU | License | Best For |
|-------|--------|---------|------|---------|----------|
| **Llama 3.1-8B** | 8B | 128K | 68.4 | Llama 3 | General RAG, instruction following |
| **Qwen 2.5-7B** | 7.6B | 128K | 70.3 | Apache 2.0 | Technical/math reasoning |
| **Mistral 7B** | 7.3B | 32K | 62.5 | Apache 2.0 | Balanced, efficient |
| **Llama 3.1-70B** | 70B | 128K | 83.6 | Llama 3 | Maximum accuracy (needs 2x A100) |

**Final Recommendation:**
- **Primary:** Llama 3.1-8B-Instruct (best overall for RAG)
- **Alternative:** Qwen 2.5-7B-Instruct (if focus on technical/mathematical content)
- **High-end:** Llama 3.1-70B-Instruct (if GPU resources available, for maximum accuracy)

**Multi-Model Strategy:**
```python
# Use different models for different domains
MODEL_ROUTING = {
    'technical': 'Qwen2.5-7B-Instruct',  # Better for specs, calculations
    'administrative': 'Llama-3.1-8B-Instruct',  # Better for policy, prose
    'qa_reports': 'Llama-3.1-8B-Instruct',  # Better for analysis
}
```

---

## 🗄️ VECTOR DATABASES

### Recommended: FAISS (Facebook AI Similarity Search)

**Library:** `faiss-cpu` or `faiss-gpu`

**Why FAISS:**
- ✅ **Optimized for offline use** - no server needed
- ✅ **Extremely fast** - billion-scale vector search in milliseconds
- ✅ **Memory efficient** - multiple index types for different use cases
- ✅ **GPU acceleration** support
- ✅ **Proven at scale** - used by Meta, OpenAI, and others
- ✅ **Easy to backup/restore** - just copy index files
- ✅ **No licensing restrictions** - MIT license

**Technical Specifications:**
```yaml
Library: FAISS
Version: 1.8.0+
License: MIT
Supported Indices:
  - IndexFlatL2: Exact search (small datasets)
  - IndexIVFFlat: Fast approximate search
  - IndexHNSW: Best accuracy/speed tradeoff
  - IndexPQ: Most memory efficient
GPU Support: Yes (CUDA)
Max Vectors: Billions
Typical Latency: <10ms for millions of vectors
```

**Usage Example:**
```python
import faiss
import numpy as np

# Create index
dimension = 1024  # BGE-M3 embedding dimension
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = HNSW parameter

# Add vectors
embeddings = np.random.random((10000, dimension)).astype('float32')
index.add(embeddings)

# Search
query = np.random.random((1, dimension)).astype('float32')
k = 10  # Top-K results
distances, indices = index.search(query, k)

# Save index (for offline backup)
faiss.write_index(index, "isro_rag_index.faiss")

# Load index (fast startup)
index = faiss.read_index("isro_rag_index.faiss")
```

**Index Selection Guide:**

| Index Type | Use Case | Speed | Memory | Accuracy |
|-----------|----------|-------|--------|----------|
| **IndexFlatL2** | <10K vectors, exact search | Slow | High | 100% |
| **IndexHNSWFlat** | <10M vectors, best quality | Fast | High | ~99% |
| **IndexIVFFlat** | <100M vectors, balanced | Very Fast | Medium | ~95% |
| **IndexPQ** | >100M vectors, memory-constrained | Fast | Low | ~90% |

**Recommendation for ISRO:**
- **Document Count < 100K:** `IndexHNSWFlat` (best accuracy)
- **Document Count 100K-1M:** `IndexIVFFlat` with 4096 clusters
- **Document Count > 1M:** `IndexPQ` with HNSW optimization

### Alternative: ChromaDB

**Library:** `chromadb`

**Why ChromaDB:**
- ✅ **Simple to use** - Python-first design
- ✅ **Built-in metadata filtering**
- ✅ **Persistent storage** out of the box
- ✅ **Apache 2.0 license**

**Technical Specifications:**
```yaml
Library: ChromaDB
Version: 0.4.0+
License: Apache 2.0
Backend: SQLite + HNSW
Typical Latency: 20-50ms
Max Vectors: Millions
Metadata Support: Yes (JSON)
```

**Usage Example:**
```python
import chromadb
from chromadb.config import Settings

# Create persistent client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/data/chromadb"
))

# Create collection
collection = client.create_collection(
    name="isro_documents",
    metadata={"description": "ISRO technical docs"}
)

# Add documents with metadata
collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=[
        {"domain": "technical", "classification": "internal"},
        {"domain": "admin", "classification": "confidential"}
    ],
    ids=[f"doc_{i}" for i in range(len(texts))]
)

# Query with metadata filtering
results = collection.query(
    query_embeddings=query_embedding,
    n_results=10,
    where={"domain": "technical"}  # Filter by metadata
)
```

### Alternative: Milvus (Self-Hosted)

**Why Milvus:**
- ✅ **Production-grade** vector database
- ✅ **Horizontal scalability**
- ✅ **Advanced filtering** and hybrid search
- ✅ **Can be deployed offline**

**When to Choose Milvus:**
- Large scale (>10M documents)
- Need for advanced filtering
- Multi-user concurrent access
- Production deployment with SLA requirements

**Comparison:**

| Feature | FAISS | ChromaDB | Milvus |
|---------|-------|----------|--------|
| **Ease of Use** | Medium | Easy | Medium |
| **Performance** | Excellent | Good | Excellent |
| **Scalability** | Good | Medium | Excellent |
| **Metadata Filtering** | No | Yes | Yes |
| **Setup Complexity** | Low | Low | High |
| **Resource Usage** | Low | Medium | High |
| **Offline Use** | Perfect | Perfect | Good |

**Final Recommendation:**
- **Start with:** FAISS (simplest, fastest, most reliable offline)
- **Upgrade to:** Milvus (if scaling beyond 10M documents or need advanced features)
- **Alternative:** ChromaDB (if metadata filtering is critical from day 1)

---

## 🔄 RERANKING MODELS

### Recommended: BGE-Reranker-V2-M3

**Model:** `BAAI/bge-reranker-v2-m3`

**Why BGE-Reranker:**
- ✅ **Purpose-built for reranking** - not a general encoder
- ✅ **Multilingual support** (same as BGE-M3 embeddings)
- ✅ **Significantly improves recall** (15-20% improvement typical)
- ✅ **Fast inference** - 100ms for 20 documents
- ✅ **Lightweight** - runs on CPU if needed

**Technical Specifications:**
```yaml
Model Name: BAAI/bge-reranker-v2-m3
Size: 568M parameters
Max Sequence Length: 8192 tokens (query + document)
Languages: 100+
Hardware: 2GB GPU VRAM (or CPU)
License: MIT
Performance: 
  - BEIR reranking: +18% nDCG improvement
  - Average latency: 100ms for 20 docs
```

**Usage Example:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load reranker model
model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Rerank documents
def rerank(query: str, documents: list[str], top_k: int = 5):
    # Create pairs
    pairs = [[query, doc] for doc in documents]
    
    # Tokenize
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Get scores
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze()
    
    # Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Return top-k
    reranked_docs = [documents[i] for i in sorted_indices[:top_k]]
    reranked_scores = [scores[i].item() for i in sorted_indices[:top_k]]
    
    return reranked_docs, reranked_scores

# Example
query = "What is the thrust of PSLV third stage?"
initial_docs = retrieve_top_20(query)  # From vector search
reranked_docs, scores = rerank(query, initial_docs, top_k=5)
```

**Alternative: ColBERT v2**

**Model:** `colbert-ir/colbertv2.0`

**Why ColBERT:**
- ✅ **Late interaction** architecture - very accurate
- ✅ **Token-level matching** - catches nuanced similarities
- ✅ **Excellent for technical text**

**When to use ColBERT:**
- Need maximum accuracy (willing to trade speed)
- Technical domain with specific terminology
- Have GPU resources for indexing

**Comparison:**

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **BGE-Reranker-v2-M3** | Fast | Very Good | Low | General purpose, multilingual |
| **ColBERT v2** | Medium | Excellent | High | Maximum accuracy |
| **Cross-Encoder (MS-MARCO)** | Fast | Good | Low | English-only baseline |

**Final Recommendation:**
- **Primary:** BGE-Reranker-v2-M3 (best balance)
- **If need max accuracy:** ColBERT v2

**Reranking Strategy:**
```python
# Two-stage retrieval
# Stage 1: Fast vector search (retrieve top-100)
candidates = vector_search(query, k=100)

# Stage 2: Rerank to top-10
final_results = rerank(query, candidates, k=10)
```

---

## 📄 DOCUMENT PROCESSING

### PDF Processing: PyMuPDF (fitz)

**Library:** `PyMuPDF` (imports as `fitz`)

**Why PyMuPDF:**
- ✅ **Fastest PDF parsing** in Python
- ✅ **Preserves layout** - tables, images, formatting
- ✅ **Text extraction** with position information
- ✅ **Image extraction** from PDFs
- ✅ **No external dependencies**
- ✅ **GNU AGPL / Commercial license**

**Usage Example:**
```python
import fitz  # PyMuPDF

def extract_pdf_with_metadata(pdf_path: str):
    doc = fitz.open(pdf_path)
    
    results = {
        'metadata': doc.metadata,
        'pages': []
    }
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with position
        text = page.get_text("text")
        
        # Extract tables
        tables = page.find_tables()
        
        # Extract images
        images = page.get_images()
        
        results['pages'].append({
            'page_num': page_num + 1,
            'text': text,
            'tables': [t.extract() for t in tables],
            'images': len(images)
        })
    
    doc.close()
    return results
```

**Alternatives:**
- **pdfplumber:** Better table extraction, slower
- **PyPDF2:** Lightweight, basic functionality
- **Apache Tika:** Multi-format support (Java dependency)

### OCR: Tesseract + DocTR

**For Scanned Documents:**

**Tesseract OCR**
```yaml
Library: pytesseract
Engine: Tesseract 5.x
Languages: 100+ (including Hindi, Tamil, Telugu)
Accuracy: 90-95% on good quality scans
Speed: ~1 page/second
License: Apache 2.0
```

**Usage:**
```python
from PIL import Image
import pytesseract

def ocr_image(image_path: str, lang='eng'):
    img = Image.open(image_path)
    
    # OCR with confidence
    data = pytesseract.image_to_data(img, lang=lang, output_type='dict')
    
    # Filter low confidence
    text = ' '.join([
        data['text'][i] 
        for i in range(len(data['text'])) 
        if int(data['conf'][i]) > 60
    ])
    
    return text
```

**DocTR (Document Text Recognition)**
```yaml
Library: python-doctr
Features: Deep learning OCR, layout analysis
Accuracy: 95%+ on printed documents
Speed: GPU-accelerated
License: Apache 2.0
```

**Usage:**
```python
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load model
model = ocr_predictor(pretrained=True)

# Process document
doc = DocumentFile.from_pdf("document.pdf")
result = model(doc)

# Extract text
text = result.export()
```

### Document Parsing: Unstructured

**Library:** `unstructured`

**Why Unstructured:**
- ✅ **Multi-format support** - PDF, DOCX, HTML, Markdown, Email
- ✅ **Layout analysis** - preserves document structure
- ✅ **Table detection**
- ✅ **Chunking strategies** built-in
- ✅ **Open-source** (Apache 2.0)

**Usage Example:**
```python
from unstructured.partition.auto import partition

# Automatically detect format and parse
elements = partition(filename="document.pdf")

# Filter by element type
tables = [el for el in elements if el.category == "Table"]
text = [el for el in elements if el.category == "NarrativeText"]

# Extract with metadata
for element in elements:
    print(f"Type: {element.category}")
    print(f"Text: {element.text}")
    print(f"Metadata: {element.metadata}")
```

### Excel/Spreadsheet: Pandas + openpyxl

**Libraries:** `pandas`, `openpyxl`

**Usage:**
```python
import pandas as pd

# Read Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# Convert to text for RAG
text_representation = df.to_string()

# Or structure-aware
for idx, row in df.iterrows():
    text = f"Row {idx}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
```

### Document Processing Stack Summary

| Format | Library | Purpose |
|--------|---------|---------|
| **PDF (text)** | PyMuPDF | Fast text extraction |
| **PDF (scanned)** | DocTR + Tesseract | OCR |
| **DOCX** | python-docx | Word documents |
| **XLSX/CSV** | pandas + openpyxl | Spreadsheets |
| **HTML** | BeautifulSoup4 | Web pages |
| **Markdown** | markdown | MD files |
| **Multi-format** | Unstructured | Unified pipeline |
| **Images** | Pillow + pytesseract | Image processing |
| **Email** | Unstructured | Email parsing |

---

## 🔧 RAG FRAMEWORKS & ORCHESTRATION

### Recommended: LlamaIndex

**Library:** `llama-index`

**Why LlamaIndex:**
- ✅ **Built specifically for RAG** - not a general LLM framework
- ✅ **Advanced retrieval strategies** - hybrid search, reranking, query expansion
- ✅ **Modular architecture** - easy to customize
- ✅ **Extensive integrations** - works with all major models and vector DBs
- ✅ **Production-ready** - used by many enterprises
- ✅ **Excellent documentation**
- ✅ **MIT license**

**Key Features:**
- Multiple retrieval modes (vector, keyword, hybrid)
- Built-in chunking strategies
- Query transformations (HyDE, query rewriting)
- Response synthesis modes
- Evaluation framework
- Multi-document agents

**Usage Example:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# Configure models
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto"
)

# Load documents
documents = SimpleDirectoryReader("/data/isro_docs").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine(
    similarity_top_k=10,
    response_mode="compact"
)

response = query_engine.query("What is the payload capacity of PSLV?")
print(response)
```

**Advanced RAG with LlamaIndex:**
```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer

# Custom retrieval pipeline
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=20,
)

# Reranking/filtering
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

# Response synthesis
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",  # For long contexts
    use_async=True
)

# Combine into query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[postprocessor],
)

response = query_engine.query("Analyze QA failures in Q4 2024")
```

### Alternative: LangChain

**Library:** `langchain`

**Why LangChain:**
- ✅ **Most popular** LLM framework
- ✅ **Massive ecosystem** - integrations with everything
- ✅ **Agent capabilities** - for complex workflows
- ✅ **Active development**
- ✅ **MIT license**

**When to choose LangChain:**
- Need complex multi-step workflows
- Want to build agents with tools
- Prefer more control over each component

**Usage Example:**
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Setup embedding
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# Load and split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# Setup LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation",
    device=0,
    pipeline_kwargs={"max_new_tokens": 512}
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is PSLV payload capacity?"})
print(result['result'])
print(result['source_documents'])
```

### Custom Python (Recommended for Maximum Control)

**Why Custom:**
- ✅ **Full control** over every component
- ✅ **No abstraction overhead**
- ✅ **Easier to debug**
- ✅ **Better performance optimization**
- ✅ **No framework lock-in**

**Architecture:**
```python
class ISRORAGSystem:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.vector_store = load_vector_store()
        self.reranker = load_reranker()
        self.llm = load_llm()
        self.bm25 = load_bm25_index()
    
    def retrieve(self, query: str, k: int = 100):
        # Hybrid retrieval
        dense_results = self.vector_store.search(
            self.embedding_model.encode(query), 
            k=k
        )
        sparse_results = self.bm25.search(query, k=k)
        
        # Fusion
        combined = reciprocal_rank_fusion(dense_results, sparse_results)
        return combined
    
    def rerank(self, query: str, documents: list, k: int = 10):
        scores = self.reranker.score(query, documents)
        sorted_docs = sort_by_scores(documents, scores)
        return sorted_docs[:k]
    
    def generate(self, query: str, context: list):
        prompt = self.build_prompt(query, context)
        response = self.llm.generate(prompt)
        return response
    
    def query(self, question: str):
        # Full pipeline
        candidates = self.retrieve(question, k=100)
        reranked = self.rerank(question, candidates, k=10)
        answer = self.generate(question, reranked)
        
        return {
            'answer': answer,
            'sources': reranked,
            'confidence': self.calculate_confidence(answer, reranked)
        }
```

**Framework Comparison:**

| Framework | Ease of Use | Flexibility | Performance | Best For |
|-----------|------------|-------------|-------------|----------|
| **LlamaIndex** | Easy | Medium | Good | Quick prototyping, standard RAG |
| **LangChain** | Medium | High | Good | Complex workflows, agents |
| **Custom Python** | Hard | Maximum | Best | Production systems, specific requirements |

**Recommendation:**
- **Start with:** LlamaIndex (fastest to prototype)
- **Production:** Custom Python (for ISRO's specific security and performance needs)

---

## 💻 DEVELOPMENT & PROGRAMMING

### Primary Language: Python 3.11+

**Why Python:**
- ✅ De facto standard for AI/ML
- ✅ Massive ecosystem
- ✅ Easy to maintain
- ✅ Excellent libraries

### Core Libraries

**Essential:**
```yaml
# ML/AI Core
torch: 2.1.0+          # PyTorch for model inference
transformers: 4.36.0+  # Hugging Face models
sentence-transformers: 2.2.0+  # Embeddings
accelerate: 0.25.0+    # GPU optimization
bitsandbytes: 0.41.0+  # Quantization

# Vector Search
faiss-cpu: 1.8.0       # or faiss-gpu
chromadb: 0.4.0+       # Alternative vector DB

# RAG Framework
llama-index: 0.9.0+    # RAG orchestration
langchain: 0.1.0+      # Alternative framework

# Document Processing
PyMuPDF: 1.23.0+       # PDF processing
python-docx: 1.0.0+    # Word documents
openpyxl: 3.1.0+       # Excel files
pandas: 2.1.0+         # Data manipulation
unstructured: 0.11.0+  # Multi-format parsing
pytesseract: 0.3.10+   # OCR
python-doctr: 0.7.0+   # Deep learning OCR

# Web & API
fastapi: 0.108.0+      # API framework
uvicorn: 0.25.0+       # ASGI server
gradio: 4.11.0+        # UI for demos
streamlit: 1.29.0+     # Alternative UI

# Utilities
pydantic: 2.5.0+       # Data validation
loguru: 0.7.0+         # Logging
python-dotenv: 1.0.0+  # Configuration
tqdm: 4.66.0+          # Progress bars

# Testing
pytest: 7.4.0+         # Unit testing
pytest-cov: 4.1.0+     # Coverage
```

**requirements.txt:**
```txt
# Core ML
torch==2.1.2
transformers==4.36.2
sentence-transformers==2.2.2
accelerate==0.25.0
bitsandbytes==0.41.3

# Vector Search
faiss-cpu==1.8.0
chromadb==0.4.18

# RAG
llama-index==0.9.48
langchain==0.1.0

# Document Processing
PyMuPDF==1.23.8
python-docx==1.1.0
openpyxl==3.1.2
pandas==2.1.4
unstructured==0.11.6
pytesseract==0.3.10
python-doctr[torch]==0.7.0

# API & Web
fastapi==0.108.0
uvicorn[standard]==0.25.0
gradio==4.11.0
pydantic==2.5.3

# Utilities
loguru==0.7.2
python-dotenv==1.0.0
tqdm==4.66.1
pyyaml==6.0.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1

# Security
cryptography==41.0.7
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

### Development Tools

**Code Quality:**
```yaml
black: 23.12.0+        # Code formatting
isort: 5.13.0+         # Import sorting
flake8: 7.0.0+         # Linting
mypy: 1.7.0+           # Type checking
pylint: 3.0.0+         # Additional linting
```

**Development Environment:**
```yaml
IDE: VSCode or PyCharm
Python Version Manager: pyenv
Package Manager: pip + venv
Dependency Management: pip-tools or poetry
Pre-commit Hooks: pre-commit framework
```

---

## 🚀 INFRASTRUCTURE & DEPLOYMENT

### Containerization: Docker

**Why Docker:**
- ✅ Consistent environments
- ✅ Easy deployment
- ✅ Isolation
- ✅ Reproducibility

**Dockerfile Example:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models (offline deployment)
RUN python -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('BAAI/bge-m3')"
RUN python -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')"

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: isro_rag
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  grafana_data:
```

### Orchestration: Kubernetes (Optional)

**For Large-Scale Deployment:**

**Deployment YAML:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: isro-rag-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: isro-rag
  template:
    metadata:
      labels:
        app: isro-rag
    spec:
      containers:
      - name: rag-api
        image: isro-rag:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: data
          mountPath: /app/data
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc
```

### Model Serving: vLLM or Text Generation Inference (TGI)

**vLLM (Recommended)**

**Why vLLM:**
- ✅ **Fastest inference** - PagedAttention algorithm
- ✅ **High throughput** - batch processing
- ✅ **Dynamic batching**
- ✅ **OpenAI-compatible API**
- ✅ **Excellent GPU utilization**

**Installation:**
```bash
pip install vllm
```

**Usage:**
```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,  # Number of GPUs
    dtype="float16",
    max_model_len=8192
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512
)

# Generate
prompts = ["What is PSLV?", "Explain telemetry."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**vLLM Server:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype float16 \
    --max-model-len 8192 \
    --port 8001
```

**Alternative: Text Generation Inference (TGI)**

**By Hugging Face:**
```bash
# Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
```

### Database: PostgreSQL + pgvector

**For Metadata and Hybrid Search:**

```yaml
Database: PostgreSQL 15+
Extension: pgvector
Purpose: Store document metadata, user data, audit logs
```

**Schema Example:**
```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE,
    title TEXT,
    content TEXT,
    domain VARCHAR(50),
    classification VARCHAR(50),
    embedding vector(1024),  -- For BGE-M3
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Create index for vector search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Audit logs table
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    query TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    response_time_ms INTEGER
);
```

### Hardware Recommendations

**Minimum Specifications:**
```yaml
CPU: 16 cores (Intel Xeon or AMD EPYC)
RAM: 64GB DDR4
GPU: 1x NVIDIA A10 (24GB VRAM) or RTX 4090 (24GB)
Storage: 1TB NVMe SSD
Network: 10 Gbps (air-gapped, internal)
```

**Recommended Specifications:**
```yaml
CPU: 32 cores (Intel Xeon Gold or AMD EPYC)
RAM: 128GB DDR4
GPU: 2x NVIDIA A100 (40GB VRAM each) or 2x H100
Storage: 2TB NVMe SSD (RAID 1)
Network: 10 Gbps (air-gapped, internal)
```

**For Different Scales:**

| Scale | Documents | GPU | RAM | Storage |
|-------|-----------|-----|-----|---------|
| **Small** | <100K | 1x RTX 4090 | 64GB | 500GB |
| **Medium** | 100K-1M | 1x A100 40GB | 128GB | 1TB |
| **Large** | 1M-10M | 2x A100 80GB | 256GB | 2TB |
| **Enterprise** | >10M | 4x H100 | 512GB | 4TB |

---

## 🔒 SECURITY & MONITORING

### Security Stack

**Authentication & Authorization:**
```yaml
JWT: PyJWT or python-jose
OAuth2: FastAPI OAuth2
MFA: pyotp (TOTP)
Encryption: cryptography library
HSM Integration: PyKCS11
```

**Secrets Management:**
```yaml
Environment Variables: python-dotenv
Vault: hvac (HashiCorp Vault client)
Config Encryption: Fernet (cryptography)
```

### Monitoring Stack

**Application Monitoring:**
```yaml
Prometheus: prometheus-client (Python)
Grafana: Data visualization
Logging: Loguru or Python logging
Distributed Tracing: OpenTelemetry
```

**System Monitoring:**
```yaml
CPU/GPU: psutil, GPUtil
Resource Usage: nvidia-smi
Process Monitoring: supervisor or systemd
```

**Example Prometheus Integration:**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
latency_histogram = Histogram('rag_latency_seconds', 'Query latency')
gpu_memory = Gauge('gpu_memory_used_mb', 'GPU memory used')

@latency_histogram.time()
def process_query(query):
    query_counter.inc()
    # ... RAG processing
    
    # Update GPU metric
    import GPUtil
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_memory.set(gpus[0].memoryUsed)
    
    return result

# Start metrics server
start_http_server(9090)
```

**Logging Configuration:**
```python
from loguru import logger
import sys

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

logger.add(
    "/var/log/isro_rag/app_{time}.log",
    rotation="500 MB",
    retention="30 days",
    compression="zip",
    level="DEBUG"
)

# Usage
logger.info("Query received: {query}", query=user_query)
logger.error("Retrieval failed: {error}", error=str(e))
```

---

## 🧪 TESTING & EVALUATION

### Testing Frameworks

**Unit Testing:**
```yaml
Framework: pytest
Coverage: pytest-cov
Mocking: pytest-mock
Async: pytest-asyncio
```

**Example Tests:**
```python
import pytest
from src.rag_system import ISRORAGSystem

@pytest.fixture
def rag_system():
    return ISRORAGSystem()

def test_retrieval(rag_system):
    query = "What is PSLV payload capacity?"
    results = rag_system.retrieve(query, k=10)
    
    assert len(results) == 10
    assert all(hasattr(r, 'score') for r in results)
    assert results[0].score >= results[-1].score  # Sorted by score

@pytest.mark.asyncio
async def test_generation(rag_system):
    query = "What is PSLV?"
    context = ["PSLV is a launch vehicle..."]
    
    response = await rag_system.generate(query, context)
    
    assert len(response) > 0
    assert "PSLV" in response
```

### Evaluation Libraries

**RAG Evaluation:**
```yaml
RAGAS: ragas library
BEIR: beir library
Custom Metrics: Own implementation
```

**Example Evaluation:**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    'question': questions,
    'answer': generated_answers,
    'contexts': retrieved_contexts,
    'ground_truths': reference_answers
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)
# Output: {'faithfulness': 0.95, 'answer_relevancy': 0.87, ...}
```

---

## 📊 COMPLETE STACK SUMMARY

### Production Stack (Recommended)

```yaml
# ============================================
# CORE AI/ML STACK
# ============================================
Embedding Model: BAAI/bge-m3
LLM: meta-llama/Meta-Llama-3.1-8B-Instruct
Reranker: BAAI/bge-reranker-v2-m3
Vector Database: FAISS
Lexical Search: BM25 (rank-bm25 library)

# ============================================
# DOCUMENT PROCESSING
# ============================================
PDF: PyMuPDF (fitz)
OCR: DocTR + Tesseract
DOCX: python-docx
XLSX: pandas + openpyxl
Multi-format: Unstructured

# ============================================
# RAG FRAMEWORK
# ============================================
Primary: Custom Python (for production control)
Prototyping: LlamaIndex
Utilities: LangChain (for specific components)

# ============================================
# INFRASTRUCTURE
# ============================================
Programming: Python 3.11+
Inference: vLLM
API: FastAPI + Uvicorn
Database: PostgreSQL 15 + pgvector
Containerization: Docker + Docker Compose
Queue: Redis (for job queuing)

# ============================================
# SECURITY
# ============================================
Authentication: OAuth2 + JWT
Encryption: cryptography library
MFA: pyotp
Secrets: HashiCorp Vault or AWS Secrets Manager (offline)
Audit: PostgreSQL + custom logging

# ============================================
# MONITORING & OBSERVABILITY
# ============================================
Metrics: Prometheus
Visualization: Grafana
Logging: Loguru
Tracing: OpenTelemetry
Alerting: Prometheus Alertmanager

# ============================================
# DEVELOPMENT & TESTING
# ============================================
Testing: pytest
Code Quality: black + isort + flake8
Type Checking: mypy
CI/CD: GitHub Actions or GitLab CI
Version Control: Git
```

### Alternative Lightweight Stack

**For Smaller Deployments:**

```yaml
Embedding: BGE-Large-EN-v1.5 (smaller, English-only)
LLM: Mistral-7B-Instruct (smaller model)
Vector DB: ChromaDB (simpler setup)
Framework: LlamaIndex (easier to use)
Deployment: Single Docker container
Database: SQLite (for small scale)
Monitoring: Basic Python logging
```

---

## 🗺️ IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-2)

**Objectives:**
- Set up development environment
- Install core dependencies
- Download and test models

**Tasks:**
```bash
# Week 1: Environment Setup
1. Install Python 3.11
2. Create virtual environment
3. Install core ML libraries (torch, transformers)
4. Download BGE-M3 embedding model
5. Download Llama 3.1-8B model
6. Test basic inference

# Week 2: Core Components
1. Implement embedding pipeline
2. Set up FAISS vector store
3. Implement BM25 indexing
4. Test retrieval performance
5. Document processing pipeline (PDF, DOCX)
```

**Deliverables:**
- Working development environment
- Basic retrieval system (vector + BM25)
- Document parsing pipeline

### Phase 2: RAG Pipeline (Weeks 3-4)

**Objectives:**
- Build end-to-end RAG system
- Implement reranking
- Optimize prompts

**Tasks:**
```bash
# Week 3: Retrieval Pipeline
1. Implement hybrid retrieval (dense + sparse)
2. Add BGE-reranker integration
3. Build reciprocal rank fusion
4. Optimize retrieval parameters
5. Add metadata filtering

# Week 4: Generation Pipeline
1. Set up vLLM for inference
2. Design prompt templates
3. Implement response synthesis
4. Add citation generation
5. Test end-to-end pipeline
```

**Deliverables:**
- Complete RAG pipeline
- Prompt templates for different domains
- Citation system

### Phase 3: Domain Adaptation (Weeks 5-6)

**Objectives:**
- Adapt system for ISRO domains
- Implement domain-specific optimizations

**Tasks:**
```bash
# Week 5: Technical Domain
1. Ingest technical documentation
2. Fine-tune chunking for specs
3. Test on technical queries
4. Optimize for accuracy

# Week 6: Administrative & QA Domains
1. Ingest GFR, procurement docs
2. Ingest QA reports
3. Domain-specific prompt templates
4. Cross-domain testing
```

**Deliverables:**
- Domain-specific configurations
- Ingested ISRO documentation
- Domain evaluation results

### Phase 4: Security & Deployment (Weeks 7-8)

**Objectives:**
- Implement security features
- Deploy offline system

**Tasks:**
```bash
# Week 7: Security Implementation
1. Add authentication (MFA)
2. Implement RBAC
3. Add encryption (at rest, in transit)
4. Implement audit logging
5. Security testing

# Week 8: Deployment
1. Create Docker containers
2. Set up PostgreSQL
3. Configure monitoring (Prometheus, Grafana)
4. Deploy to air-gapped environment
5. User acceptance testing
```

**Deliverables:**
- Secured RAG system
- Deployed offline system
- Monitoring dashboards
- User documentation

### Phase 5: Evaluation & Optimization (Weeks 9-10)

**Objectives:**
- Comprehensive evaluation
- Performance optimization

**Tasks:**
```bash
# Week 9: Evaluation
1. Create test dataset (1000+ queries)
2. Run RAGAS evaluation
3. Calculate all metrics
4. Identify failure modes
5. Expert review sessions

# Week 10: Optimization
1. Fix identified issues
2. Fine-tune parameters
3. Optimize inference speed
4. Final testing
5. Documentation
```

**Deliverables:**
- Evaluation report
- Optimized system
- Complete documentation
- Training materials

---

## 📦 INSTALLATION GUIDE

### Complete Setup Script

```bash
#!/bin/bash
# ISRO RAG System Setup Script

# 1. Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    postgresql-15 \
    postgresql-contrib \
    redis-server \
    nvidia-cuda-toolkit

# 3. Create project directory
echo "Creating project structure..."
mkdir -p ~/isro-rag
cd ~/isro-rag

# 4. Create virtual environment
echo "Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# 5. Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Download models
echo "Downloading AI models..."
python -c "
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Embedding model
print('Downloading BGE-M3...')
SentenceTransformer('BAAI/bge-m3')

# Reranker
print('Downloading BGE-Reranker...')
AutoModel.from_pretrained('BAAI/bge-reranker-v2-m3')

# LLM
print('Downloading Llama 3.1...')
AutoModelForCausalLM.from_pretrained(
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    torch_dtype='float16'
)
"

# 7. Set up PostgreSQL
echo "Setting up PostgreSQL..."
sudo -u postgres psql <<EOF
CREATE DATABASE isro_rag;
CREATE USER rag_admin WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE isro_rag TO rag_admin;
\c isro_rag
CREATE EXTENSION vector;
EOF

# 8. Create directory structure
mkdir -p data/{documents,indices,models,logs}
mkdir -p configs
mkdir -p tests

echo "Setup complete!"
echo "Activate environment with: source venv/bin/activate"
```

---

## ✅ SUCCESS CHECKLIST

### Before Deployment

- [ ] All models downloaded and tested
- [ ] FAISS indices created and optimized
- [ ] PostgreSQL database configured with pgvector
- [ ] Authentication system implemented and tested
- [ ] Encryption configured (at rest and in transit)
- [ ] Audit logging operational
- [ ] Monitoring dashboards set up (Grafana)
- [ ] Alert system configured
- [ ] API endpoints secured
- [ ] Rate limiting implemented
- [ ] Backup system configured
- [ ] Disaster recovery plan documented
- [ ] User documentation complete
- [ ] Admin documentation complete
- [ ] Test dataset created (1000+ queries)
- [ ] Evaluation metrics baseline established
- [ ] Security penetration testing completed
- [ ] Performance benchmarking done
- [ ] Offline operation verified
- [ ] Air-gapped deployment tested

---

## 📞 SUPPORT & RESOURCES

### Official Documentation

**Models:**
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- Llama 3.1: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- Qwen 2.5: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

**Libraries:**
- LlamaIndex: https://docs.llamaindex.ai
- LangChain: https://python.langchain.com
- FAISS: https://faiss.ai
- vLLM: https://docs.vllm.ai

**Evaluation:**
- RAGAS: https://docs.ragas.io
- BEIR: https://github.com/beir-cellar/beir

---

**Document Version:** 1.0  
**Last Updated:** February 10, 2026  
**Recommended Review:** Quarterly

*This technology stack is designed for ISRO VSSC's secure, offline RAG deployment.*
