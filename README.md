# Janhit AI — Legal RAG with Memory
## This was made for Convolve 4.0 Hackathon (refer to setup.txt for setup)

## Problem

Accessing and understanding legal information in India is **difficult, fragmented, and risky**.

- Legal documents (FIRs, forms, notices) are **complex and technical**
- Citizens often don’t know **how to fill documents correctly** or **what to be careful about**
- Existing chatbots **hallucinate legal advice** without grounding it in documents
- Legal queries are usually **conversational and follow-up based**, not single-shot
- There is **no persistence of context or memory** across interactions

This leads to:
- Incorrect or incomplete filings  
- Legal delays and repeated visits to authorities  
- Dependence on intermediaries for basic legal clarity  

---

## Solution

**Janhit AI** is a **document-grounded Legal Assistant** that combines:

- Retrieval-Augmented Generation (RAG)
- Long-term conversational memory
- Structured query decomposition
- Evidence-based response generation

The system ensures that **every answer is grounded in retrieved legal documents**, not assumptions.

---
## Demo Video

**Demo Link:** https://drive.google.com/file/d/1RwIwQzGGGb3f8eqqh5Ibsnt3EB7E2Pv7/view?usp=drive_link
---

## How It Works

### 1. Document Ingestion
- Users attach legal documents (PDFs, scans)
- OCR + parsing extracts structured text
- Documents are chunked and embedded
- Embeddings are stored in **Qdrant**

### 2. Query Understanding
- User queries are analyzed and decomposed into semantic sub-questions
- Conversational or follow-up queries gracefully fall back to a single-topic flow
- No hard failures on ambiguous input

### 3. Retrieval + Memory
- Relevant document chunks are retrieved from Qdrant
- Long-term session memory is queried to maintain context
- Document filters ensure answers stay scoped to the correct source

### 4. Grounded Generation
- The LLM generates responses **only from retrieved evidence**
- Outputs include actionable guidance and caution points
- Hallucinations are actively avoided

### 5. Persistent Memory
- User queries and assistant responses are stored as embeddings
- Future questions benefit from past context automatically

---

## Architecture Highlights

- **LangGraph** for agent orchestration
- **Qdrant** for vector search and memory
- **BGE-M3 embeddings** for semantic + hybrid retrieval
- Modular agents for ingestion, retrieval, generation, and safety

---

## Guiding Principle

> **“जनहित में जारी — सही जानकारी, सही समय पर”**  
(Released in public interest — the right information, at the right time.)

---

## Disclaimer

Janhit AI is an **information and assistance system**, not a replacement for professional legal counsel. It is designed to improve clarity, awareness, and correctness in legal interactions.

