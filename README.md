# SPI Log Analyzer

An AI-assisted SPI communication diagnosis system that analyzes SPI capture logs, extracts protocol-level features, identifies communication failures, retrieves similar historical captures using vector similarity, and generates structured diagnostic explanations with a local LLM.

## Overview

Debugging SPI communication failures often requires manually inspecting clock timing, frame length, response bytes, signal transitions, and protocol behavior.

This project automates that workflow.

The system combines:

- SPI waveform feature extraction
- Deterministic failure diagnosis
- Sentence-transformer embeddings
- ChromaDB vector retrieval
- Retrieval-Augmented Generation (RAG)
- Local LLM-based diagnostic explanations
- Structured JSON output
- Automated evaluation

The current implementation uses synthetic SPI captures covering 12 failure classes.

---

## System Architecture

```text
                    SPI Capture CSV
                          |
                          v
                 +------------------+
                 | Feature Extractor|
                 +------------------+
                          |
                          v
                 +------------------+
                 | SPI Features     |
                 +------------------+
                    /           \
                   /             \
                  v               v
       +----------------+   +------------------+
       | Deterministic  |   | Sentence         |
       | Diagnosis      |   | Transformer      |
       +----------------+   | Embeddings       |
                  |          +------------------+
                  |                   |
                  |                   v
                  |          +------------------+
                  |          | ChromaDB         |
                  |          | Vector Search    |
                  |          +------------------+
                  |                   |
                  |                   v
                  |          Similar Historical
                  |             SPI Captures
                  |                   |
                  +---------+---------+
                            |
                            v
                  +------------------+
                  | LLM Analyzer     |
                  | Ollama / Gemini  |
                  +------------------+
                            |
                            v
                  +------------------+
                  | Structured       |
                  | Diagnosis        |
                  +------------------+
                     /       |       \
                    v        v        v
                 Failure   Root Cause  Checks
                  Type