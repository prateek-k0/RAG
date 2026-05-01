# Simple RAG

A Node.js and TypeScript implementation of a local, two-step Retrieval-Augmented Generation (RAG) pipeline for PDF documents.

The app loads PDF files, chunks them with section context, retrieves relevant chunks through hybrid search, and answers questions with a locally running Llama 3.1 8B model through Ollama. It also keeps chat history per session and summarizes longer conversations before injecting them back into the prompt.

## Features

- Local Llama 3.1 chat model through Ollama.
- Local `nomic-embed-text` embeddings through Ollama.
- PDF directory loading with recursive folder support.
- Section-aware chunking that preserves detected section titles in chunk text and metadata.
- Multi-query retrieval that rewrites each user question into three targeted search queries.
- Hybrid retrieval using FAISS semantic search and BM25 keyword search.
- In-memory message history with automatic summarization after the configured watermark.
- Streaming command-line chat interface.

## How It Works

The project follows a two-step RAG flow:

1. Retrieve context from PDF documents.
2. Generate an answer using the retrieved context and chat history.

At runtime, `src/index.ts` performs the setup:

1. Loads PDF documents from `SOURCES_PATH`.
2. Splits pages into chunks with `SectionAwareChunker`.
3. Builds a `HybridRetriever` over the chunks.
4. Creates the Llama RAG runnable chain.
5. Starts an interactive terminal prompt.

For each question, the chain:

1. Generates three search-focused query variations.
2. Runs hybrid retrieval for each generated query.
3. Deduplicates matching chunks.
4. Formats retrieved documents as prompt context.
5. Summarizes longer message history.
6. Calls Llama 3.1 and streams the answer back to the terminal.

## Architecture

```text
PDF files
  -> Directory PDF loader
  -> Section-aware chunker
  -> Hybrid retriever
       -> FAISS vector search with Ollama embeddings
       -> BM25 keyword search
  -> Multi-query retrieval
  -> Prompt with retrieved context + summarized chat history
  -> Local Llama 3.1 answer
```

## Project Structure

```text
src/
  chunkers/
    contextAwareTextChunker.ts       Section-aware document chunking
  documentLoaders/
    directoryPDFLoader.ts            Recursive PDF directory loader
  messageStore/
    createMemoryHistoryChain.ts      In-memory chat history wrapper
  models/
    llama-rag-chain.ts               Main RAG chain and prompts
    messageHistorySummarizer.ts      Chat history summarizer
  searchers/
    hybrid.ts                        FAISS + BM25 ensemble retriever
  utils/
    constants.ts                     Shared chain input keys
  index.ts                           CLI entry point
```

## Prerequisites

- Node.js with npm.
- Ollama installed and running locally.
- The required local models pulled into Ollama:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Start Ollama before running the app:

```bash
ollama serve
```

If Ollama is already running as a background service, you do not need to start it manually.

## Setup

Install dependencies:

```bash
npm run prepare
```

Create or update `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLAMA_BASE_URL=http://localhost:11434
SOURCES_PATH=../../sources
DEBUG_PATH=../../logs
```

`SOURCES_PATH` is resolved relative to the loader module at runtime. With the default value, place PDFs in:

```text
RAG/simple-rag/sources/
```

The loader supports nested folders and loads `.pdf` files recursively.

## Running

Build the TypeScript project:

```bash
npm run build
```

Start the CLI:

```bash
npm start
```

The app will:

1. Load PDFs from the configured source folder.
2. Create section-aware chunks.
3. Build the FAISS and BM25 retrievers in memory.
4. Start an interactive prompt.

Example:

```text
Enter your prompt: What was Tesla's automotive gross margin?
```

Answers are streamed to the terminal, and the same session keeps message history across follow-up questions.

## Retrieval Details

`HybridRetriever` combines two retrieval strategies:

- FAISS vector search for semantic similarity.
- BM25 keyword search for exact term matching.

The current setup retrieves five vector matches and five BM25 matches, then combines them with weights:

```ts
[0.3, 0.7]
```

This favors keyword precision while still allowing semantic matches from the vector store.

## Section-Aware Chunking

`SectionAwareChunker` detects uppercase section headers in PDF text and keeps the section title attached to the chunk. This helps preserve context from financial reports and other structured PDFs where tables, notes, and section labels are important.

If no section headers are found, the chunker falls back to recursive character splitting.

The current entry point uses:

```ts
new SectionAwareChunker(400, 100)
```

That means chunks target 400 characters with 100 characters of overlap.

## Message History

The RAG chain is wrapped with `RunnableWithMessageHistory` and stores messages in memory by `sessionId`.

When the message history grows beyond the summarizer watermark, `messageHistorySummarizer.ts` summarizes the conversation and injects the summary into the next prompt. This keeps follow-up questions contextual without sending an ever-growing transcript to the model.

The default CLI session id is:

```text
session1
```

## TODO

- Use Redis or another database-backed store for persistent message history.
- Replace the built-in `RunnableWithMessageHistory` with a custom message storage and update mechanism inside the chain lifecycle.

## Notes

- The FAISS index is built in memory each time the app starts.
- Chat history is also in memory and resets when the process exits.
- The main prompt is tuned for financial PDF reports and includes rules for handling imperfect PDF table extraction.
- The current `src/index.ts` metadata example is tailored to a Tesla Q4 2025 earnings report. Update that metadata if you use a different document set.
