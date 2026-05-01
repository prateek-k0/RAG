import { OllamaEmbeddings } from "@langchain/ollama";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { EnsembleRetriever } from "@langchain/classic/retrievers/ensemble";
import { Document } from "@langchain/core/documents";
import { VectorStoreRetriever } from "@langchain/core/vectorstores";

export class HybridRetriever {
  private vectorStoreRetriever!: VectorStoreRetriever<FaissStore>;
  private bm25Retriever!: BM25Retriever;
  private chunks!: Document<Record<string, any>>[];

  constructor(chunks: Document<Record<string, any>>[]) {
    this.chunks = chunks;
  }

  async setupStore() {
    // 1. setup ollama embeddings
    const embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
    });

    // 2. Setup Vector Store Retriever (Semantic Search)
    // This turns your chunks into numbers and stores them in RAM via FAISS
    const vectorStore = await FaissStore.fromDocuments(this.chunks, embeddings);
    this.vectorStoreRetriever = vectorStore.asRetriever({
      k: 10
    });

    // 3. setup BM25 Retriever (Keyword Search)
    this.bm25Retriever = BM25Retriever.fromDocuments(this.chunks, { k: 10 });
  }

  async getRetriever(
    k_vector: number,
    k_bm25: number,
    weights: [number, number] = [0.3, 0.7]
  ) {
    if(!this.vectorStoreRetriever || !this.bm25Retriever) {
      await this.setupStore();
    }
    this.vectorStoreRetriever.k = k_vector;
    this.bm25Retriever.k = k_bm25;
    return new EnsembleRetriever({
      retrievers: [this.vectorStoreRetriever, this.bm25Retriever],
      weights: weights,
    });
  }
}