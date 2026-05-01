import "dotenv/config";
import readline from "readline";
import { loadTextDocuments } from "./documentLoaders/directoryPDFLoader.js";
import { SectionAwareChunker } from "./chunkers/contextAwareTextChunker.js";
import { HybridRetriever } from "./searchers/hybrid.js";
import { createLlamaRagChainWithRunnables } from "./models/llama-rag-chain.js";
import ora from "ora";
import { Runnable } from "@langchain/core/runnables";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const spinner = ora();

async function runner(chain: Runnable<any>, sessionId?: string) {
  rl.question("Enter your prompt: ", async (promptInput) => {
    // 5. Run the RAG chain
    spinner.start("Thinking");
    const resultStream = await chain.stream({
      input: promptInput,
    }, { configurable: { sessionId } });
    spinner.succeed("Done thinking. Streaming response...");
    // 6. Stream the result
    // spinner.start("Streaming response...");
    await resultStream.pipeTo(
      new WritableStream({
        write: async (chunk) => {
          rl.write(chunk);
        },
        close: () => {
          rl.write("\n-------\n\n");
        },
      }),
    );
    // spinner.succeed("Result streamed");
    runner(chain, sessionId);
  });
}

async function main() {
  try {
    // 1. Load the documents
    spinner.start("Loading documents...");
    const documents = await loadTextDocuments(process.env.SOURCES_PATH ?? "");
    spinner.succeed("Documents loaded");
    const sectionAwareChunker = new SectionAwareChunker(400, 100);
    spinner.start("Chunking documents...");
    const chunks = await sectionAwareChunker.chunk(documents, {
      context: "TESLA Q4-2025 Earnings Report",
      // year: "2025",
      // quarter: "Q4",
      // company: "TESLA",
      // ticker: "TSLA",
      report_type: "Earnings Report",
    });
    // console.log(chunks.map(chunk => chunk.metadata));
    // log first 3 chunks to a json file
    // fs.writeFile(path.join(import.meta.dirname, '../logs/chunks.json'), JSON.stringify(chunks.slice(0, 3), null, 2), 'utf-8');
    spinner.succeed("Chunks created");
    // 3. Create the hybrid search store
    const hybridStore = new HybridRetriever(chunks);
    spinner.start("Setting up hybrid retriever...");
    await hybridStore.setupStore();
    const hybridRetriever = await hybridStore.getRetriever(5, 5, [0.3, 0.7]);
    spinner.succeed("Hybrid retriever setup");
    // 4. Create the RAG chain
    spinner.start("Creating RAG chain... and finishing setup");
    const retrieverChain = await createLlamaRagChainWithRunnables(hybridRetriever);
    spinner.succeed("RAG setup complete");
    rl.write('--------------------------\n');
    await runner(retrieverChain, "session1");
  } catch (error) {
    console.error("❌ Error:", error);
    spinner.fail("Error occurred");
  }
}

main();
