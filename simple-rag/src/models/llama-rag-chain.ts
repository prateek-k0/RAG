import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate, MessagesPlaceholder, PromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";

import { BaseRetriever } from "@langchain/core/retrievers";
import { EnsembleRetriever } from "@langchain/classic/retrievers/ensemble";
import { MultiQueryRetriever } from "@langchain/classic/retrievers/multi_query";
import { INPUT_KEY, MESSAGE_HISTORY_KEY } from "../utils/constants";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda, RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { Document } from "@langchain/core/documents";
import { summarizeMessages } from "./messageHistorySummarizer";
// import fs from 'fs';
// import path from 'path';
import 'dotenv/config';
import { createMemoryHistoryChain } from "../messageStore/createMemoryHistoryChain";

// DEPRECATED: no more black-box built-ins like MultiQueryRetriever
export async function createLlamaRagChain(retriever: BaseRetriever | EnsembleRetriever) {
  // 1. Initialize our local Llama 3.1
  const model = new ChatOllama({
    model: "llama3.1",
    temperature: 0,
    baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
    numCtx: 8192, // 8k context window
    verbose: true,
  });

  // 1.5. Create the Multi Query Retriever
  // first, set prompt to generate different ways to ask the same thing
  /*
    the Standard Setup (and the one most likely to work with Llama 3.1 without errors) is to keep the 
    variable as {question} inside the MultiQueryRetriever template, while keeping your main chain 
    input as {input}.
  */
  const CUSTOM_MULTI_QUERY_PROMPT = new PromptTemplate({
    inputVariables: ["question"],
    template: `You are an AI language model assistant. Your task is to generate 
    three (3) different versions of the given user question to retrieve relevant 
    documents from a vector database. By generating multiple perspectives on the 
    user question, your goal is to help the user overcome some of the limitations 
    of distance-based similarity search.
    
    Provide ONLY the alternative questions, separated by a newline. Do not include introductory text or numbering.
    Original question: {question}`
  });

  // before the actual retrieval, generate multiple queries to retrieve more relevant documents
  const multiQueryRetriever = MultiQueryRetriever.fromLLM({
    llm: model, // Your Llama 3.1 instance
    retriever: retriever, // maybe a hybrid retriever
    // verbose: true,
    prompt: CUSTOM_MULTI_QUERY_PROMPT,
  });

  // 2. Create the RAG Prompt
  // This tells the AI to ONLY use the provided context to answer.
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a financial analyst. Use the following context to answer the user's question. If you don't know, say you don't know.\n\nContext:\n{context}"],
    ["human", INPUT_KEY],
  ]);

  // 3. Create the "Stuff" Chain
  // This step defines HOW the documents are formatted into the prompt.
  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt
  });

  // 4. Create the final Retrieval Chain
  // This connects your Hybrid Search logic to the Document Chain.
  const retrieverChain = await createRetrievalChain({
    retriever: multiQueryRetriever,
    combineDocsChain,
  });

  return retrieverChain;
}

// much more intuitive, no more black-box built-ins to use
export async function createLlamaRagChainWithRunnables(retriever: BaseRetriever | EnsembleRetriever) {
  // 1. Initialize our local Llama 3.1
  const model = new ChatOllama({
    model: "llama3.1",
    temperature: 0,
    baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
    numCtx: 8192, // 8k context window
    // verbose: true,
  });

  // 2. create a multi query generator runnable
  const queryGenPrompt = ChatPromptTemplate.fromTemplate(`
    You are a Senior Equity Analyst. Generate 3 specific search queries to find exact 
    numerical data in the Tesla Q4 2025 report for the question: "{${INPUT_KEY}}".
    Focus on technical terms like 'GAAP', 'Automotive Gross Margin', 'Free Cash Flow', 
    and 'EBITDA'. Output only the queries. Provide ONLY the queries, separated by a newline, no numbering, no preamble.
  `);
  // make sure to retain user's input as the input variable
  const queryGenerator = RunnableSequence.from([
    queryGenPrompt,
    model,
    new StringOutputParser(),
    RunnableLambda.from(async (output: string) => output.trim()),
    // to get an array of 3 query strings from the output
    RunnableLambda.from(async (output: string) => output.split('\n').filter((line) => line.trim().length > 0).slice(0, 3)),
  ]);

  // 3. with the multiple queries generated, run the hybrid search for each query in parallel
  const retrieverChain = RunnableLambda.from(async (queries: string[]) => {
    // console.log("🔍 Generated Variations:", queries);
    // 3.1 Run Hybrid Search for EVERY query in parallel
    const searchResults = await Promise.all(
      queries.map((q) => retriever.invoke(q))
    );
    // 3.2 Combine and deduplicate based on pageContent
    const flattenedResults = searchResults.flat();
    const deduplicatedResults = Array.from(
      new Map(flattenedResults.map((r) => [r.pageContent, r])).values()
    );
    // // 3.3 [optional] Save results to a JSON file for debugging
    // const filePath = path.join(import.meta.dirname, process.env.DEBUG_PATH ?? '', 'hybrid_search_results.json');
    // queueMicrotask(() => {
    //   fs.writeFileSync(filePath, JSON.stringify(deduplicatedResults, null, 2), 'utf-8');
    // });
    return deduplicatedResults;
  });
  // 3.4: format the documents:
  const formatDocs = (docs: Document<Record<string, any>>[]) => docs.map(d => `[SOURCE: ${d.metadata.sourceFile}]\n${d.pageContent}`).join("\n\n");

  // 4. create a chain to combine the documents into a prompt
  const chainPrompt = ChatPromptTemplate.fromMessages([
    ["system", `
      ### SYSTEM ROLE
      You are a Senior Financial Auditor. Your task is to extract precise data from company earnings reports. The provided context is extracted from PDFs and may contain "text soup" (where paragraphs and table rows are mixed) or "squashed numbers" (where spaces between columns are missing). Also, you may use the previous message history to get more context about the user's question.

      ### DATA DECODING RULES
      1. **The Row-Logic Rule:** If a line contains a label followed by a string of numbers (e.g., "Revenue10,50012,000"), treat the numbers as a chronological sequence. Usually, the last number in the sequence represents the most recent quarter/year mentioned in the header.
      2. **The "Z-Pattern" Awareness:** If a sentence seems to be interrupted by financial data or location names (e.g., "The margin improvedShanghaiModel YProduction"), ignore the interruptions and reconstruct the narrative. The interrupted data likely belongs to a table that was positioned to the right of the paragraph.
      3. **The Footnote Anchor:** Always check the end of a section for labels like "(1)", "(2)", or "Note:". These often contain critical context (e.g., "Excludes regulatory credits") that changes the meaning of the numbers.
      4. **The Scale Check:** Verify if the context specifies "($ in millions)" or "in thousands". Always include the unit in your final answer.

      ### RESPONSE GUIDELINES
      - If the user asks for a specific metric (e.g., Gross Margin), look for the row that matches that exact term. 
      - If multiple numbers appear for the same metric, default to the most recent period unless otherwise specified.
      - If the data is truly ambiguous due to formatting, provide your best estimate but add a "Data Quality Note" explaining the ambiguity.
      - If you don't know, say you don't know. Do not hallucinate numbers.

      ### CONTEXT:
      {context}
    `],
    new MessagesPlaceholder(MESSAGE_HISTORY_KEY), // IMPORTANT: this is where the message history will be injected
    ["human", `{${INPUT_KEY}}`],
  ]);
  // TODO: use your own message store helpers (not the built-in RunnableWithMessageHistory)
  const ragChain = RunnableSequence.from([
    // 1. assign the summarized message history to the input
    RunnablePassthrough.assign({
      [MESSAGE_HISTORY_KEY]: RunnableLambda.from(async (chainInput: any, chainConfig: any) => {
        // console.log("🔍 Chain Config:", chainConfig);
        return await summarizeMessages(chainInput[MESSAGE_HISTORY_KEY])
      }) 
    }),
    // 2. assign the context to the input
    RunnablePassthrough.assign({
      context: RunnableSequence.from([
        queryGenerator,
        retrieverChain,
        formatDocs,
      ]),
    }),
    // OR, we can use RunnableParallel to execute all the runnables (input, context and history ) in parallel
    // create multiple queries, and fetch chunks from the retriever, and format them, while letting the input pass through as is
    // RunnableParallel will automatically invoke all the runnables inside the object in parallel, and return the result as an object.
    // When LangChain sees an object, it secretly wraps it in a RunnableParallel class.\
    // RunnableParallel.from({
    //   context: RunnableSequence.from([
    //     queryGenerator,
    //     retrieverChain,
    //     formatDocs,
    //   ]),
    //   [INPUT_KEY]: RunnableLambda.from((chainInput: any) => chainInput[INPUT_KEY]),
    //   [MESSAGE_HISTORY_KEY]: RunnableLambda.from((chainInput: any) => chainInput[MESSAGE_HISTORY_KEY]),
    // }),
    
    // [optional] log the chain input
    // RunnableLambda.from(async (chainInput: any) => {
    //   console.log("🔍 Chain Input:", chainInput);
    //   return chainInput;
    // }),
    // 3. combine the documents into a prompt
    chainPrompt,
    // 4. call the model
    model,
    // 5. parse the output
    new StringOutputParser()
  ]);

  return createMemoryHistoryChain(ragChain);
}