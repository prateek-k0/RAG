/**
 * Short term memory:
 * Short term memory lets your application remember previous interactions 
 * within a single thread or conversation.
 * 
 * Short-term memory is the agent's working memory or scratchpad. It is completely 
 * isolated to the current conversation or execution pipeline.
 * 
 * - What it stores: The chronological back-and-forth chat history (HumanMessage, 
 *   AIMessage, ToolMessage), current loop counters, intermediate tool data, and mid-flight variables.
 * - How it maps to LangGraph: This is exactly what we manage using Threads and the Checkpointer Engine.
 *   When you pass a thread_id, the checkpointer restores the short-term state of that specific chat session.
 * - Lifetime: It lasts for the duration of that specific conversation thread. If you switch to a 
 *   brand-new thread_id, your short-term memory completelyresets to blank.
 * 
 * Short term memory is bound to a single Thread ID.
 * Its purpose is managing immediate context for the current task.
 * Example: "The user just asked for the weather in Paris."
 */

// for this example, lets look at a small llm based graph with short term memory retained.

import { StateGraph, MessagesAnnotation, START, END, MemorySaver } from "@langchain/langgraph";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import "dotenv/config";
import { ChatOllama } from "@langchain/ollama";

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.5,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192, // 8k context window
});

const checkpointer = new MemorySaver();   // used to store the short term memory

const llmNode = async (state: typeof MessagesAnnotation.State) => {
  const response = await llmModel.invoke(state.messages);
  return {
    messages: [response]
  }
}

const workflowGraph = new StateGraph(MessagesAnnotation)
  .addNode("llm", llmNode)
  .addEdge(START, "llm")
  .addEdge("llm", END);

const compiledWorkflow = workflowGraph.compile({
  checkpointer: checkpointer  // here we are passing the checkpointer to the compiled workflow
});

export const runWorkflowWithShortTermMemory = async () => {
  const configWithThreadId = { configurable: { thread_id: '123' } }
  let finalStateOutput = await compiledWorkflow.invoke({
    messages: [new HumanMessage("Hi! My name is Bob.")],
  }, configWithThreadId);
  console.log(finalStateOutput.messages[finalStateOutput.messages.length - 1]?.content ?? 'No response');
  console.log('--------------------------------');
  finalStateOutput = await compiledWorkflow.invoke({
    messages: [new HumanMessage("What is my name?")],
  }, configWithThreadId);
  console.log(finalStateOutput.messages[finalStateOutput.messages.length - 1]?.content ?? 'No response');
}