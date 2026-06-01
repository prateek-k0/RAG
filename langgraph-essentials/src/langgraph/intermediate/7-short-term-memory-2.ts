// StateGraph version of the same example
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