/**
 * This example is about message history, and how we can use it to trim it, if it crosses
 * a certain number of messages. IRL, we need to count it in terms of context, and would want to
 * summarize the "not needed" messages, instead of removing them all together.
 */

import { StateGraph, MessagesAnnotation, START, END, MemorySaver } from "@langchain/langgraph";
import { ChatOllama } from "@langchain/ollama";
import { RemoveMessage, HumanMessage, AIMessage, BaseMessage } from "@langchain/core/messages";

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.5,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192
});

const CUT_OFF_MESSAGES = 4; // cut off after 4 messages

// lets add a "housekeeping" node at the start of the graph
// to trim the message history if it crosses a certain number of messages
const trimMessageHistoryNode = async (state: typeof MessagesAnnotation.State) => {
  // 1. Identify all the historic messages that need to be dropped
  // For example, if array has 8 messages, slice out the first 4 to delete them
  // remove the first length - CUT_OFF_MESSAGES messages
  const idsToDelete = state.messages.slice(0, -CUT_OFF_MESSAGES).map((message: BaseMessage) => {
    return new RemoveMessage({ id: message.id ?? '' });
  });
  // 3. Return the array of RemoveMessages. 
  // The upsert reducer reads these IDs and deletes them from the checkpointer.
  // if no ids to delete, return an empty object (so no update is made)
  return idsToDelete.length > 0 ? { messages: idsToDelete } : {};
}

const chatNode = async (state: typeof MessagesAnnotation.State) => {
  console.log(`🤖 [CHAT NODE] LLM Ingesting true trimmed history size: ${state.messages.length}`);
  const response = await llmModel.invoke(state.messages);
  return { messages: [response] };
};

const conversationGraph = new StateGraph(MessagesAnnotation)
  .addNode("memory_purger", trimMessageHistoryNode)
  .addNode("llm_chat", chatNode)
  .addEdge(START, "memory_purger")
  .addEdge("memory_purger", "llm_chat")
  .addEdge("llm_chat", END);
