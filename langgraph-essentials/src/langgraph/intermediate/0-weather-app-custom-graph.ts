/**
 * MessagesState:
 * LangGraph.js provides a pre-built schema called MessagesState. 
 * It is an object containing a single messages key, which is an array of LangChain 
 * BaseMessage objects (HumanMessage, AIMessage, ToolMessage).
 * 
 * it does 2 operations: 
 * 1. Append: If you return a new message, it appends it to the history array.
 * 2. Upsert/Overwrite by ID: If you return a message with an id that already exists in the state, 
 *    the reducer overwrites the old message with the new one. This is the exact primitive used to 
 *    support streaming token-by-token updates and message editing.
 * 
 * Tools:
 * An LLM cannot call a database, fetch the weather, or execute code natively. It can only generate text. 
 * A Tool is a structured wrapper around a standard JavaScript function that tells the LLM how and when 
 * to request its execution.
 * You define a tool using the tool() utility, giving it a name, a description, and a Zod schema for validation.
 */

import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOllama } from "@langchain/ollama";
import "dotenv/config";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage } from "@langchain/core/messages";

/**
 * While createReactAgent / createAgent (from langchain) is incredibly clean, it hides the mechanics. 
 * To build Agentic RAG workflows later (where you need to grade documents or rewrite 
 * search terms before making a tool call), you must know how to build a custom agent 
 * loop using explicit graph mechanics.
 */

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.2,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192, // 8k context window
});

// 1. Define the tool
const getWeatherTool = tool(
  // the actual function
  async ({ location }) => {
    // This is the actual execution code run by your server, in production, use an api or something
    return `The weather in ${location ?? 'Paris'} is 28°C and sunny.`;
  },
  // zod schema for validation
  {
    name: 'get_weather',
    description: 'Get the current weather in a given location',
    schema: z.object({
      location: z.string().describe('The city, e.g. San Francisco, Mumbai, etc.'),
    }),
  }
)

// 2. create the tool node
const tools = [getWeatherTool];
const weatherToolNode = new ToolNode(tools); // Prebuilt node that runs matching tools automatically

// 3. agent node
const callAgentNode = async (state: typeof MessagesAnnotation.State) => {
  // Bind tools to the model so it knows it has the options available
  const llmModelWithTools = llmModel.bindTools(tools);
  // Invoke the model with the state
  const response = await llmModelWithTools.invoke(state.messages);
  // Return the AI response message to append to MessagesState
  return {
    messages: [response]
  };
}

// 4. create conditional edge between the tool and the agent
const shouldCallTools = (state: typeof MessagesAnnotation.State) => {
  const lastMessage = state.messages[state.messages.length - 1];
  // if the last message is a tool_call, route it to the tool node
  if(
    lastMessage
    && AIMessage.isInstance(lastMessage)
    && (lastMessage.tool_calls ?? []).length > 0) {
      return "tools"
  }
  // else, response is complete, route to the next node
  return "end"
}

// 5. construct the graph
const weatherAgentWorkflow = new StateGraph(MessagesAnnotation);

weatherAgentWorkflow
.addNode("agent", callAgentNode)
.addNode("tools", weatherToolNode)
.addEdge(START, "agent")
.addConditionalEdges("agent", shouldCallTools, {
  tools: "tools",
  end: END,
})
.addEdge("tools", "agent")  // once the tool executes, route its response back to the agent


const customWeatherAgent = weatherAgentWorkflow.compile(); // compile the graph into a runnable object

// 6. execute the graph
export const runWeatherAgentWorkflow = async () => {
  const response = await customWeatherAgent.invoke({
    messages: [{ role: "user", content: "What is the weather like in Mumbai right now?" }]
  });
  console.log(response.messages[response.messages.length - 1]?.content ?? 'No response');
}