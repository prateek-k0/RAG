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

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
// both are fine, choose one
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { createAgent } from "langchain";
import "dotenv/config";

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.5,
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

/**
 * Crucial Distinction: When you pass this tool to an LLM, the LLM does not execute your function. 
 * The LLM simply inspects the description, and if it needs weather data, it stops generating text 
 * and outputs a structured JSON instruction called a tool_call. 
 * Your LangGraph framework catches this object, runs your local function with the provided
 * arguments, and feeds the response back to the LLM.
 */

/**
 * the ReAct pattern:
 * The standard loop pattern for agents is known as ReAct (Reasoning and Acting). 
 * It translates directly into a 2-node loop within a state graph:
 * 
 * 1. The Agent Node (Reasoning): The LLM receives the chat history. It decides whether it has
 *    the answer or needs to execute an action. If it needs an action, it outputs a tool_call
 *    and routes to the tool node.
 * 2. The Tools Node (Acting): The graph executes the JavaScript function matching the tool_call,
 *    appends the output to the message state as a ToolMessage, and loops directly back to the Agent Node.
 * 
 * The loop breaks only when the LLM decides it has enough context to reply with 
 * a standard text response instead of a tool call.
 */

// lets create 2 variations of this, first we'll use the built in createReActGraph function
// in the next example, we'll build it manually from scratch, with langgraph nodes and edges

// 2. create the agent
const weatherAgent = createAgent({
  model: llmModel,
  tools: [getWeatherTool]
});

export async function runWeatherAgent() {
  // 3. execute the agent
  const response = await weatherAgent.invoke({
    messages: [{ role: "user", content: "What is the weather like in Los Angeles right now?" }]
  });

  console.log(response.messages[response.messages.length - 1]?.content ?? 'No response');
}