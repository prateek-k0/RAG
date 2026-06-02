/**
 * Lets look at the middlewares in the langgraph
 * For this example, we'll only look at the node style hooks 
 * 
 * 1. Node-Style Hooks (beforeModel, afterModel, beforeAgent, afterAgent)
 *    These run sequentially right at the boundaries of your execution stages.
 *    a. Use Case: Perfect for logging telemetry, checking input schemas, incrementing loop counters, or parsing data.
 *    b. State Modification: You return a plain object containing your partial updates. 
 *       LangGraph automatically applies it to the global agent state using your defined reducers.
 *    c. Control Flow: These hooks support explicit routing using "jumpTo".
 * 
 * beforeAgent: Before agent starts (once per invocation)
 * beforeModel: Before each model call
 * afterModel: After each model response
 * afterAgent: After agent completes (once per invocation)
 * 
 * To alter control flow from a node-style hook, your hook must be declared as an object with 
 * an explicit canJumpTo allowance array. 
 * You can route the flow to three key targets:
 * 
 * 'end': End the graph 
 * 'tools': Call the tools
 * 'model': Call the model
 *  for 'model': This is the exact mechanism used to build Self-Correction and Reflection loops.
 *    If the model returns an answer that fails your structural criteria, you can append a critique 
 *    directly to the message history and force the LLM to try again with that corrective feedback.
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { createAgent, createMiddleware, AIMessage } from "langchain";
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

// define the schema for the state
const stateSchema = z.object({
  messages: z.array(z.any()).describe('The messages in the state').default([]),
  logs: z.array(z.string()).default([]),
})

// create a middleware that logs the state, before and after the model is invoked:
const loggerMiddleware = createMiddleware({
  name: "logger_before_model",
  stateSchema: stateSchema,
  // we could pass a function directly, or a middleware object
  beforeModel: async (state: any) => {
    const message = `[MIDDLEWARE] Before Model`;
    console.log("🔍", message);
    return { logs: [...state.logs, message] };
  },
  // for jump example, lets assume that the model returns a tool_call, and we want to jump to the end for a specific condition
  afterModel: {
    canJumpTo: ['end', 'tools'], // to tell that the middleware may alter the execution of the node
    hook: async (state: any) => {
      // check last message for tool_call
      const lastMessage = state.messages[state.messages.length - 1];
      // below condition only executes afterthe model, and below tool call is executed
      if(AIMessage.isInstance(lastMessage) && (lastMessage.tool_calls ?? []).length > 0) {
        // if the city is London, jump to the end
        const location = lastMessage.tool_calls?.[0]?.args?.location;
        if(lastMessage.tool_calls?.[0].name === 'get_weather' && location === 'London') {
          const message = `[MIDDLEWARE] Cant find location of ${location}`;
          return {
            messages: [new AIMessage(message)],
            logs: [...state.logs, message],
            jumpTo: 'end',
          }
        }
        // else, continue to calling the tool
        const message = `[MIDDLEWARE] Calling tool`;
        console.log("🔍", message);
        return { logs: [...state.logs, message], jumpTo: 'tools' }; // return the new state
      }
      const message = `[MIDDLEWARE] After Model`;
      console.log("🔍", message);
      return { logs: [...state.logs, message] }; // return the new state
    }
  }
})

// create the agent
const weatherAgentWithMiddleware = createAgent({
  model: llmModel,
  tools: [getWeatherTool],
  middleware: [loggerMiddleware], 
})

export async function runNodeStyleMiddleware() {
  // 3. execute the agent
  const response = await weatherAgentWithMiddleware.invoke({
    messages: [{ role: "user", content: "What is the weather like in Paris right now?" }]
  });

  console.log(JSON.stringify(response, null, 2));
}