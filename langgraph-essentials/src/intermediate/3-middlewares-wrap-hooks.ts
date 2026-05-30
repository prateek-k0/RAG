/**
 * These intercept execution blocks and control when (or if) the underlying handler is called.
 * a. Use Case: Perfect for caching results, handling network retries, stripping out sensitive 
 *  data (PII redaction), or injecting dynamic tools at runtime.
 * b. State Modification: Because these are closures, they typically return a Command o
 *    object containing an explicit state update alongside the response.
 * c. Control Flow: They manipulate control flow implicitly by choosing whether or not to 
 *    invoke the underlying handler(request/response). If they catch an exception, 
 *    they can short-circuit and return an alternative response.
 */

/*
  state updation with wrap hooks
  There are 2 scenarios:
  1. if you do not return a command,you HAVE TO return the response directly. 
  2. you return a command - 
    a. command doesnt update messages - langgraph implicitly adds message from the model / tool to the messages array
    b. command updates messages - langgraph will update the messages array with the new messages attribute
  either way, the messages array is updated with the new message.

example: 

const usageTrackingStateSchema = z.object({
  lastModelCallTokens: z.number().optional(),
});

const trackUsage = createMiddleware({
  name: "trackUsage",
  stateSchema: usageTrackingStateSchema,
  wrapModelCall: async (request, handler) => {
    const response = await handler(request);
    return new Command({ update: { lastModelCallTokens: 150 } });
    // we havent updated the messages in the command, langgraph will automatically 
    // add the response to the messages array of the state
  },
});

compare that to this:
wrapModelCall: async (request, handler) => {
  const response = await handler(request);
  
  // By targeting 'messages', you tell the framework: "I am taking manual control of the text history now."
  return new Command({
    update: {
      lastModelCallTokens: 150,
      messages: [new AIMessage("[SYSTEM INTERCEPT] The original response was hidden.")]
    }
  });
  // we have updated the messages in the command, langgraph will update the messages array 
  // with the new messages attribute
}

Summary Checklist for wrap-hooks Returns
1. Just want the text to flow through cleanly? return await handler(request);
2. Want to modify the text directly? return new AIMessage({ ... });
3. Want to keep the text but log metadata variables to a state key / jump to another node? 
  return new Command({ update: { ... } });
  or return new Command({ update: { ... }, jumpTo: 'end' });
*/

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatOllama } from "@langchain/ollama";
import { AIMessage, createAgent } from "langchain";
import { createMiddleware } from "langchain";
import "dotenv/config";
import { Command } from "@langchain/langgraph";

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.5,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192
});

// 1. Define the tool
const getWeatherTool = tool(
  async ({ location }) => {
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
);

// 2. create middleware around the model call
// 2.1 performance middleware for logging the time taken to call the model
const performanceMiddleware = createMiddleware({
  name: "model_performance_middleware",
  wrapModelCall: async (request, handler) => {
    // console.log("🔍model call request: ", request)
    // before the model call
    const start = performance.now()
    // call the handler (model call)
    const response = await handler(request);
    // after the model call
    const end = performance.now()
    const duration = end - start
    console.log(`Model call took ${duration}ms`)
    // return the response, cleanly, without any modifications
    return response
  }
});

// 2.2 tool call middleware to see if the location is London, and if so, return a command to jump to the end
const toolLocationCheckMiddleware = createMiddleware({
  name: "tool_location_check_middleware",
  wrapToolCall: async (request, handler) => {
    // console.log("🔍tool call request: ", request)
    const toolName = request.toolCall.name;
    const { location } = request.toolCall.args
    // if the tool is called for london, return a command to jump to the end
    if(toolName === 'get_weather' && location === 'London') {
      return new Command({
        update: {
          messages: [new AIMessage(`[MIDDLEWARE] Cant find location of ${location}`)]
        },
        // for wrap-style hooks, goto doesnt work inside createAgent, as node names are hidden
        goto: '__end__' // doesnt work, passes this output of the tool to the next node (model in this case)
      })
    }
    // else, continue to calling the tool
    return await handler(request)
  }
})

// 3. create the agent
const weatherAgentWithMiddleware = createAgent({
  model: llmModel,
  tools: [getWeatherTool],
  middleware: [
    performanceMiddleware,
    toolLocationCheckMiddleware
  ],
});

export async function runWrapStyleMiddleware() {
  // 3. execute the agent
  const response = await weatherAgentWithMiddleware.invoke({
    messages: [{ role: "user", content: "What is the weather like in London right now?" }]
  });

  console.log(JSON.stringify(response, null, 2));
}
