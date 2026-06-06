/**
 * For langgraph (nodes and edges), to use middleware functionality,
 * We wrap the nodes with Higher Order Functions (HOF)
 * 
 * To mutate state, we can either return a plain object or a Command object.
 * If we return a command object, it alos allows us to jump to other nodes using the "goto" property.
 * 
 * messages manipulated by middleware are upserted/overwritten by ID (message id),
 * instead of being appended blindly, so that no duplicate messages are added.
 * 
 * example: 
 * const node = async () => {...}
 * 
 * const nodeWithMiddleware = async function withMiddleware(node) => {
 *  return async (state) => {
 *    const response = await node(state)
 *    return response
 *    // or return new Command({ update: { ... }, jumpTo: '__end__' })
 *  }
 * }
 * 
 * For this example, lets again use the weather app, with middlewares to log and route for "london" location
 * also, we dont need to define too many edges for this, as Command's "goto" can be used to jump to other nodes
 */

import { StateGraph, MessagesAnnotation, START, END, Annotation, Command } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import "dotenv/config";

// 1. Define the llm model
const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.5,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192, // 8k context window
});

// 2. Define the tool
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

// 3. create tool and agent nodes
const tools = [getWeatherTool];
const weatherToolNode = new ToolNode(tools); // Prebuilt node that runs matching tools automatically

const callAgentNode = async (state: typeof MessagesAnnotation.State) => {
  const llmModelWithTools = llmModel.bindTools(tools);
  const response = await llmModelWithTools.invoke(state.messages);
  return {
    messages: [response]
  };
}

// 4. create a middleware to log the messages
const withLoggerMiddleware = (nodeFunction: (state: typeof MessagesAnnotation.State) => Promise<Command | Partial<typeof MessagesAnnotation.State>>) => {
  return async (state: typeof MessagesAnnotation.State) => {
    console.log('[Logger Middleware]')
    const response = await nodeFunction(state)
    return response
  }
}

// 4.1 middleware for agent_node to see if the response is a tool_call, and if so, redirect to the tool_node
const withRedirectionMiddlewareForAgentNode = (agentNode: typeof callAgentNode) => {
  return async (state: typeof MessagesAnnotation.State): Promise<Command | Partial<typeof MessagesAnnotation.State>> => {
    const response = await agentNode(state)
    const lastMessage = response.messages[response.messages.length - 1] as AIMessage;
    if(lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
      return new Command({
        update: {
          messages: [
            // langgraph will automatically add the response to the messages array, 
            // so we dont need to manually update as below
            // ...state.messages,
            ...response.messages
          ]
        },
        goto: 'tools_node'
      });
    }
    // else, move to end
    return new Command({
      update: {
        messages: [
          // langgraph will automatically add the response to the messages array, 
          // so we dont need to manually update as below
          // ...state.messages,
          ...response.messages
        ]
      },
      goto: END
    });
  }
}

// 4.2 middleware to redirect for tool_call depending on the location argument for get_weather tool
const withRedirectionMiddlewareForToolNode = (toolNode: typeof weatherToolNode) => {
  return async (state: typeof MessagesAnnotation.State): Promise<Command | Partial<typeof MessagesAnnotation.State>> => {
    const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
    const toolCall = lastMessage?.tool_calls?.[0];
    const toolName = toolCall?.name;
    const toolArgs = toolCall?.args;
    // if the tool is called for london, return a command to jump to the end
    if(toolName === 'get_weather' && toolArgs?.location === 'London') {
      console.error(`[MIDDLEWARE] Cant find location of ${toolArgs?.location}`)
      const closureToolMessage = new ToolMessage({
        content: `Execution Access Error: Weather searches for "${toolArgs?.location}" are restricted.`,
        tool_call_id: toolCall?.id ?? '',
        name: toolName
      });
      return new Command({
        update: {
          messages: [
            // langgraph will automatically add the response to the messages array, 
            // so we dont need to manually update as below
            // ...state.messages,
            closureToolMessage
          ]
        },
        goto: END // Graceful structural termination
      });
    }
    // else, continue to calling the tool
    // toolNodes must be invoked, since ToolNode extends Runnable, and is not a function as other nodes.
    const response = await toolNode.invoke(state) 
    return new Command({
      // langgraph will automatically add the response to the messages array, 
      // so we dont need to manually update as below
      update: {
        messages: [
          // langgraph will automatically add the response to the messages array, 
          // so we dont need to manually update as below
          // ...state.messages,
          ...response.messages
        ]
      },
      goto: 'agent_node'
    })
  }
}

// 5. Assembling the Custom Graph Architecture
// notice how we have an edgeless graph, since the middleware commands handle the routing
const workflowGraph = new StateGraph(MessagesAnnotation)
  // Wrap your custom node using the higher-order middleware function container
  .addNode("agent_node", withLoggerMiddleware(withRedirectionMiddlewareForAgentNode(callAgentNode)), {
    ends: ["tools_node", END] // Inform the compiler that our middleware Command can route to END or tools_node
  })
  .addNode("tools_node", withRedirectionMiddlewareForToolNode(weatherToolNode), {
    ends: ['agent_node', END] // Inform the compiler that our middleware Command can route to agent_node or END
  })
  .addEdge(START, "agent_node")

const compiledWeatherAgent = workflowGraph.compile();

// 6. execute the graph
export const runWeatherAgentWorkflowWithMiddlewares = async () => {
  const finalStateOutput = await compiledWeatherAgent.invoke({
    messages: [new HumanMessage("What's the weather in Paris right now?")]
  });
  console.log('--------------------------------');
  console.log(finalStateOutput.messages);
}