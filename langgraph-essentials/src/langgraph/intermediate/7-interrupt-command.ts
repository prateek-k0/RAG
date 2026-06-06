/**
 * For interrupts, , the Command object changes from an outbound routing tool returned by a node 
 * into an inbound variable handoff passed by the client.
 * 
 * Think of an inline interrupt() call as a two-way portal sitting inside your node.
 * 1. The First Pass (The Pause): When the node reaches the line const answer = interrupt("Question"), 
 *    it stops cold. It freezes the state to the checkpointer, packages up your question string, 
 *    throws a GraphInterrupt exception, and goes offline.
 * 2. The Second Pass (The Resume): When the client passes a new Command({ resume: "My Value" }) back 
 *    into that exact same thread ID, LangGraph boots up and re-runs that node from the very beginning.
 * 3. The Trick: When the node execution hits that exact same interrupt() line for the second time, 
 *    the engine says, "Ah, I already have a resume token for this!" It skips throwing the error entirely, 
 *    grabs the value "My Value", and assigns it directly to your answer variable.
 * For this example, we will again look at the weather agent workflow, where the tool 
 * will interrupt for approval if the location is London.
 */

import { StateGraph, MessagesAnnotation, START, END, Command, MemorySaver, interrupt } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOllama } from "@langchain/ollama";
import "dotenv/config";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

const llmModel = new ChatOllama({
  model: "llama3.1",
  temperature: 0.1,
  baseUrl: process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
  numCtx: 8192, // 8k context window
});

const getWeatherTool = tool(
  async ({ location }) => {
    // add interrupt command here
    if(location === 'London') {
      const answer: string = interrupt("Do you approve of this weather search for London?");
      if(answer.trim().match(/yes|y/i)) {
        return `The weather in London is 22°C and stormy.`;
      } else {
        return `Execution Access Error: Weather searches for London are restricted.`;
      }
    }
    return `The weather in ${location ?? 'Paris'} is 28°C and sunny.`;
  },
  {
    name: 'get_weather',
    description: 'Get the current weather in a given location',
    schema: z.object({
      location: z.string().describe('The city, e.g. San Francisco, Mumbai, etc.'),
    }),
  }
)

const tools = [getWeatherTool];
const weatherToolNode = new ToolNode(tools);

const callAgentNode = async (state: typeof MessagesAnnotation.State) => {
  const llmModelWithTools = llmModel.bindTools(tools);
  const response = await llmModelWithTools.invoke(state.messages);
  return {
    messages: [response]
  };
}

const shouldCallTools = (state: typeof MessagesAnnotation.State) => {
  const lastMessage = state.messages[state.messages.length - 1];
  if(
    lastMessage
    && AIMessage.isInstance(lastMessage)
    && (lastMessage.tool_calls ?? []).length > 0) {
      return "tools"
  }
  return "end"
}

const weatherAgentWorkflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callAgentNode)
  .addNode("tools", weatherToolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldCallTools, {
    tools: "tools",
    end: END,
  })
  .addEdge("tools", "agent")
  .compile({
    checkpointer: new MemorySaver(),
  });
  

export const runWeatherAgentWorkflowWithInterruptCommand = async () => {
  const config = { configurable: { thread_id: "admin_session_101" } };
  console.log("--- STEP 1: INITIAL TRIGGER ---");
  // The first execution will boot up the node and pause instantly at the interrupt() line
  const firstResult = await weatherAgentWorkflow.invoke({
    messages: [new HumanMessage('What is the weather in London?')]
  }, config);
  // at this point, the graph has paused at the interrupt() line
  // and the state is frozen to the checkpointer, with the corresponding thread_id
  console.log(firstResult.messages.map(msg => msg.content));

  console.log("--- STEP 2: CLIENT RESUME COMMAND ---");
  // To resume, you pass a brand-new Command object straight into the stream/invoke parameters
  const resumeStream = await weatherAgentWorkflow.invoke(
    new Command({ resume: 'yes' }),  // whatever value is passed here, will be passed to the "answer" variable in the node
    config
  );
  console.log(resumeStream.messages.map(msg => msg.content));
}