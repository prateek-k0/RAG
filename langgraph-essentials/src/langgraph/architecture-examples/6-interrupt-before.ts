/**
 * interrupBefore
 * used to interrupt the graph before the node is executed (every time, not conditionally)
 */

import {
  StateGraph,
  MessagesAnnotation,
  START,
  END,
  MemorySaver
} from "@langchain/langgraph";

import { AIMessage } from "@langchain/core/messages";

// define a generic node
const genericNode = async (state: typeof MessagesAnnotation.State) => {
  console.log('inside generic node')
  return {
    messages: [new AIMessage('generic node response')]
  }
}

// define a node that must be interrupted before execution
const prohibitedNode = async (state: typeof MessagesAnnotation.State) => {
  console.log('inside prohibited node')
  return {
    messages: [new AIMessage('prohibited node response')]
  }
}

// build the graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode('generic', genericNode)
  .addNode('prohibited', prohibitedNode)
  .addEdge(START, 'generic')
  .addEdge('generic', 'prohibited')
  .addEdge('prohibited', END)
  .compile({
    checkpointer: new MemorySaver(),
    // ⚡ THE GUARDRAIL: Automatically freeze execution BEFORE entering this specific node
    interruptBefore: ['prohibited'],
    // theres also interruptAfter, which is used to interrupt the graph after the node is executed
  });

export const runInterruptBefore = async () => {
  const config = { configurable: { thread_id: "admin_session_101" } };
  console.log("--- FIRST INVOCATION ---");
  const firstRunResult = await workflow.invoke({
    messages: []
  }, config);
  // Notice that 'prohibited node' NEVER ran. Execution stopped at the boundary.
  console.log('Result of first invocation: ', firstRunResult.messages.map(msg => msg.content));
  // assume that the user clicks approve action, and we resume the graph
  console.log('---Resuming workflow execution...---');
  // To resume an interrupted graph, you invoke it passing 'null' as the state update.
  // This tells the engine: "Pick up exactly where you left off on this thread configuration."
  const secondRunResult = await workflow.invoke(null, config);
  // notice it only starts from the node that was interrupted, not from the beginning of the graph
  // messages from the generic nodes is still present tho, as the state was preserved
  console.log('Result of second invocation: ', secondRunResult.messages.map(msg => msg.content));
}