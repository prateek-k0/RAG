/**
 * Interrupts
 * 
 * Interrupts allow you to pause graph execution at specific points and wait for external input 
 * before continuing. This enables human-in-the-loop patterns where you need external input to proceed.
 * 
 * When an interrupt is triggered, LangGraph saves the graph state using its persistence 
 * layer and waits indefinitely until you resume execution.
 * 
 * Interrupts work by calling the interrupt() function at any point in your graph nodes. 
 * The function accepts any JSON-serializable value which is surfaced to the caller. 
 * When you’re ready to continue, you resume execution by re-invoking the graph using Command, 
 * which then becomes the return value of the interrupt() call from inside the node.
 * 
 * 1. Checkpointing keeps your place: the checkpointer writes the exact graph state so you can resume later, 
 *    even when in an error state.
 * 2. thread_id is your pointer: use { configurable: { thread_id: ... } } as options to the 
 *    invoke method to tell the checkpointer which state to load.
 * 3. Interrupt payloads surface as __interrupt__: the values you pass to interrupt() 
 *    return to the caller in the __interrupt__ field so you know what the graph is waiting on.
 */

import { HumanMessage } from "@langchain/core/messages";
import { interrupt, MessagesAnnotation, Command, StateGraph, END, START, MemorySaver } from "@langchain/langgraph";

/**
 * Pausing using interrupt function
 * The interrupt function pauses graph execution and returns a value to the caller. 
 * When you call interrupt within a node, LangGraph saves the current graph state and waits 
 * for you to resume execution with input.
 * 
 * To use interrupt, you need:
 * 1. A checkpointer to persist the graph state (use a durable checkpointer in production)
 * 2. A thread ID in your config so the runtime knows which state to resume from
 * 3. To call interrupt() where you want to pause (payload must be JSON-serializable)
 */

// example for pausing
async function approvalNode(state: typeof MessagesAnnotation.State) {
  // pause and ask for approval
  const approved = interrupt("Do you approve of this action?");

  // Command({ resume: ... }) provides the value to this variable
  return { approved }
}

/**
 * How interrupts work
 * Think of an inline interrupt() call as a two-way portal sitting inside your node.
 * 1. The First Pass (The Pause): When the node reaches the line const answer = interrupt("Question"), 
 *    it stops cold. It freezes the state to the checkpointer, packages up your question string, 
 *    throws a GraphInterrupt exception, and goes offline.
 * 2. The Second Pass (The Resume): When the client passes a new Command({ resume: "My Value" }) back 
 *    into that exact same thread ID, LangGraph boots up and re-runs that node from the very beginning.
 * 3. The Trick: When the node execution hits that exact same interrupt() line for the second time, 
 *    the engine says, "Ah, I already have a resume token for this!" It skips throwing the error
 *    entirely, grabs the value "My Value", and assigns it directly to your answer variable.
 */

/**
 * Resuming interrupts
 * 
 * After an interrupt pauses execution, you resume the graph by invoking it again with a Command 
 * that contains the resume value. The resume value is passed back to the interrupt call, allowing 
 * the node to continue execution with the external input.
 */

const config = { configurable: { thread_id: '123' } }
const graph = new StateGraph(MessagesAnnotation)
  .addNode("approval", approvalNode)
  .addEdge(START, "approval")
  .addEdge("approval", END)
  .compile({
    checkpointer: new MemorySaver(),
  });

// Initial run - hits the interrupt and pauses
// thread_id is the durable pointer back to the saved checkpoint
const result = await graph.invoke({
  messages: [new HumanMessage('data')]
}, config)

// Check what was interrupted
// __interrupt__ mirrors every payload you passed to interrupt()
console.log((result as any).__interrupt__)
// [{ value: 'Do you approve this action?', ... }]

// Resume with the human's response
// Command({ resume }) returns that value from interrupt() in the node
await graph.invoke(new Command({ resume: true }), config);

/**
 * Key points about resuming:
 * 1. You must use the same thread ID when resuming that was used when the interrupt occurred
 * 2. The value passed to new Command({ resume: ... }) becomes the return value of the interrupt call
 * 3. The node restarts from the beginning of the node where the interrupt was called when resumed, so 
 * any code before the interrupt runs again
 * 4. You can pass any JSON-serializable value as the resume value
 */

/**
 * Rules for interrupts:
 * 1. Do not wrap interrupt calls in try/catch - as interrupts use a special exception to stop the flow
 * 2. Do not reorder interrupt calls within a node
 * 3. Do not conditionally skip interrupt calls within a node
 * 4. Do not loop interrupt calls using logic that isn’t deterministic across executions
 * 5. Do not return complex values in interrupt calls
 * 6. Pass simple, JSON-serializable types to interrupt
 * 7. Pass dictionaries/objects with simple values
 * 8. Do not pass functions, class instances, or other complex objects to interrupt
 * 9. Side effects called before interrupt must be idempotent
 *    Because interrupts work by re-running the nodes they were called from, side effects called before interrupt 
 *    should (ideally) be idempotent. For context, idempotency means that the same operation can be applied multiple 
 *    times without changing the result beyond the initial execution. 
 */