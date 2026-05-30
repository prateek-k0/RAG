import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

/**
 * This is a basic flow that transforms a string into a different string.
 * Uses conditional edge to convert into different strings, based on input
 */

// Step 1: Create schema for the state of the graph
const GraphState = Annotation.Root({
  text: Annotation<string>,  // default update => with object.assign(state, newState)
  // reducer determines how to update the state, by default, it does object.assign(state, newState)
  stepCount: Annotation<number>({
    default: () => 0,
    reducer: (current, next) => current + next, // current and next point to the same attribute (stepCount) of current and new state object
  }),
  processingLog: Annotation<string[]>({
    reducer: (current, next) => current.concat(next),
    default: () => [],
  }),
});

// Step 2: Create the graph with this state
const workflow = new StateGraph(GraphState)

// Step 3: Define the nodes of the graph
// A node reads from state and returns an update object
// langgraph waraps the nodes in RunnableLambda, so we can use async/await
const sanitizedInputNode = async (state: typeof GraphState.State) => {
  return {
    text: state.text.trim(),
    stepCount: 1,
    processingLog: ["Sanitized string whitespace"],
  }
}

const upperCaseNode = async (state: typeof GraphState.State) => {
  return {
    text: state.text.toUpperCase(),
    stepCount: 2,
    processingLog: ["Converted to uppercase"],
  }
}

const fallbackNode = async (state: typeof GraphState.State) => {
  return {
    text: state.text.replace(/[\s\-]/g, "_"),
    stepCount: 3,
    processingLog: ["FALLBACK -> Replaced all whitespace with underscore"],
  }
}

// Step 4: Define conditional edges of the graph
const evaluationEdge = (state: typeof GraphState.State) => {
  // If input length is less than 20 characters, route to uppercase processor
  if (state.text.length < 25) {
    return "upperCase";
  }
  // If input length is greater than 20 characters, route to fallback processor
  return "fallback";
}

// Step 5: Construct the graph
workflow
  .addNode("sanitizedInput", sanitizedInputNode)
  .addNode("upperCase", upperCaseNode)
  .addNode("fallback", fallbackNode)
  .addEdge(START, "sanitizedInput")
  .addConditionalEdges("sanitizedInput", evaluationEdge, {
    upperCase: "upperCase",
    fallback: "fallback",
  })
  .addEdge("upperCase", END)
  .addEdge("fallback", END)

// Step 6: build and run
export const runGraph = async () => {
  const executableApp = workflow.compile();
  
  const result = await executableApp.invoke({ 
    text: "   agentic architectures   " 
  });
  
  console.log("Final State Object:", JSON.stringify(result, null, 2));
};