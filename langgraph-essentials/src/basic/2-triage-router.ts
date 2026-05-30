import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

/**
 * In an Agentic RAG system, you often route a user's query to completely different vector databases depending
 *  on their intent (e.g., routing a billing question to a billing database vs. a technical question to 
 * engineering documentation).
 * 
 * The graph takes a raw request, inspects a metadata category, 
 * routes it to an isolated processing node, and updates a centralized telemetry state.
 */

// 1. The State tracks the payload and the target service
const TriageState = Annotation.Root({
  payload: Annotation<string>,
  category: Annotation<"billing" | "technical" | "general">,
  routedTo: Annotation<string>,
  executionLogs: Annotation<string[]>({
    reducer: (current, next) => current.concat(next),
    default: () => [],
  })
});

// 2. Router that handles the branching logic
const triageRouter = async (state: typeof TriageState.State) => {
  return state.category
}

// 3. Service nodes that handle the actual processing
const billingService = async (state: typeof TriageState.State) => {
  return {
    executionLogs: ["Routed to billing service"],
    routedTo: "billing",
  }
}

const technicalService = async (state: typeof TriageState.State) => {
  return {
    executionLogs: ["Routed to technical service"],
    routedTo: "technical",
  }
}

const generalService = async (state: typeof TriageState.State) => {
  return {
    executionLogs: ["Routed to general service"],
    routedTo: "general",
  }
}

// 3.1 classifier node -> routes to the appropriate service node
const classifierNode = async (state: typeof TriageState.State) => {
  return {
    payload: state.payload,
    category: state.category,
  }
}

// 4. Construct the graph
const triageWorkflow = new StateGraph(TriageState);

triageWorkflow
  .addNode("classifier", classifierNode)
  .addNode("billingService", billingService)
  .addNode("technicalService", technicalService)
  .addNode("generalService", generalService)
  .addEdge(START, "classifier")
  .addConditionalEdges("classifier", triageRouter, {
    billing: "billingService",
    technical: "technicalService",
    general: "generalService",
  })
  .addEdge("billingService", END)
  .addEdge("technicalService", END)
  .addEdge("generalService", END)

export const runTriageRouter = async () => {
  const executableApp = triageWorkflow.compile();
  const result = await executableApp.invoke({
    payload: "How do I cancel my subscription?",
    category: "billing",
  });
  console.log("Final State Object:", JSON.stringify(result, null, 2));
}