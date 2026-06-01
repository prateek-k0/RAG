import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

type MathCommand = {
  op: "add" | "subtract" | "multiply";
  value: number;
}

// step 1: create central graph state schema
const GraphState = Annotation.Root({
  commands: Annotation<MathCommand[]>,
  accumulator: Annotation<number>({
    default: () => 0,
    reducer: (current, next) => next, // no update
  }),
  // currentCommand: Annotation<number>,  // to keep track of which command is ongoing, dont need it though, we can mutate commands array directly
  commandHistory: Annotation<MathCommand[]>({
    default: () => [],
    reducer: (current, next) => current.concat(next),
  })
})

// step 2: defined the nodes of the graph

// 2.1 we have an orchestrator node, that decides what needs to be done as per the current command
// we return no change to the state (empty object, when used with Object.assign yields the same state object)
const orchestratorNode = async (state: typeof GraphState.State) => {
  return {}
}

// 2.2 add op node
const addNode = async (state: typeof GraphState.State) => {
  const currentValue = state.commands[0].value;
  const nextValue = state.accumulator + currentValue;
  return {
    commands: state.commands.slice(1),
    accumulator: nextValue,
    // currentCommand: state.currentCommand + 1,
    commandHistory: { op: `add ${currentValue}`, value: nextValue },
  }
}

// 2.3 subtract op node
const subtractNode = async (state: typeof GraphState.State) => {
  const currentValue = state.commands[0].value;
  const nextValue = state.accumulator - currentValue;
  return {
    commands: state.commands.slice(1),
    accumulator: nextValue,
    // currentCommand: state.currentCommand + 1,
    commandHistory: { op: `subtract ${currentValue}`, value: nextValue },
  }
}

// 2.4 multiply op node
const multiplyNode = async (state: typeof GraphState.State) => {
  const currentValue = state.commands[0].value;
  const nextValue = state.accumulator * currentValue;
  return {
    commands: state.commands.slice(1),
    accumulator: nextValue,
    // currentCommand: state.currentCommand + 1,
    commandHistory: { op: `multiply ${currentValue}`, value: nextValue },
  }
}

// step 3: define the conditional edges:
// we will create an edge that routes from the orchestrator node to the appropriate operation node, based on the current command
// if no command exists, we route to the end node
const evaluationEdge = (state: typeof GraphState.State) => {
  if (state.commands.length > 0) {
    const currentCommand = state.commands[0].op;
    switch (currentCommand) {
      case "add":
        return "addNode";
      case "subtract":
        return "subtractNode";
      case "multiply":
        return "multiplyNode";
      default:
        return "end";
    }
  }
  // else, no more commands to process, route to the end node
  return "end";
}

// step 4: construct the graph
const workflow = new StateGraph(GraphState);

workflow
  .addNode("orchestrator", orchestratorNode)
  .addNode("addNode", addNode)
  .addNode("subtractNode", subtractNode)
  .addNode("multiplyNode", multiplyNode)
  .addEdge(START, "orchestrator")
  .addConditionalEdges("orchestrator", evaluationEdge, {
    addNode: "addNode",
    subtractNode: "subtractNode",
    multiplyNode: "multiplyNode",
    end: END,
  })
  .addEdge("addNode", "orchestrator")
  .addEdge("subtractNode", "orchestrator")
  .addEdge("multiplyNode", "orchestrator");

const app = workflow.compile();

export const runCaluclator = async () => {
  const result = await app.invoke({
    commands: [{ op: "add", value: 30 }, { op: "subtract", value: 5 }, { op: "multiply", value: 2 }]
    // currentCommand: 0,
  })
  console.log("Calculation Complete!\n", JSON.stringify(result, null, 2));
}