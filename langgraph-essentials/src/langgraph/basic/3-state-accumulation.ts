import { Annotation, StateGraph, START, END } from "@langchain/langgraph";

/**
  When building AI agents, you often want multiple "evaluator" components to inspect an output without 
  halting the system on the first issue. You want them to accumulate a list of critiques.

  Instead of a loop or a branch, this is a linear pipeline where every single node runs sequentially, 
  evaluates a specific aspect of the input data, and appends anomalies or formatting actions to a 
  shared error/log array via a reducer.
*/

type InputData = {
  email: string;
  password: string;
}

const ValidationGraphState = Annotation.Root({
  inputData: Annotation<InputData>,
  isValid: Annotation<boolean>({
    default: () => true,
    reducer: (current, next) => next,
  }),
  errors: Annotation<string[]>({
    default: () => [],
    reducer: (current, next) => current.concat(next),
  })
});

// define the nodes
// 1. email validation
const emailValidationNode = (state: typeof ValidationGraphState.State) => {
  const { email } = state.inputData;
  const isEmailValid = email.includes("@");
  return {
    isValid: state.isValid && isEmailValid,
    errors: isEmailValid ? [] : [`Email ${email} is invalid`],
  }
}

// 2. password validation
const passwordValidationNode = (state: typeof ValidationGraphState.State) => {
  const { password } = state.inputData;
  const isPasswordValid = password.length >= 8;
  return {
    isValid: state.isValid && isPasswordValid,
    errors: isPasswordValid ? [] : [`Password ${password} is invalid`],
  }
}

// 3. database node
const databaseNode = (state: typeof ValidationGraphState.State) => {
  const { email, password } = state.inputData;
  return {
    isValid: state.isValid,
    errors: state.errors,
  }
}

// 4. construct the conditional edges
const evaluationEdge = (state: typeof ValidationGraphState.State) => {
  if (state.isValid) {
    return "validNode";
  }
  return "invalidNode";
}

// 3. construct the graph
const workflow = new StateGraph(ValidationGraphState);

workflow.addNode("emailValidation", emailValidationNode)
  .addNode("passwordValidation", passwordValidationNode)
  .addNode("databaseNode", databaseNode)
  .addEdge(START, "emailValidation")
  .addEdge("emailValidation", "passwordValidation")
  .addConditionalEdges("passwordValidation", evaluationEdge, {
    validNode: "databaseNode",
    invalidNode: END,
  })
  .addEdge("databaseNode", END)

export const runValidationGraph = async () => {
  const executableApp = workflow.compile();
  const result = await executableApp.invoke({
    inputData: { email: "test_test.com", password: "123458" },
  });
  console.log("Final State Object:", JSON.stringify(result, null, 2));
}