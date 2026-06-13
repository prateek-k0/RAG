/**
 * Similar to previous example, we'll build a custom graph architecture
 * we wont use llm, just node functions and middlewares for routing
 * we'll create a random series generator, that generates an finite series of 'a', 'b' or 'c'
 */

import { StateGraph, START, END, Annotation, Command } from "@langchain/langgraph";
import "dotenv/config";

const allowedLetters: string[] = ['a', 'b', 'c'];

const GraphState = Annotation.Root({
  sequence: Annotation<string[]>({
    default: () => [],
    reducer: (current, next) => current.concat(next),
  }),
  logs: Annotation<string[]>({
    default: () => [],
    reducer: (current, next) => current.concat(next),
  })
});

const nodeA = async (state: typeof GraphState.State) => {
  return {
    sequence: ['a'],
    logs: ['Added "a" to the sequence'],
  }
}

const nodeB = async (state: typeof GraphState.State) => {
  return {
    sequence: ['b'],
    logs: ['Added "b" to the sequence'],
  }
}

const nodeC = async (state: typeof GraphState.State) => {
  return {
    sequence: ['c'],
    logs: ['Added "c" to the sequence'],
  }
}

const withRedirectionMiddleware = (node: (state: typeof GraphState.State) => Promise<typeof GraphState.State>) => {
  return async (state: typeof GraphState.State): Promise<Command | Partial<typeof GraphState.State>> => {
    // if the sequence is less than 10, redirect to the next node
    if(state.sequence.length < 10) {
      const response = await node(state);
      const nextLetter = allowedLetters[Math.floor(Math.random() * allowedLetters.length)];
      return new Command({
        update: {
          sequence: (response as typeof GraphState.State).sequence,
          logs: (response as typeof GraphState.State).logs,
        },
        goto: nextLetter
      })
    }
    // else, move to end
    return new Command({
      update: {
        logs: ['Sequence is complete'],
      },
      goto: END
    })
  }
}

const workflowGraph = new StateGraph(GraphState)

workflowGraph
// important to add ends, otherwise throws unreachable error
// see how we have added the nodes themselves as end, as an example of self loops
.addNode('a', withRedirectionMiddleware(nodeA), { ends: ['a', 'b', 'c', END] })
.addNode('b', withRedirectionMiddleware(nodeB), { ends: ['a', 'b', 'c', END] })
.addNode('c', withRedirectionMiddleware(nodeC), { ends: ['a', 'b', 'c', END] })
.addEdge(START, 'a')

const compiledWorkflow = workflowGraph.compile();

export const runSeriesGenerator = async () => {
  const finalStateOutput = await compiledWorkflow.invoke({});
  console.log('--------------------------------');
  console.log(finalStateOutput)
}