/**
 * We'll use the finite state machine example for this example.
 * Lets have 4 nodes for this: a,b,c,x.
 * If it goes to x, we'll ask user if the user wants to exit,
 * if yes exit the graph
 * else, we'll again ask the user to choose which node the user wants to route to (from a, b, c)
 * we'll use middlewares with interrupts for this example.
 */

import { StateGraph, START, END, Annotation, Command, interrupt, MemorySaver } from "@langchain/langgraph";
import readline from "readline";
import "dotenv/config";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

type GraphNodeType<T> = (state: T) => Promise<Partial<T> | Command>

const allNodes = ['a', 'b', 'c', 'x'];
const restrictiveNodes = ['x'];
const allowedNodes = allNodes.filter(node => !restrictiveNodes.includes(node)); // a, b, c

const GraphState = Annotation.Root({
  sequence: Annotation<string[]>({
    default: () => [],
    reducer: (current, next) => current.concat(next)
  })
})

// create a node creator function
const nodeCreator = (name: string) => {
  return async (state: typeof GraphState.State) => {
    rl.write(state.sequence.concat(name).join(' | ') + '\n')
    return {
      ...state,
      sequence: [name]
    }
  }
}

// lets create 2 middlewares - 1 for a,b,c and 1 for x
const withRedirectionForAllowed = (node: GraphNodeType<typeof GraphState.State>) => {
  return async (state: typeof GraphState.State): Promise<Command | Partial<typeof GraphState.State>> => {
    const response = await node(state);
    const nextNode = allNodes[Math.floor(Math.random() * allNodes.length)];
    return new Command({
      update: {
        sequence: (response as typeof GraphState.State).sequence
      },
      goto: nextNode
    })
  }
}

// in prod, we'd want to separate interrupt into different nodes altogether,
// to avoid re-running the same node over and over again
const withRedirectionForRestrictive = (node: GraphNodeType<typeof GraphState.State>) => {
  return async (state: typeof GraphState.State): Promise<Command | Partial<typeof GraphState.State>> => {
    // with every resumption of the interrupt, this miidleware and node runs completely, everytime
    // check if the user wants to continue to the restricted node
    const userInput = interrupt('Do you want to continue to the restricted node? (y/n)')
    if(userInput.trim().match(/y|yes/i)) {
      // process with the restricted node's response
      // notice in the console, how this runs twice - 
      // once for approval and other when the user chooses next node
      // hence we separate interrupts into multiple nodes
      const response = await node(state);
      // now, interrupt again to ask the user to choose from other allowed nodes (a, b, c)
      const nextNode = interrupt('Which node do you want to route to? (a/b/c)')
      if(allowedNodes.includes(nextNode.trim().toLowerCase())) {
        return new Command({
          update: {
            sequence: (response as typeof GraphState.State).sequence
          },
          goto: nextNode.trim().toLowerCase()
        });
      }
    }
    // else, return to the end of the graph
    return new Command({
      goto: END
    })
  }
}

// construct the graph
const workflow = new StateGraph(GraphState)
  .addNode('a', withRedirectionForAllowed(nodeCreator('a')), { ends: allNodes })
  .addNode('b', withRedirectionForAllowed(nodeCreator('b')), { ends: allNodes })
  .addNode('c', withRedirectionForAllowed(nodeCreator('c')), { ends: allNodes })
  .addNode('x', withRedirectionForRestrictive(nodeCreator('x')), { ends: allowedNodes.concat(END) })
  .addEdge(START, 'a')
  .compile({
    checkpointer: new MemorySaver(),
  });

const config = { configurable: { thread_id: "admin_session_101" } };
let cummulativeState: typeof GraphState.State | Command = { sequence: [] };

export const runInterruptCommand2 = async () => {
  let response = await workflow.invoke(cummulativeState as any, config) as any;
  const interrupts = response.__interrupt__ as any[] | undefined;
  // when there are no interrupts, the sequence is complete
  if(interrupts === undefined) {
    rl.write('\nSequence completed');
    rl.close();
    return;
  }
  rl.question(interrupts[interrupts.length - 1].value, async (answer: string) => {
    cummulativeState = new Command({ resume: answer });
    runInterruptCommand2();
  })
}