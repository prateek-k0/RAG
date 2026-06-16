# Runtime and Config

In LangGraph (JavaScript), runtime context and config serve different purposes, even though both can carry data across nodes in a graph.

Here’s the breakdown:

## 1. Runtime context

- Purpose: Pass **immutable**, **per-run data** that is tied to the **execution of the graph**.
- Scope: Exists only for the duration of a single graph run.
- Use Cases: Database connections, API clients, User session data, Request-specific metadata
- Characteristics: Not meant for persistent configurations, Accissible in all nodes during that run, Does not survive between runs

example:

```js
const node = async ({ state, runtime }) => {
  console.log(runtime)
  return {}   // no-op node
}

const graph = new StateGraph(MessagesAnnotation.State)
  .addNode(node)
  .addEdge(START, node)
  .addEdge(node, END)
  .compile({ checkpointer: new MemorySaver() })

const runtime = {
  db: myDatabaseConnection,
  userId: "user_123",
};

await graph.invoke({ messages: [] }, { runtime })
```

## 2. Configurable (`config.configurable`)

Persistent settings like model and thread_id.

```js
import { StateGraph } from "@langchain/langgraph";

// 1️⃣ Define the state shape
const state = {
  query: null,
  result: null,
};

// 2️⃣ Define nodes
async function fetchUserData( state, { runtime, config }) {
  console.log("Node 1: Fetching user data...");

  // runtime: per-run data (e.g., DB connection, user session)
  console.log("Runtime userId:", runtime.userId);

  // config: persistent settings
  console.log("Config model:", config.configurable.model);

  // Simulate fetching data
  state.result = `Data for ${runtime.userId} using model ${config.configurable.model}`;
  return state;
}

async function processData({ state }) {
  console.log("Node 2: Processing data...");
  state.result = state.result.toUpperCase();
  return state;
}

// 3️⃣ Build the graph
const graph = new StateGraph({ channels: state })
  .addNode("fetchUserData", fetchUserData)
  .addNode("processData", processData)
  .addEdge("fetchUserData", "processData")
  .setEntryPoint("fetchUserData");

// 4️⃣ Compile the graph
const app = graph.compile();

// 5️⃣ Run the graph with runtime + config
(async () => {
  const runtime = {
    userId: "user_123", // per-run
    db: { /* mock DB connection */ },
  };

  const config = {
    configurable: {
      model: "gpt-4o", // persistent setting
      thread_id: "thread_456",
    },
  };

  const finalState = await app.invoke(
    { query: "Hello" }, // initial state
    { runtime, config } // execution context
  );

  console.log("✅ Final State:", finalState);
})();
```
