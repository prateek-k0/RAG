# Subgraphs

A subgraph is a graph that is used as a node in another graph.

Subgraphs are useful for:
- Building multi agent systems
- Reusing a set of nodes in multiple graphs
- Distributing development: when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

## Define subgraph communication

When adding subgraphs, you need to define how the parent graph and the subgraph communicate:

1. Call a subgraph inside a node - Parent and subgraph have different state schemas (no shared keys), or you need to transform state between them. You write a wrapper function that maps parent state to subgraph input and subgraph output back to parent state.

2. Add a subgraph as a node - Parent and subgraph share state keys—the subgraph reads from and writes to the same channels as the parent. You pass the compiled subgraph directly to `addNode` — no wrapper function needed.

## 1. Call a subgraph inside a node

When the parent graph and subgraph have different state schemas (no shared keys), invoke the subgraph inside a node function. This is common when you want to keep a private message history for each agent in a multi-agent system. 

The node function transforms the parent state to the subgraph state before invoking the subgraph, and transforms the results back to the parent state before returning.

```js
import { StateGraph, StateSchema, START } from "@langchain/langgraph";
import * as z from "zod";

const SubgraphState = new StateSchema({
  bar: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "hi! " + state.bar };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const State = new StateSchema({
  foo: z.string(),
});

// Transform the state to the subgraph state and back
const builder = new StateGraph(State)
  .addNode("node1", async (state) => {
    const subgraphOutput = await subgraph.invoke({ bar: state.foo });
    return { foo: subgraphOutput.bar };
  })
  .addEdge(START, "node1");

const graph = builder.compile();
```

## 2. Add a subgraph as a node

When the parent graph and subgraph share state keys, you can pass a compiled subgraph directly to `addNode`. No wrapper function is needed—the subgraph reads from and writes to the parent’s state channels automatically. For example, in multi-agent systems, the agents often communicate over a shared messages key.

![subgraph](../../images/subgraph.avif)

If your subgraph shares state keys with the parent graph, you can follow these steps to add it to your graph:
1. Define the subgraph workflow (subgraphBuilder in the example below) and compile it
2. Pass compiled subgraph to the .addNode method when defining the parent graph workflow

```js
import { StateGraph, StateSchema, START } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)   // add as a node, since state schema is same
  .addEdge(START, "node1");

const graph = builder.compile();
```

## Subgraph Persistence

When you use a subgraph, you need to decide what happens to its internal data between calls. The `checkpointer` parameter on `.compile()` controls subgraph persistence:

1. Per-invocation - checkpointer = undefined: Each call starts fresh and inherits the parent’s checkpointer to support interrupts and durable execution within a single call.

2. per-thread - checkpointer = true: State accumulates across calls on the same thread. Each call picks up where the last one left off.

3. stateless - checkpointer = false: No checkpointing at all—runs like a plain function call. No interrupts or durable execution.

Per-invocation is the right choice for most applications, including multi-agent systems where subagents handle independent requests. Use per-thread when a subagent needs multi-turn conversation memory (for example, a research assistant that builds context over several exchanges).


## 1. Stateful persistence: Per invocation

Use per-invocation persistence when each call to the subgraph is independent and the subagent doesn’t need to remember anything from previous calls. This is the most common pattern, especially for multi-agent systems where subagents handle one-off requests like “look up this customer’s order” or “summarize this document.”

Omit checkpointer or set it to `undefined`. Each call starts fresh, but within a single call the subgraph inherits the parent’s checkpointer and can use interrupt() to pause and resume.

The following examples use two subagents (fruit expert, veggie expert) wrapped as tools for an outer agent:

```js
import { createAgent, tool } from "langchain";
import { MemorySaver, Command, interrupt } from "@langchain/langgraph";
import * as z from "zod";

// create tools
const fruitInfo = tool(
  (input) => `Info about ${input.fruitName}`,
  {
    name: "fruit_info",
    description: "Look up fruit info.",
    schema: z.object({ fruitName: z.string() }),
  }
);

const veggieInfo = tool(
  (input) => `Info about ${input.veggieName}`,
  {
    name: "veggie_info",
    description: "Look up veggie info.",
    schema: z.object({ veggieName: z.string() }),
  }
);

// create agents
const fruitAgent = createAgent({
  model: "gpt-5.4-mini",
  tools: [fruitInfo],
  prompt: "You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
});

const veggieAgent = createAgent({
  model: "gpt-5.4-mini",
  tools: [veggieInfo],
  prompt: "You are a veggie expert. Use the veggie_info tool. Respond in one sentence.",
});

// Wrap subagents as tools for the outer agent
const askFruitExpert = tool(
  async (input) => {
    const response = await fruitAgent.invoke({
      messages: [{ role: "user", content: input.question }],
    });
    return response.messages[response.messages.length - 1].content;
  },
  {
    name: "ask_fruit_expert",
    description: "Ask the fruit expert. Use for ALL fruit questions.",
    schema: z.object({ question: z.string() }),
  }
);

const askVeggieExpert = tool(
  async (input) => {
    const response = await veggieAgent.invoke({
      messages: [{ role: "user", content: input.question }],
    });
    return response.messages[response.messages.length - 1].content;
  },
  {
    name: "ask_veggie_expert",
    description: "Ask the veggie expert. Use for ALL veggie questions.",
    schema: z.object({ question: z.string() }),
  }
);

// Outer agent with checkpointer
const agent = createAgent({
  model: "gpt-5.4-mini",
  tools: [askFruitExpert, askVeggieExpert],
  prompt:
    "You have two experts: ask_fruit_expert and ask_veggie_expert. " +
    "ALWAYS delegate questions to the appropriate expert.",
  checkpointer: new MemorySaver(),
});
```

Each invocation starts with a fresh subagent state. The subagent does not remember previous calls:

```js
const config = { configurable: { thread_id: "1" } };

// First call
let response = await agent.invoke(
  { messages: [{ role: "user", content: "Tell me about apples" }] },
  config,
);
// Subagent message count: 4

// Second call - subagent starts fresh, no memory of apples
response = await agent.invoke(
  { messages: [{ role: "user", content: "Now tell me about bananas" }] },
  config,
);
// Subagent message count: 4 (still fresh!)
```

for Multiple calls to the same subgraph work without conflicts, since each invocation gets its own checkpoint namespace:

```js
const config = { configurable: { thread_id: "1" } };

// LLM calls ask_fruit_expert for both apples and bananas
const response = await agent.invoke(
  { messages: [{ role: "user", content: "Tell me about apples and bananas" }] },
  config,
);
// Subagent message count: 4 (apples - fresh)
// Subagent message count: 4 (bananas - fresh)
```

## 2. Stateful persistence: Per thread

Use per-thread persistence when a subagent needs to remember previous interactions. For example, a research assistant that builds up context over several exchanges, or a coding assistant that tracks what files it has already edited. The subagent’s conversation history and data accumulate across calls on the same thread. Each call picks up where the last one left off.

Compile with `checkpointer=true` to enable this behavior.

The following examples use a fruit expert subagent compiled with checkpointer=True:

```js
// Subagent with checkpointer=true for persistent state
const fruitAgent = createAgent({
  model: "gpt-5.4-mini",
  tools: [fruitInfo],
  prompt: "You are a fruit expert. Use the fruit_info tool. Respond in one sentence.",
  checkpointer: true,
});

// Outer agent with checkpointer
// Use toolCallLimitMiddleware to prevent parallel calls to per-thread subagents,
// which would cause checkpoint conflicts.
const agent = createAgent({
  model: "gpt-5.4-mini",
  tools: [askFruitExpert],
  prompt: "You have a fruit expert. ALWAYS delegate fruit questions to ask_fruit_expert.",
  middleware: [  
    toolCallLimitMiddleware({ toolName: "ask_fruit_expert", runLimit: 1 }),
  ],
  checkpointer: new MemorySaver(),
});
```

State accumulates across invocations—the subagent remembers past conversations:

```js
const config = { configurable: { thread_id: "1" } };

// First call
let response = await agent.invoke(
  { messages: [{ role: "user", content: "Tell me about apples" }] },
  config,
);
// Subagent message count: 4

// Second call - subagent REMEMBERS apples conversation
response = await agent.invoke(
  { messages: [{ role: "user", content: "Now tell me about bananas" }] },
  config,
);
// Subagent message count: 8 (accumulated!)
```

For multiple subgraph calls, When you have multiple different per-thread subgraphs (for example, a fruit expert and a veggie expert), each one needs its own storage space so their checkpoints don’t overwrite each other. This is called namespace isolation.

If you call subgraphs inside a node, LangGraph assigns namespaces based on call order (first call, second call, etc.). This means reordering your calls can mix up which subgraph loads which state. To avoid this, wrap each subagent in its own StateGraph with a unique node name—this gives each subgraph a stable, unique namespace:

```js
import { StateGraph, StateSchema, MessagesValue, START } from "@langchain/langgraph";

function createSubAgent(model: string, { name, ...kwargs }: { name: string; [key: string]: any }) {
  const agent = createAgent({ model, name, ...kwargs });
  return new StateGraph(new StateSchema({ messages: MessagesValue }))
    .addNode(name, agent)  // unique name → stable namespace
    .addEdge(START, name)
    .compile();
}

const fruitAgent = createSubAgent("gpt-5.4-mini", {
  name: "fruit_agent", tools: [fruitInfo], prompt: "...", checkpointer: true,
});
const veggieAgent = createSubAgent("gpt-5.4-mini", {
  name: "veggie_agent", tools: [veggieInfo], prompt: "...", checkpointer: true,
});
const config = { configurable: { thread_id: "1" } };

// First call - LLM calls both fruit and veggie experts
let response = await agent.invoke(
  { messages: [{ role: "user", content: "Tell me about cherries and broccoli" }] },
  config,
);
// Fruit subagent message count: 4
// Veggie subagent message count: 4

// Second call - both agents accumulate independently
response = await agent.invoke(
  { messages: [{ role: "user", content: "Now tell me about oranges and carrots" }] },
  config,
);
// Fruit subagent message count: 8 (remembers cherries!)
// Veggie subagent message count: 8 (remembers broccoli!)
```

## 3. Stateless Persistence

Use this when you want to run a subagent like a plain function call with no checkpointing overhead. The subgraph cannot pause/resume and does not benefit from durable execution. Compile with `checkpointer=false`.

```js
const subgraphBuilder = new StateGraph(...);
const subgraph = subgraphBuilder.compile({ checkpointer: false });
```

## View Subgraph state

When you enable persistence, you can inspect the subgraph state using the `subgraphs` option. With stateless checkpointing (`checkpointer=false`), no subgraph checkpoints are saved, so subgraph state is not available.

### view subgraph state for per-invocation

Returns subgraph state for the current invocation only. Each invocation starts fresh.

```js
import { StateGraph, StateSchema, START, MemorySaver, interrupt, Command } from "@langchain/langgraph";
import * as z from "zod";

const State = new StateSchema({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    const value = interrupt("Provide value:");
    return { foo: state.foo + value };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();  // inherits parent checkpointer

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };

await graph.invoke({ foo: "" }, config);

// View subgraph state for the current invocation
const subgraphState = (await graph.getState(config, { subgraphs: true })).tasks[0].state;

// Resume the subgraph
await graph.invoke(new Command({ resume: "bar" }), config);
```

### view subgraph state for per-state

Returns accumulated subgraph state across all invocations on this thread.
```js
import { StateGraph, StateSchema, MessagesValue, START, MemorySaver } from "@langchain/langgraph";

// Subgraph with its own persistent state
const SubgraphState = new StateSchema({
  messages: MessagesValue,
});

const subgraphBuilder = new StateGraph(SubgraphState);
// ... add nodes and edges
const subgraph = subgraphBuilder.compile({ checkpointer: true });

// Parent graph
const builder = new StateGraph(SubgraphState)
  .addNode("agent", subgraph)
  .addEdge(START, "agent");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };

await graph.invoke({ messages: [{ role: "user", content: "hi" }] }, config);
await graph.invoke({ messages: [{ role: "user", content: "what did I say?" }] }, config);

// View accumulated subgraph state (includes messages from both invocations)
const subgraphState = (await graph.getState(config, { subgraphs: true })).tasks[0].state;
```