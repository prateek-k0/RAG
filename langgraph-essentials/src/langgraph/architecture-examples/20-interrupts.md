# Interrupts

Interrupts allow you to pause graph execution at specific points and wait for external input before continuing. This enables **human-in-the-loop** patterns where you need external input to proceed. When an interrupt is triggered, LangGraph saves the graph state using its persistence layer and waits indefinitely until you resume execution.

Interrupts work by calling the `interrupt()` function at any point in your graph nodes. The function accepts any JSON-serializable value which is surfaced to the caller. you resume execution by re-invoking the graph using `Command`, which then becomes the return value of the `interrupt()` call from inside the node.

Unlike static breakpoints (which pause before or after specific nodes), interrupts are dynamic: they can be placed anywhere in your code and can be conditional based on your application logic.

- **Checkpointing keeps your place**: the checkpointer writes the exact graph state so you can resume later, even when in an error state.
- `thread_id` is your pointer: use `{ configurable: { thread_id: ... } }` as options to the invoke method to tell the checkpointer which state to load.
- Interrupt payloads surface as `__interrupt__`: the values you pass to interrupt() return to the caller in the `__interrupt__` field so you know what the graph is waiting on.

**The `thread_id` you choose is effectively your persistent cursor. Reusing it resumes the same checkpoint; using a new value starts a brand-new thread with an empty state.**

## Pause using `interrupt()`
The `interrupt` function pauses graph execution and returns a value to the caller. When you call `interrupt` within a node, LangGraph saves the current graph state and waits for you to resume execution with input.

To use interrupt, you need:
1. A `checkpointer` to persist the graph state (use a durable checkpointer in production)
2. A **thread ID** in your config so the runtime knows which state to resume from
3. To call `interrupt()` where you want to pause (payload must be JSON-serializable)

```js
import { interrupt } from "@langchain/langgraph";

async function approvalNode(state: State) {
    // Pause and ask for approval
    const approved = interrupt("Do you approve this action?");

    // Command({ resume: ... }) provides the value returned into this variable
    return { approved };
}
```

When you call `interrupt`, this is what happens:
1. Graph execution gets suspended at the exact point where `interrupt` is called
2. State is saved using the checkpointer so execution can be resumed later, In production, this should be a persistent checkpointer (e.g. backed by a database)
3. Value is returned to the caller under `__interrupt__`; it can be any JSON-serializable value (string, object, array, etc.)
4. Graph waits indefinitely until you resume execution with a response
5. Response is passed back into the node when you resume, becoming the return value of the interrupt() call

## Resuming interrupts
After an `interrupt` pauses execution, you resume the graph by invoking it again with a `Command` that contains the `resume` value. The `resume` value is passed back to the `interrupt` call, allowing the node to continue execution with the external input.

```js
import { Command } from "@langchain/langgraph";

// Initial run - hits the interrupt and pauses
// thread_id is the durable pointer back to the saved checkpoint
const config = { configurable: { thread_id: "thread-1" } };
const result = await graph.invoke({ input: "data" }, config);

// Check what was interrupted
// __interrupt__ mirrors every payload you passed to interrupt()
console.log(result.__interrupt__);
// [{ value: 'Do you approve this action?', ... }]

// Resume with the human's response
// Command({ resume }) returns that value from interrupt() in the node
await graph.invoke(new Command({ resume: true }), config);
```

Key points about resuming:
- You must use the same thread ID when resuming that was used when the interrupt occurred
- The value passed to `new Command({ resume: ... })` becomes the return value of the `interrupt` call
- The `node` restarts from the beginning of the node where the `interrupt` was called when resumed, so any code before the `interrupt` runs again
- You can pass any JSON-serializable value as the resume value

# Common interrupt patterns:

The key thing that interrupts unlock is the ability to pause execution and wait for external input. This is useful for a variety of use cases, including:

- Approval workflows: Pause before executing critical actions (API calls, database changes, financial transactions)
- Handling multiple interrupts: Pair interrupt IDs with resume values when resuming multiple interrupts in a single invocation
- Review and edit: Let humans review and modify LLM outputs or tool calls before continuing
- Interrupting tool calls: Pause before executing tool calls to review and edit the tool call before execution
- Validating human input: Pause before proceeding to the next step to validate human input

## 1. Stream with human-in-the-loop (HITL) interrupts
When building interactive agents with human-in-the-loop workflows, you can use event streaming to consume message chunks and state snapshots concurrently while handling interrupts.

Use the typed projections returned by graph.stream_events(..., version="v3") in a loop until the run finishes:
- Stream AI responses token-by-token via stream.messages
- Observe per-step state snapshots via stream.values
- Detect interrupts via stream.interrupted and read their payloads from stream.interrupts
- Resume execution by calling stream_events again with `Command(resume=...)` and repeat until `stream.interrupted` is `false`

```js
import { Command } from "@langchain/langgraph";

let streamInput: Record<string, unknown> | Command = initialInput;

while (true) {
  const stream = await graph.streamEvents(streamInput, {
    ...config,
    version: "v3",
  });

  // Stream LLM message chunks (including any in subgraphs) as they arrive.
  for await (const message of stream.messages) {
    for await (const token of message.text) {
      displayStreamingContent(token);
    }
  }

  // After the run finishes (or pauses), check for interrupts and resume.
  if (!stream.interrupted) {
    const finalState = await stream.output;
    break;
  }

  const interruptInfo = stream.interrupts[0].value;
  const userResponse = await getUserInput(interruptInfo);
  streamInput = new Command({ resume: userResponse });
}
```

- `stream.messages`: Chat-model output as content blocks; iterate `message.text` for token deltas. For nested subgraphs, read message chunks from `stream.subgraphs[*].messages`.
- `stream.values`: Full state snapshots after each step
- `stream.interrupted` / `stream.interrupts`: After each run, check whether the graph paused; read payloads from `stream.interrupts`
- `Command(resume=...)`: Pass as the next `streamEvents` input to resume; loop until the run completes without interrupting

## 2. Handling multiple interrupts
When parallel branches interrupt simultaneously (for example, fan-out to multiple nodes that each call `interrupt()`), you may need to resume multiple interrupts in a single invocation. When resuming multiple interrupts with a single invocation, map each interrupt ID to its resume value. This ensures each response is paired with the correct interrupt at runtime.

```js
import {
  Annotation,
  Command,
  END,
  INTERRUPT,
  MemorySaver,
  START,
  StateGraph,
  interrupt,
  isInterrupted,
} from "@langchain/langgraph";

const State = Annotation.Root({
  vals: Annotation<string[]>({
    reducer: (left, right) =>
      left.concat(Array.isArray(right) ? right : [right]),
    default: () => [],
  }),
});

function nodeA(_state: typeof State.State) {
  const answer = interrupt("question_a") as string;
  return { vals: [`a:${answer}`] };
}

function nodeB(_state: typeof State.State) {
  const answer = interrupt("question_b") as string;
  return { vals: [`b:${answer}`] };
}

const graph = new StateGraph(State)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addEdge(START, "a")
  .addEdge(START, "b")
  .addEdge("a", END)
  .addEdge("b", END)
  .compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: "1" } };

async function main() {
  // Step 1: invoke - both parallel nodes hit interrupt() and pause
  const interruptedResult = await graph.invoke({ vals: [] }, config);
  console.log(interruptedResult);
  /*
  {
    vals: [],
    __interrupt__: [
      { id: '...', value: 'question_a' },
      { id: '...', value: 'question_b' }
    ]
  }
  */

  // Step 2: resume all pending interrupts at once, with a single command
  const resumeMap: Record<string, string> = {};
  if (isInterrupted(interruptedResult)) {
    for (const i of interruptedResult[INTERRUPT]) {
      if (i.id != null) {
        resumeMap[i.id] = `answer for ${i.value}`;
      }
    }
  }
  const result = await graph.invoke(new Command({ resume: resumeMap }), config);

  console.log("Final state:", result);
  //> Final state: { vals: ['a:answer for question_a', 'b:answer for question_b'] }
}

main().catch(console.error);
```

## 3. Approve or reject
One of the most common uses of interrupts is to pause before a critical action and ask for approval. For example, you might want to ask a human to approve an API call, a database change, or any other important decision.

```js
import { interrupt, Command } from "@langchain/langgraph";

const approvalNode: typeof State.Node = (state) => {
  // Pause execution; payload surfaces in result.__interrupt__
  // the aargument that we pass to interrupt is stored as interrupt.value
  const isApproved = interrupt({
    question: "Do you want to proceed?",
    details: state.actionDetails
  });

  // Route based on the response
  if (isApproved) {
    return new Command({ goto: "proceed" }); // Runs after the resume payload is provided
  } else {
    return new Command({ goto: "cancel" });
  }
}
```

When you resume the graph, pass `true` to approve or `false` to reject:
```js
// whatever we pass to resume attribute, will be passed to the isApproved variable
// To approve
await graph.invoke(new Command({ resume: true }), config);

// To reject
await graph.invoke(new Command({ resume: false }), config);
```

## 4. Review and edit state
Sometimes you want to let a human review and edit part of the graph state before continuing. This is useful for correcting LLMs, adding missing information, or making adjustments.

```js
import { interrupt } from "@langchain/langgraph";

const reviewNode: typeof State.Node = (state) => {
  // Pause and show the current content for review (surfaces in result.__interrupt__)
  const editedContent = interrupt({
    instruction: "Review and edit this content",
    content: state.generatedText
  });

  // Update the state with the edited version
  return { generatedText: editedContent };
}
```

When resuming, provide the edited content (passing a dummy string in this example)
```js
await graph.invoke(
  new Command({ resume: "The edited and improved text" }), // Value becomes the return from interrupt()
  config
);
```

## 5. Interrupting tools
You can also place interrupts directly inside tool functions. This makes the tool itself pause for approval whenever it’s called, and allows for human review and editing of the tool call before it is executed.
First, define a tool that uses `interrupt`:

```js
import { tool } from "@langchain/core/tools";
import { interrupt } from "@langchain/langgraph";
import * as z from "zod";

const sendEmailTool = tool(
  async ({ to, subject, body }) => {
    // Pause before sending; payload surfaces in result.__interrupt__
    const response = interrupt({
      action: "send_email",
      to,
      subject,
      body,
      message: "Approve sending this email?",
    });

    if (response?.action === "approve") {
      // Resume value can override inputs before executing
      const finalTo = response.to ?? to;
      const finalSubject = response.subject ?? subject;
      const finalBody = response.body ?? body;
      return `Email sent to ${finalTo} with subject '${finalSubject}'`;
    }
    return "Email cancelled by user";
  },
  {
    name: "send_email",
    description: "Send an email to a recipient",
    schema: z.object({
      to: z.string(),
      subject: z.string(),
      body: z.string(),
    }),
  },
);
```

This approach is useful when you want the approval logic to live with the tool itself, making it reusable across different parts of your graph. The LLM can call the tool naturally, and the interrupt will pause execution whenever the tool is invoked, allowing you to approve, edit, or cancel the action.

## 6. Validating human input
Sometimes you need to validate input from humans and ask again if it’s invalid. You can do this using multiple interrupt calls in a loop.

```js
import { interrupt } from "@langchain/langgraph";

const getAgeNode: typeof State.Node = (state) => {
  let prompt = "What is your age?";

  while (true) {
    const answer = interrupt(prompt); // payload surfaces in result.__interrupt__

    // Validate the input
    if (typeof answer === "number" && answer > 0) {
      // Valid input: continue
      return { age: answer };
    } else {
      // Invalid input: ask again with a more specific prompt
      prompt = `'${answer}' is not a valid age. Please enter a positive number.`;
    }
  }
}
```


# Rules of interrupts

When you call interrupt within a node, LangGraph suspends execution by raising an exception that signals the runtime to pause. This exception propagates up through the call stack and is caught by the runtime, which notifies the graph to save the current state and wait for external input.

When execution resumes (after you provide the requested input), the runtime restarts the entire node from the beginning—it does not resume from the exact line where interrupt was called. This means any code that ran before the interrupt will execute again. Because of this, there’s a few important rules to follow when working with interrupts to ensure they behave as expected.

1. **Do not wrap interrupt calls in try/catch** - The way that interrupt pauses execution at the point of the call is by throwing a special exception. If you wrap the interrupt call in a try/catch block, you will catch this exception and the interrupt will not be passed back to the graph.

2. **Do not reorder interrupt calls within a node** - It’s common to use multiple interrupts in a single node, however this can lead to unexpected behavior if not handled carefully. When a node contains multiple interrupt calls, LangGraph keeps a list of resume values specific to the task executing the node. Whenever execution resumes, it starts at the beginning of the node. For each interrupt encountered, LangGraph checks if a matching value exists in the task’s resume list. Matching is **strictly index-based**, so the order of interrupt calls within the node is important.

3. **Do not return complex values in interrupt calls** - Depending on which checkpointer is used, complex values may not be serializable (e.g. you can’t serialize a function). To make your graphs adaptable to any deployment, it’s best practice to only use values that can be reasonably serialized.

4. **Side effects called before `interrupt` must be idempotent** - Because interrupts work by re-running the nodes they were called from, side effects called before interrupt should (ideally) be idempotent. For context, idempotency means that the same operation can be applied multiple times without changing the result beyond the initial execution. As an example, you might have an API call to update a record inside of a node. If interrupt is called after that call is made, it will be re-run multiple times when the node is resumed, potentially overwriting the initial update or creating duplicate records.

## Using with subgraphs called as functions
When invoking a subgraph within a node, the parent graph will resume execution from the beginning of the node where the subgraph was invoked and the `interrupt` was triggered. Similarly, the subgraph will also resume from the beginning of the node where `interrupt` was called.

```js
async function nodeInParentGraph(state: State) {
    someCode(); // <-- This will re-execute when resumed
    // Invoke a subgraph as a function.
    // The subgraph contains an `interrupt` call.
    const subgraphResult = await subgraph.invoke(someInput);
    // ...
}

async function nodeInSubgraph(state: State) {
    someOtherCode(); // <-- This will also re-execute when resumed
    const result = interrupt("What's your name?");
    // ...
}
```

## Debugging with interrupts
To debug and test a graph, you can use static interrupts as breakpoints to step through the graph execution one node at a time. Static interrupts are triggered at defined points either before or after a node executes. You can set these by specifying interruptBefore and interruptAfter when compiling the graph.

```js
const graph = builder.compile({
    interruptBefore: ["node_a"],
    interruptAfter: ["node_b", "node_c"],
    checkpointer,
});

// Pass a thread ID to the graph
const config = {
    configurable: {
        thread_id: "some_thread"
    }
};

// Run the graph until the breakpoint
await graph.invoke(inputs, config);
// after the breakpoint
await graph.invoke(null, config);
```

We can also pass the breakpoints during run time, inside `.invoke`
```js
// Run the graph until the breakpoint
graph.invoke(inputs, {
    interruptBefore: ["node_a"],
    interruptAfter: ["node_b", "node_c"],
    configurable: {
        thread_id: "some_thread"
    }
});

// Resume the graph
await graph.invoke(null, config);
```

1. `graph.invoke` is called with the `interruptBefore` and `interruptAfter` parameters. This is a run-time configuration and can be changed for every invocation.
2. `interruptBefore` specifies the nodes where execution should pause before the node is executed.
3. `interruptAfter` specifies the nodes where execution should pause after the node is executed.
4. The graph is run until the first breakpoint is hit.
5. The graph is resumed by passing in `null` for the input. This will run the graph until the next breakpoint is hit.