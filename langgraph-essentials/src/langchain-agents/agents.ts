/**
 * An agent is a model calling tools in a loop until a given task is complete.
 * create_agent is a highly configurable harness. At its simplest:
 */

import { MemorySaver } from "@langchain/langgraph";
import { createAgent, tool } from "langchain";
import z from "zod";

// model can be a string or a model instance
// Shape how the agent approaches tasks. Accepts a string or SystemMessage. 
// For dynamic prompts at runtime, use middleware.
// Return a validated schema from the agent using response_format=
// Optional identifier used as the node name when embedding this agent as a subgraph in multi-agent systems.
const Answer = z.object({ summary: z.string(), confidence: z.number() });
const llmAgent = createAgent({
  name: "research_assistant",
  model: "ollama:llama3.1",
  tools: [],
  systemPrompt: "You are a helpful assistant. Be concise and accurate.",
  responseFormat: Answer
});

/**
 * Middlewares are useful to provide fault tolerance (retries on failure),
 * specific guardrails (PII redaction), human in the loop (HITL), etc.
 * 
 * We can create custom middlewares with createMiddleware function, or use the prebuilt once
*/ 

/**
 * Models
 * Models are the reasoning engine of agents. They drive the agent’s decision-making process, 
 * determining which tools to call, how to interpret results, and when to provide a final answer.
 * 
 * Models can be utilized in two ways:
 * 1. With agents - Models can be dynamically specified when creating an agent.
 * 2. Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, 
 *    classification, or extraction without the need for an agent framework.
 * 
 * Models can be created using the createModel function.
 */

/**
 * Messages
 * Messages are the fundamental unit of context for models in LangChain. They represent the input and 
 * output of models, carrying both the content and metadata needed to represent the state of a 
 * conversation when interacting with an LLM.
 * 
 * Messages are objects that contain:
 * 1. Role - Identifies the message type (e.g. system, user)
 * 2. Content - Represents the actual content of the message (like text, images, audio, documents, etc.)
 * 3. Metadata - Optional fields such as response information, message IDs, and token usage
 */

/**
 * Tools
 * Tools extend what agents can do—letting them fetch real-time data, execute code, query external 
 * databases, and take actions in the world.  * Under the hood, tools are callable functions with 
 * well-defined inputs and outputs that get passed to a chat model. 
 * The model decides when to invoke a tool based on the conversation context, and what input
 * arguments to provide. * 
 * 
 */

// example: 
const searchDatabase = tool(
  ({ query, limit }) => `Found ${limit} results for '${query}'`,
  {
    name: "search_database",
    description: "Search the customer database for records matching the query.",
    schema: z.object({
      query: z.string().describe("Search terms to look for"),
      limit: z.number().describe("Maximum number of results to return"),
    }),
  }
);

/**
 * Short term memory
 * Short term memory lets your application remember previous interactions within a 
 * single thread or conversation.
 * 
 * To add short-term memory (thread-level persistence) to an agent, 
 * you need to specify a checkpointer when creating an agent.
 */

const getUserInfo = tool(() => "No user profile on file.", {
  name: "get_user_info",
  description: "Look up information about the current user.",
  schema: z.object({}),
});

const checkpointer = new MemorySaver();

const agent = createAgent({
  model: "google-genai:gemini-3.5-flash",
  tools: [getUserInfo],
  checkpointer,
});

const threadConfig = { configurable: { thread_id: "1" } };
let result = await agent.invoke(
  { messages: [{ role: "user", content: "Hi! My name is Bob." }] },
  threadConfig,
);
let response = result.messages.at(-1)?.content;
console.log(response); // "Hi Bob! Nice to see you here. How are you doing?"

result = await agent.invoke(
  { messages: [{ role: "user", content: "What's my name?" }] },
  threadConfig,
);
response = result.messages.at(-1)?.content;
console.log(response); // "You are Bob!"
