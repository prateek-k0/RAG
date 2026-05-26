import { BaseCallbackHandler } from "@langchain/core/callbacks/base";
import { HumanMessage, AIMessage, Message } from "@langchain/core/messages";
import { ChainValues } from "@langchain/core/utils/types";

class MessageStore {
  // in prod, you would use a database to store the message history
  private messageStore: Map<string, Message[]>;

  constructor() {
    this.messageStore = new Map<string, Message[]>();
  }

  public getMessageHistory(sessionId: string): Message[] {
    if (!this.messageStore.has(sessionId)) {
      this.messageStore.set(sessionId, []);
    }
    const messageHistory = this.messageStore.get(sessionId) ?? [];
    return messageHistory;
  }

  public addMessage(sessionId: string, message: Message) {
    if (!this.messageStore.has(sessionId)) {
      this.messageStore.set(sessionId, []);
    }
    this.messageStore.get(sessionId)?.push(message);
  }

  public clearMessageHistory(sessionId: string) {
    this.messageStore.delete(sessionId);
  }
}

export const messageStoreService = new MessageStore();

// chain handler to update message store
export class MessageHistoryCallbackHandler extends BaseCallbackHandler {
  name = "MessageHistoryCallbackHandler";
  private sessionId?: string;
  private userInput?: string;

  constructor(sessionId: string) {
    super();
    this.sessionId = sessionId;
  }

  async handleChainStart(
    chain: any,
    inputs: any,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, any>, // Runtime configurations live here
  ) {
    // Only capture data at the root chain level (where parentRunId is undefined)
    if (!parentRunId) {
      // 1. Grab the original user question right out of the root inputs object!
      this.userInput = inputs.input;
    }
  }

  async handleChainEnd(outputs: ChainValues, runId: string, parentRunId?: string) {
    // Only update memory at the root chain completion
    if (!parentRunId) {
      if (!this.sessionId || !this.userInput) {
        console.warn("⚠️ Callback missed context initialization. History not updated.");
        return;
      }
      let aiResponse: string | undefined = undefined;
      if (typeof outputs === "string") {
        aiResponse = outputs; // This block handles StringOutputParser output directly
      } else if (outputs && typeof outputs === "object") {
        aiResponse = outputs.text || outputs.content || outputs.output;
      }
      // update message store with the new message
      messageStoreService.addMessage(this.sessionId!, new HumanMessage(this.userInput!));
      messageStoreService.addMessage(this.sessionId!, new AIMessage(aiResponse!));

      // console.log("🔍 Updated Message History:", messageStoreService.getMessageHistory(this.sessionId!));
    }
  }
}
