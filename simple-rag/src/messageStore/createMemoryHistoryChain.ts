import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { RunnableSequence, RunnableWithMessageHistory } from "@langchain/core/runnables";
import { INPUT_KEY, MESSAGE_HISTORY_KEY } from "../utils/constants";

// in production, you would use a database to store the message history
const messageStore = new Map<string, InMemoryChatMessageHistory>();

export function createMemoryHistoryChain(chain: RunnableSequence<any>) {
  return new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: (sessionId: string) => {
      if (!messageStore.has(sessionId)) {
        messageStore.set(sessionId, new InMemoryChatMessageHistory());
      }
      return messageStore.get(sessionId)!;
    },
    inputMessagesKey: INPUT_KEY,
    historyMessagesKey: MESSAGE_HISTORY_KEY,
  })
}