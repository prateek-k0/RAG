import { ChatOllama } from "@langchain/ollama";
import { SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";

export async function summarizeMessages(messageHistory: any[], waterMark: number = 5) {
  if (!Array.isArray(messageHistory) || messageHistory.length < waterMark) {
    return messageHistory;
  }

  const summarizer = new ChatOllama({
    temperature: 0,
    repeatPenalty: 1.2,
    model: "llama3.1",
    baseUrl: process.env.LLAMA_BASE_URL ?? "http://localhost:11434",
  });

  try {
    const transcript = messageHistory
    .map((m) => `${m._getType() === "human" ? "User" : "AI"}: ${m.content}`)
    .join("\n");

    const response = await summarizer.pipe(new StringOutputParser()).invoke([
      ["system", "Summarize the following chat transcript concisely. Focus on facts. Output ONLY the summary text, without any other text, preamble or explanation."],
      ["human", `Transcript:\n${transcript}`],
    ])

    const summaryText = response.trim();

    if (!summaryText) {
      console.warn("⚠️ Summary was empty. Keeping original history.");
      return messageHistory;
    }

    console.log("✅ Summary Created:", summaryText);

    return [new SystemMessage(`This is a summary of the conversation so far: ${summaryText}`)];
  } catch (error) {
    console.error("❌ Summarization Error:", error);
    return messageHistory;
  }
}