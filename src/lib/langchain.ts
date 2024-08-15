import { getVectorStore } from "./vector-store";
import { getPineconeClient } from "./pinecone-client";
import {
  StreamingTextResponse,
  experimental_StreamData,
  LangChainStream,
} from "ai-stream-experimental";
import { streamingModel, nonStreamingModel } from "./llm";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

type callChainArgs = {
  question: string;
  chatHistory: string;
};

export async function callChain({ question, chatHistory }: callChainArgs) {
  try {
    // Open AI recommendation
    const sanitizedQuestion = question.trim().replaceAll("\n", " ");
    const pineconeClient = await getPineconeClient();
    const vectorStore = await getVectorStore(pineconeClient);
    const { stream, handlers } = LangChainStream({
      experimental_streamData: true,
    });
    const data = new experimental_StreamData();

    // Create the contextualize question prompt
    const contextualizeQSystemPrompt = `
      Given a chat history and the latest user question
      which might reference context in the chat history,
      formulate a standalone question which can be understood
      without the chat history. Do NOT answer the question, just
      reformulate it if needed and otherwise return it as is.`;
    const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
      ["system", contextualizeQSystemPrompt],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);
    const historyAwareRetriever = await createHistoryAwareRetriever({
      llm: nonStreamingModel,
      retriever: vectorStore.asRetriever(),
      rephrasePrompt: contextualizeQPrompt,
    });

    // Create the question-answering prompt
    const qaSystemPrompt = `
      You are an assistant for question-answering tasks. Use
      the following pieces of retrieved context to answer the
      question. If you don't know the answer, just say that you
      don't know. Use three sentences maximum and keep the answer
      concise.
      \n\n
      {context}`;
    const qaPrompt = ChatPromptTemplate.fromMessages([
      ["system", qaSystemPrompt],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);

    const questionAnswerChain = await createStuffDocumentsChain({
      llm: streamingModel,
      prompt: qaPrompt,
    });

    const ragChain = await createRetrievalChain({
      retriever: historyAwareRetriever,
      combineDocsChain: questionAnswerChain,
    });

    // Question using chat-history
    const response = await ragChain.invoke({
      chat_history: chatHistory,
      input: sanitizedQuestion,
    });

    const sourceDocuments = response?.sourceDocuments;
    const firstTwoDocuments = sourceDocuments.slice(0, 2);
    const pageContents = firstTwoDocuments.map(
      ({ pageContent }: { pageContent: string }) => pageContent
    );

    data.append({
      sources: pageContents,
    });
    data.close();

    // Return the readable stream
    return new StreamingTextResponse(stream, {}, data);
  } catch (e) {
    console.error(e);
    throw new Error("Call chain method failed to execute successfully!!");
  }
}
