import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'
import type { BaseChatModel } from '@langchain/core/language_models/chat_models'

import type { QueryRequest, RAGResponse } from '../types.js'
import { llm } from './openia.js'
import { searchDocuments } from './query.js'

const PROMPT_TEMPLATE = ChatPromptTemplate.fromMessages([
  [
    'system',
    `
      You are an AI assistant that answers questions based on the provided documents.

      Rules
        - Use only the information provided in the documents to answer.
        - If the information isn't in the documents, answer: "Sorry, I didn't find any relevant information about that question."
        - Provide concise and direct answers
        - Cite the sources used in the answer in the format: [1], [2], etc.
    `
  ],
  [
    'user',
    
    `CONTEXT:
      {context}
      
    QUESTION:
      {question}
    
      ANSWER:
    `,
],
])

export async function generatedRAGResponse(
  { question, topK = 3 }: QueryRequest,
  model: BaseChatModel = llm, // override to compare providers, see compare-llms.ts
): Promise<RAGResponse> {
  const searchResults = await searchDocuments({
    question,
    topK,
  })

  if (searchResults.answers.length === 0) {
    return {
      question,
      answer: `Sorry, didn't found any information relevation information about that question`
    }
  }

  const context = searchResults.answers
    .map((item, idx) => `[${idx + 1}]: ${item.text}`)
    .join('\n\n')


  const chains = PROMPT_TEMPLATE.pipe(model).pipe(new StringOutputParser)

  const answer = await chains.invoke({context, question})

  const sources = searchResults.answers.map(({ metadata, score }) => ({
      fileName: metadata.fileName,
      ...(metadata.page !== undefined && { page: metadata.page }),
      score,
    }))

  return {
    question,
    answer,
    sources,
  }
}