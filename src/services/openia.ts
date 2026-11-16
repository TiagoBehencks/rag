import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai'

import { config } from '../config.js'

export const embeddings = new OpenAIEmbeddings({
  openAIApiKey: config.openai.apiKey,
  model: 'text-embedding-3-small',
  maxRetries: 2,
  timeout: 10000, // 10s
})

export const llm = new ChatOpenAI({
  openAIApiKey: config.openai.apiKey,
  model: 'gpt-4o-mini',
  temperature: 0,
  maxRetries: 2,
   timeout: 20000, // 20s
})