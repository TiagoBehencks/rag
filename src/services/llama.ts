import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama'

const baseUrl = process.env.OLLAMA_URL || 'http://localhost:11434'

// nomic-embed-text is trained with task prefixes; without them retrieval quality drops.
class NomicEmbeddings extends OllamaEmbeddings {
  override embedDocuments(texts: string[]) {
    return super.embedDocuments(texts.map((text) => `search_document: ${text}`))
  }

  override embedQuery(text: string) {
    return super.embedQuery(`search_query: ${text}`)
  }
}

export const embeddings = new NomicEmbeddings({
  model: 'nomic-embed-text', // 768 dimensions, must match the collection size in qdrant.ts
  baseUrl,
})

export const llm = new ChatOllama({
  model: 'llama3.1',
  temperature: 0,
  baseUrl,
})
