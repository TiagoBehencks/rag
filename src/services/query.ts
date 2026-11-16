import { config } from '../config.js'
import { qdrantClient } from './qdrant.js'
import { embeddings } from './llama.js'
import type { QueryRequest, QueryResponse, SearchResponse } from '../types.js'

export async function searchDocuments({
  question,
  topK = 3,
}: QueryRequest): Promise<QueryResponse> {
  const queryVector = await embeddings.embedQuery(question)

  const searchResult = await qdrantClient.query(config.qdrandt.collectionName, {
    query: queryVector,
    limit: topK,
    with_payload: true
  })

  const answers: SearchResponse[] = searchResult.points.map((item) => ({
    id: item.id as string,
    text: item.payload?.text as string,
    score: item.score,
    metadata: {
      documentId: item.payload?.documentId as string,
      fileName: item.payload?.fileName as string,
      page: item.payload?.page as number,
      chunkIndex: item.payload?.chunkIndex as number
    }
  }))

  return {
    answers,
    question,
    countChunks: answers.length
  }
}