export type SearchResponse = {
  id: string
  text: string
  score: number
  metadata: {
    documentId: string
    fileName: string
    page?: number
    chunkIndex: number
  }
}

export type QueryRequest = {
  question: string
  topK?: number
}

export type QueryResponse = {
  question: string
  answers: SearchResponse[]
  countChunks: number
}

export type ProcessDocumentParams = { 
  filePath: string
  fileName: string
}

export type UploadResponse = {
  success: boolean
  documentId: string
  chuncksCount: number
  message?: string
}

type Source = {
  fileName: string
  page?: number
  score?: number
}

export type RAGResponse = {
  question: string
  answer: string
  sources?: Source[]
  tokenUsed?: number
}