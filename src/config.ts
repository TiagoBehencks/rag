import dotenv from 'dotenv'

dotenv.config();

const FILE_SIZE_LIMIT = 10 * 1024 * 1024 // 20 MB

export const config = {
  openai: {
    apiKey: process.env.OPENAI_API_KEY || ''
  },
  qdrandt: {
    url: process.env.QDRANT_URL || 'http://localhost:6333',
    collectionName: process.env.QDRAND_COLLECTION || 'documents',
  },
  uploads: {
    directory: process.env.UPLOAD_DIR 
  },
  server: {
    port: process.env.PORT || 3000,
    maxFileSize: FILE_SIZE_LIMIT
  }
}