
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { v4 as uuidV4 } from 'uuid';

import { embeddings } from "./llama.js";
import { qdrantClient } from "./qdrant.js";

import { config } from "../config.js";

import type { ProcessDocumentParams, UploadResponse } from "../types.js";
import { loadPdf } from "../utils/loadPdf.js";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
})


export async function processDocument({
  filePath,
  fileName
}: ProcessDocumentParams): Promise<UploadResponse> {
  const docs = await loadPdf(filePath)

  if (!docs?.length) { 
    throw new Error('No docs')
  }

  const allSplits = await textSplitter.splitDocuments(docs)

  if (!allSplits.length) { 
    throw new Error('No chuncks')
  }

  const documentId = uuidV4()
  const documentsChuncksWithMetadata = allSplits.map((chunk, index) => ({
    id: uuidV4(),
    text: chunk.pageContent,
    metadata: {
      documentId,
      chunkIndex: index,
      uploadAt: new Date().toISOString(),
      page: chunk.metadata.page,
      fileName,
    }
  }))

  const texts = documentsChuncksWithMetadata.map(doc => doc.text)
  const vectors = await embeddings.embedDocuments(texts)

  const data = documentsChuncksWithMetadata.map((chunk, idx) => {
    const vector = vectors[idx];

    if (!vector || !Array.isArray(vector)) {
      throw new Error('Invalid vector')
    }

    return {
      id: chunk.id,
      payload: {
        text: chunk.text,
        ...chunk.metadata,
      },
      vector,
    }
  })

  await qdrantClient.upsert(config.qdrandt.collectionName, {
    points: data,
    wait: true,
  })

  return {
    success: true,
    chuncksCount: documentsChuncksWithMetadata.length,
    message: 'Documents process with success',
    documentId
  }
}