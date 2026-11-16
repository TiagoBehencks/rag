import { QdrantClient } from "@qdrant/js-client-rest";

import { config } from '../config.js'

export const qdrantClient = new QdrantClient({
  url: config.qdrandt.url
})

export async function initQdrantCollection() {
  const collections = await qdrantClient.getCollections();

  const exists = collections.collections.find(
    col => col.name === config.qdrandt.collectionName
  )

  if (!exists) {
    await qdrantClient.createCollection(config.qdrandt.collectionName, {
      vectors: {
        size: 768,
        distance: 'Cosine'
      }
    })

    console.log(`Collection ${config.qdrandt.collectionName} created!`)
  } else { 
    console.log(`Collection ${config.qdrandt.collectionName} already exists!`)
  }
}