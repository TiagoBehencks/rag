import { generatedRAGResponse } from "./services/rag.js";

async function main() {
  try {
     const result = await generatedRAGResponse({
      question: `What is Nike's net income in 2023?`,
    })

    console.log(result)
  } catch (error) {
    console.error(error)
  }
}

main()