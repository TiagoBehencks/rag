// Compares the RAG answer of OpenAI vs local Ollama on the same retrieved context.
// Run: pnpm tsx src/compare-llms.ts ["question" ...]
// ponytail: eyeball comparison + latency, no automatic scoring; add an LLM judge if the question set grows.
import { llm as openai } from './services/openia.js'
import { llm as ollama } from './services/llama.js'
import { generatedRAGResponse } from './services/rag.js'

const MODELS = { 'openai gpt-4o-mini': openai, 'ollama llama3.1': ollama }

const args = process.argv.slice(2)
const QUESTIONS = args.length
  ? args
  : [
      `What is Nike's net income in 2023?`,
      'What was Nike total revenue in fiscal 2023?',
      'How many employees does Nike have worldwide?',
      'Who is the President and Chief Executive Officer?',
      'How does Nike hedge foreign currency exchange risk?',
      'What is the population of Brazil?', // not in the documents: both should refuse
    ]

const totals: Record<string, number> = {}

for (const question of QUESTIONS) {
  console.log(`\n${'='.repeat(80)}\nQ: ${question}`)
  for (const [name, model] of Object.entries(MODELS)) {
    const start = performance.now()
    try {
      const { answer, sources } = await generatedRAGResponse({ question }, model)
      const ms = Math.round(performance.now() - start)
      totals[name] = (totals[name] ?? 0) + ms
      const pages = sources?.map((s) => s.page).join(', ')
      console.log(`\n--- ${name} (${ms} ms, retrieved pages: ${pages})\n${answer.trim()}`)
    } catch (error) {
      console.log(`\n--- ${name} FAILED: ${(error as Error).message}`)
    }
  }
}

console.log(`\n${'='.repeat(80)}\nAvg latency per answer (includes retrieval):`)
console.table(Object.fromEntries(Object.entries(totals).map(([n, ms]) => [n, `${Math.round(ms / QUESTIONS.length)} ms`])))
