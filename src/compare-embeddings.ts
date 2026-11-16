// Compares retrieval quality of OpenAI vs local Ollama embeddings on the same chunks.
// Run: pnpm tsx src/compare-embeddings.ts [pdfPath]
// ponytail: in-memory cosine search, no Qdrant; fine for one PDF, use real collections if the corpus grows.
import { OllamaEmbeddings } from '@langchain/ollama'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

import { embeddings as openai } from './services/openia.js'
import { embeddings as ollama } from './services/llama.js'
import { loadPdf } from './utils/loadPdf.js'
import { config } from './config.js'
import { qdrantClient } from './services/qdrant.js'
import { searchDocuments } from './services/query.js'

// Baseline without task prefixes, to show what they buy.
const ollamaRaw = new OllamaEmbeddings({
  model: 'nomic-embed-text',
  baseUrl: process.env.OLLAMA_URL || 'http://localhost:11434',
})

const TOP_K = 3
const RANK_DEPTH = 10

// A chunk counts as relevant when it matches `expect` (crude keyword ground truth).
const QUESTIONS: { q: string; expect: RegExp }[] = [
  { q: 'What was Nike total revenue in fiscal 2023?', expect: /51,217/ },
  { q: 'How many employees does Nike have worldwide?', expect: /83,700/ },
  { q: 'Where is Nike world headquarters located?', expect: /Beaverton, Oregon/ },
  { q: 'Who is the independent registered public accounting firm that audits Nike?', expect: /PricewaterhouseCoopers/ },
  { q: 'How much revenue did Converse generate?', expect: /Converse[\s\S]{0,300}2,427|2,427[\s\S]{0,300}Converse/ },
  { q: 'What was the effective tax rate?', expect: /effective tax rate[\s\S]{0,200}18\.2%|18\.2%[\s\S]{0,200}effective tax rate/i },
  { q: 'What is the size of the share repurchase program approved by the board?', expect: /\$18 billion/ },
  { q: 'Which countries manufacture most of Nike footwear?', expect: /Vietnam, Indonesia and China/ },
  { q: 'Who is the President and Chief Executive Officer?', expect: /Donahoe/ },
  { q: 'How does Nike hedge foreign currency exchange risk?', expect: /cash flow hedges/i },
]

type Embedder = {
  name: string
  docs: (texts: string[]) => Promise<number[][]>
  query: (text: string) => Promise<number[]>
}

const EMBEDDERS: Embedder[] = [
  { name: 'openai text-embedding-3-small', docs: (t) => openai.embedDocuments(t), query: (t) => openai.embedQuery(t) },
  { name: 'ollama nomic-embed-text, no prefixes', docs: (t) => ollamaRaw.embedDocuments(t), query: (t) => ollamaRaw.embedQuery(t) },
  // The app's embedder: adds the search_document / search_query task prefixes (see services/llama.ts).
  { name: 'ollama nomic-embed-text + task prefixes (as in app)', docs: (t) => ollama.embedDocuments(t), query: (t) => ollama.embedQuery(t) },
]

function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!
    na += a[i]! * a[i]!
    nb += b[i]! * b[i]!
  }
  return dot / Math.sqrt(na * nb)
}

const ms = (start: number) => Math.round(performance.now() - start)

const pdfPath = process.argv[2] ?? 'uploads/NIKE10K2023.pdf'
const pages = await loadPdf(pdfPath)
if (!pages?.length) throw new Error(`Could not read ${pdfPath}`)

// Same splitter settings as services/document.ts
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 })
const chunks = (await splitter.splitDocuments(pages)).map((d) => d.pageContent)
console.log(`${pdfPath}: ${pages.length} pages, ${chunks.length} chunks\n`)

const relevant = QUESTIONS.map(({ expect }) => new Set(chunks.flatMap((c, i) => (expect.test(c) ? [i] : []))))
QUESTIONS.forEach(({ q }, i) => {
  if (!relevant[i]!.size) console.warn(`WARN no chunk matches ground truth for: ${q}`)
})

const names: string[] = []
const rankings: number[][][] = [] // [model][question] -> ranked chunk indexes
const summary: Record<string, unknown>[] = []

function report(name: string, dims: number, perQuestion: number[][], indexMs: number | string, queryMs: number) {
  names.push(name)
  rankings.push(perQuestion)
  // 1-based rank of the first relevant chunk, 0 if not in the top RANK_DEPTH
  const firstHit = perQuestion.map((r, qi) => r.findIndex((i) => relevant[qi]!.has(i)) + 1)
  summary.push({
    model: name,
    dims,
    [`hit@${TOP_K}`]: `${firstHit.filter((r) => r > 0 && r <= TOP_K).length}/${QUESTIONS.length}`,
    [`hit@${RANK_DEPTH}`]: `${firstHit.filter((r) => r > 0).length}/${QUESTIONS.length}`,
    MRR: (firstHit.reduce((s, r) => s + (r ? 1 / r : 0), 0) / QUESTIONS.length).toFixed(3),
    'index ms': indexMs,
    'query ms (avg)': Math.round(queryMs),
  })
}

for (const e of EMBEDDERS) {
  let t = performance.now()
  const docVectors = await e.docs(chunks)
  const indexMs = ms(t)

  t = performance.now()
  const perQuestion: number[][] = []
  for (const { q } of QUESTIONS) {
    const qv = await e.query(q)
    perQuestion.push(
      docVectors
        .map((v, i) => [cosine(qv, v), i] as const)
        .sort((a, b) => b[0] - a[0])
        .slice(0, RANK_DEPTH)
        .map(([, i]) => i),
    )
  }
  report(e.name, docVectors[0]!.length, perQuestion, indexMs, ms(t) / QUESTIONS.length)
}

// Live collection, through the app's own search path. Skipped when Qdrant is unreachable.
try {
  const collection = config.qdrandt.collectionName
  const { count } = await qdrantClient.count(collection, { exact: true })

  // Were the stored vectors embedded with task prefixes? Re-embed a sample both ways and compare.
  const { points } = await qdrantClient.scroll(collection, { limit: 5, with_payload: true, with_vector: true })
  const texts = points.map((p) => p.payload?.text as string)
  const prefixed = await ollama.embedDocuments(texts)
  const raw = await ollamaRaw.embedDocuments(texts)
  const avg = (vs: number[][]) =>
    (points.reduce((s, p, i) => s + cosine(p.vector as number[], vs[i]!), 0) / points.length).toFixed(4)
  console.log(
    `Qdrant "${collection}": ${count} points. Stored vectors vs re-embedded sample (cosine, 1 = identical): ` +
      `with prefixes ${avg(prefixed)}, without ${avg(raw)}\n`,
  )

  const t = performance.now()
  const perQuestion: number[][] = []
  for (const { q } of QUESTIONS) {
    const { answers } = await searchDocuments({ question: q, topK: RANK_DEPTH })
    // Map stored chunks back to local chunk indexes; unknown text gets a unique negative index.
    perQuestion.push(answers.map((a, i) => (chunks.indexOf(a.text) >= 0 ? chunks.indexOf(a.text) : -1 - i)))
  }
  report(`qdrant "${collection}" via searchDocuments (live app)`, (points[0]?.vector as number[]).length, perQuestion, '-', ms(t) / QUESTIONS.length)
} catch (error) {
  console.warn(`Skipping live Qdrant check: ${(error as Error).message}\n`)
}

console.table(summary)

console.log(`\nRank of first relevant chunk per question (- = not in top ${RANK_DEPTH}), and top-${TOP_K} overlap vs OpenAI:`)
console.table(
  QUESTIONS.map(({ q }, qi) => {
    const row: Record<string, unknown> = { question: q.slice(0, 60), relevantChunks: relevant[qi]!.size }
    names.forEach((_, ei) => {
      const r = rankings[ei]![qi]!
      const top = r.slice(0, TOP_K)
      const shared = top.filter((i) => rankings[0]![qi]!.slice(0, TOP_K).includes(i)).length
      row[`m${ei} rank`] = r.findIndex((i) => relevant[qi]!.has(i)) + 1 || '-'
      if (ei > 0) row[`m${ei} overlap`] = `${shared}/${TOP_K}`
    })
    return row
  }),
)
console.log(names.map((n, i) => `m${i} = ${n}`).join('\n'))
