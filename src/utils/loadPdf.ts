import { readFileSync } from "node:fs";
import { PDFParse } from "pdf-parse";
import { Document } from "@langchain/core/documents";

export async function loadPdf(filePath: string): Promise<Document[] | undefined> {
  const parser = new PDFParse({
    data: new Uint8Array(readFileSync(filePath)),
  });

  try {
    const { pages } = await parser.getText();
    return pages.map(
      (page) =>
        new Document({
          pageContent: page.text,
          metadata: { source: filePath, page: page.num - 1 },
        })
    );
  } catch (error) { 
   console.error(error)  
  } finally {
    await parser.destroy();
  }
}
