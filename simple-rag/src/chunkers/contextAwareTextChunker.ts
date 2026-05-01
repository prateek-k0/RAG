import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";
import path from "path";

export class SectionAwareChunker {
  // Matches lines with 3+ uppercase characters/spaces/&/-/,, and ending with an uppercase character, with allowed trailing spaces
  private HEADER_REGEX = /^([A-Z][A-Z\s&,\-–]+[A-Z])\s*$/gm;
  private splitter: RecursiveCharacterTextSplitter;
  private chunkSize: number;
  private chunkOverlap: number;

  constructor(chunkSize: number = 1000, chunkOverlap: number = 200) {
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.splitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      separators: ["\n\n", "\n", ". ", " ", ""]
    });
  }

  async chunk(pages: Document<Record<string, any>>[], metadataExtra: Record<string, any> = {}): Promise<Document<Record<string, any>>[]> {
    const allChunks: Document<Record<string, any>>[] = [];
    for (const page of pages) {
      const text = page.pageContent;
      const fileName = path.basename(page.metadata.source);
      const metaData =  { ...page.metadata, sourceFile: fileName };

      // find all header matches in the page
      const matches = Array.from(text.matchAll(this.HEADER_REGEX));

      // if no headers are found, use fallback splitter
      if(matches.length === 0) {
        const fallbackChunks = await this.splitter.splitDocuments([page]);
        allChunks.push(...fallbackChunks.map((chunk) => new Document({
          pageContent: chunk.pageContent,
          metadata: {
            ...chunk.metadata,
            ...metadataExtra,
          },
        })));
        continue;
      }

      // else, if a header is found
      // 1. handle the text before the first header (preamble)
      if(matches[0].index > 0) {    // index is the startIndex of the pattern matched
        const preambleText = text.substring(0, matches[0].index).trim();
        if(preambleText.length > 0) {
          allChunks.push(new Document({
              pageContent: preambleText,
              metadata: {
                ...metaData,
                ...metadataExtra,
                section: 'PREAMBLE'
              },
          }));
        }
      }

      // 2. iterate through sections (from header start to next header start)
      for(let i = 0; i < matches.length; i++) {
        const match = matches[i];
        const sectionTitle = match[1].trim();
        const sectionStart = match.index;
        const sectionEnd = matches[i + 1] ? matches[i + 1].index : text.length;

        const sectionBody = text.substring(sectionStart, sectionEnd).trim();
        const sectionMeta = { ...metaData, section: sectionTitle, ...metadataExtra };

        // 3. if the body is smaller than the allowed chunk size, add it as a single chunk
        if(sectionBody.length <= this.chunkSize) {
          allChunks.push(new Document({
            pageContent: `[SECTION: ${sectionTitle}]\n${sectionBody}`,
            metadata: sectionMeta,
          }));
        } else {
          // 4. if the body is larger than the allowed chunk size, split it into chunks
          // make sure to preserve the section title in each chunk
          const subDocs = await this.splitter.splitDocuments([
            new Document({ pageContent: sectionBody, metadata: sectionMeta })
          ]);
          
          const taggedSubDocs = subDocs.map(doc => {
            doc.pageContent = `[SECTION: ${sectionTitle}]\n${doc.pageContent}\n[END OF SECTION]`;
            return doc;
          });
          
          allChunks.push(...taggedSubDocs);
        }
      }      
    }
    return allChunks
  }
}