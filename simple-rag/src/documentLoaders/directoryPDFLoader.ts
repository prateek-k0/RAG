import { DirectoryLoader } from "@langchain/classic/document_loaders/fs/directory";
import path from "path";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

export async function loadTextDocuments(directoryPath: string) {
  const directory = path.resolve(import.meta.dirname, directoryPath);
  const loader = new DirectoryLoader(
    directory,
    {
      // ".txt": (path) => new TextLoader(path),
      ".pdf": (path) => new PDFLoader(path),
    },
    true // recursive to true for nested directories
  );

  return await loader.load();
}
