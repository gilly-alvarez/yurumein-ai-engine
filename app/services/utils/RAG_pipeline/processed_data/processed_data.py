from typing import List, Optional
from glob import glob
import os
import re

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image


class ProcessedData:
    def __init__(self, file_paths: Optional[List[str]] = None):
        self.data_directory = "app/services/utils/RAG_pipeline/data/new_docs"
        self.file_paths = file_paths or []
        self.default_chunk_size = 800
        self.default_chunk_overlap = 120

    def load_all_file_paths(self) -> List[str]:
        """Load all document file paths (PDF, TXT, DOCX, DOC, XLSX, XLS, CSV, images)."""
        if self.file_paths:
            existing_paths = [path for path in self.file_paths if os.path.exists(path)]
            print(f"Found {len(existing_paths)} explicitly provided files")
            return existing_paths

        all_paths = []
        file_extensions = ["*.pdf", "*.txt", "*.docx", "*.doc", "*.xlsx", "*.xls", "*.csv", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]

        counts = {}
        for ext in file_extensions:
            paths = glob(os.path.join(self.data_directory, f"**/{ext}"), recursive=True)
            all_paths.extend(paths)
            if paths:
                counts[ext] = len(paths)

        total = len(all_paths)
        details = ", ".join([f"{count} {ext}" for ext, count in counts.items()])
        print(f"Found {total} files ({details})")
        return all_paths

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text if text.strip() else ""
        except Exception as e:
            print(f"OCR failed for {file_path}: {str(e)}")
            return ""

    def _base_metadata(self, file_path: str, doc_type: str, doc_index: int) -> dict:
        file_name = os.path.basename(file_path)
        return {
            "source": file_path,
            "filename": file_name,
            "doc_type": doc_type,
            "doc_index": doc_index,
            "section": file_name,
        }

    def load_all_documents(self) -> List[Document]:
        """Load documents and enrich them with consistent metadata."""
        all_paths = self.load_all_file_paths()
        all_docs: List[Document] = []

        for file_path in all_paths:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                loader = None
                doc_type = None

                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    doc_type = "pdf"
                elif file_ext == ".txt":
                    loader = TextLoader(file_path, encoding="utf-8")
                    doc_type = "txt"
                elif file_ext in [".docx", ".doc"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    doc_type = "word"
                elif file_ext in [".xlsx", ".xls"]:
                    loader = UnstructuredExcelLoader(file_path)
                    doc_type = "excel"
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                    doc_type = "csv"
                elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                    text = self._extract_text_from_image(file_path)
                    if text:
                        doc = Document(page_content=text, metadata={"source": file_path})
                        docs = [doc]
                        doc_type = "image"
                    else:
                        print(f"No text extracted from image {file_path}")
                        continue
                else:
                    continue

                if loader:
                    docs = loader.load()

                loaded_count = 0
                for index, doc in enumerate(docs):
                    cleaned_text = self._clean_text(doc.page_content or "")
                    if not cleaned_text:
                        continue

                    metadata = self._base_metadata(file_path, doc_type, index)
                    metadata.update(doc.metadata or {})
                    if metadata.get("page") is not None:
                        metadata["page_number"] = int(metadata["page"]) + 1

                    all_docs.append(
                        Document(
                            page_content=cleaned_text,
                            metadata=metadata,
                        )
                    )
                    loaded_count += 1

                print(f"Loaded {loaded_count} documents from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs

    def chunking(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[Document]:
        """Split documents into cleaner, metadata-rich chunks."""
        all_docs = self.load_all_documents()
        if not all_docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or self.default_chunk_size,
            chunk_overlap=chunk_overlap or self.default_chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n\n", "\n- ", "\n", ". ", " ", ""],
        )

        raw_chunks = splitter.split_documents(all_docs)
        chunks: List[Document] = []
        for chunk_index, chunk in enumerate(raw_chunks):
            cleaned_text = self._clean_text(chunk.page_content or "")
            if not cleaned_text:
                continue

            metadata = dict(chunk.metadata or {})
            metadata["chunk_index"] = chunk_index
            metadata["char_count"] = len(cleaned_text)
            metadata["preview"] = cleaned_text[:160]

            chunks.append(
                Document(
                    page_content=cleaned_text,
                    metadata=metadata,
                )
            )

        print(f"Created {len(chunks)} chunks")
        return chunks
