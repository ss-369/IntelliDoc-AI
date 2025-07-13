import os
import tempfile
from typing import List, Optional
import logging
from pathlib import Path

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class for processing and chunking documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> str:
        """Load text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if not text.strip():
                raise ValueError("The text file is empty")
            
            return text
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return text
            except Exception as e:
                logger.error(f"Error loading TXT file with fallback encoding: {e}")
                raise
        except Exception as e:
            logger.error(f"Error loading TXT file: {e}")
            raise
    
    def load_document(self, file_path: str) -> str:
        """Load document based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.load_pdf(file_path)
        elif file_extension == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def process_uploaded_file(self, uploaded_file) -> str:
        """Process uploaded file from Streamlit"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Load document
            text = self.load_document(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return text
        except Exception as e:
            # Ensure cleanup even if error occurs
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
            raise e
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split text into chunks and create Document objects"""
        try:
            if not text.strip():
                raise ValueError("Cannot chunk empty text")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
                
                # Add custom metadata if provided
                if metadata:
                    doc_metadata.update(metadata)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(f"Successfully created {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise
    
    def process_document(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """Complete document processing pipeline"""
        try:
            # Load document
            text = self.load_document(file_path)
            
            # Add file metadata
            file_metadata = {
                "source": file_path,
                "filename": Path(file_path).name,
                "file_size": os.path.getsize(file_path)
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # Chunk text
            documents = self.chunk_text(text, file_metadata)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents"""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_chunk_size": total_chars / len(documents) if documents else 0,
            "average_words_per_chunk": total_words / len(documents) if documents else 0
        }
