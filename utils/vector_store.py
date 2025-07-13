import os
import logging
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Class for managing vector store operations with Chroma"""
    
    def __init__(self, 
                 collection_name: str = "documents",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        
        # Create a simple embedding wrapper for ChromaDB compatibility
        class SimpleEmbeddings:
            def __init__(self):
                from chromadb.utils import embedding_functions
                self.ef = embedding_functions.DefaultEmbeddingFunction()
            
            def embed_documents(self, texts):
                return self.ef(texts)
            
            def embed_query(self, text):
                return self.ef([text])[0]
        
        self.embeddings = SimpleEmbeddings()
        self.embedding_model_name = "chromadb-default"
        logger.info("Using ChromaDB default embedding function with LangChain wrapper")
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize or load existing vector store"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma vector store with embedding function
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client=self.client
            )
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store"""
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            # Add documents to vector store
            ids = self.vectorstore.add_documents(documents)
            
            # Persist changes
            self.vectorstore.persist()
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Document]:
        """Perform similarity search"""
        try:
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Perform similarity search
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        try:
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} similar documents with scores for query")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            raise
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection"""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.warning(f"Error getting collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
    
    def delete_collection(self):
        """Delete the current collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._initialize_vectorstore()
            logger.info(f"Deleted collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def update_documents(self, documents: List[Document], ids: List[str]):
        """Update existing documents in the vector store"""
        try:
            if len(documents) != len(ids):
                raise ValueError("Number of documents and IDs must match")
            
            # Delete existing documents with these IDs
            self.vectorstore.delete(ids=ids)
            
            # Add updated documents
            new_ids = self.add_documents(documents)
            
            logger.info(f"Successfully updated {len(documents)} documents")
            return new_ids
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None):
        """Get a retriever for the vector store"""
        try:
            if search_kwargs is None:
                search_kwargs = {"k": 5}
            
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def reset_collection(self):
        """Reset the collection by deleting and recreating it"""
        try:
            self.delete_collection()
            logger.info("Collection reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
