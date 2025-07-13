import logging
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms.base import LLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """Advanced retriever with contextual compression and filtering"""
    
    def __init__(self, vector_store, llm: Optional[LLM] = None):
        self.vector_store = vector_store
        self.llm = llm
        self.base_retriever = None
        self.compression_retriever = None
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Setup base and compression retrievers"""
        try:
            # Setup base retriever
            self.base_retriever = self.vector_store.get_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}  # Retrieve more initially for filtering
            )
            
            # Setup compression retriever if LLM is available
            if self.llm:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.base_retriever
                )
            
            logger.info("Retrievers setup successfully")
            
        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}")
            raise
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = 5, 
                          use_compression: bool = False,
                          similarity_threshold: float = 0.7,
                          filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve relevant documents with optional compression and filtering"""
        try:
            if not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Use compression retriever if available and requested
            if use_compression and self.compression_retriever:
                documents = self.compression_retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(documents)} compressed documents")
            else:
                # Use base similarity search with scores for filtering
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k * 2,  # Get more to allow for filtering
                    filter_dict=filters
                )
                
                # Filter by similarity threshold - Chroma uses distance (lower is better)
                # Convert similarity_threshold to distance threshold
                distance_threshold = 2.0  # More lenient threshold for better retrieval
                filtered_results = [
                    (doc, score) for doc, score in results_with_scores 
                    if score <= distance_threshold
                ]
                
                # Sort by score and take top k
                filtered_results.sort(key=lambda x: x[1])
                documents = [doc for doc, score in filtered_results[:k]]
                
                logger.info(f"Retrieved {len(documents)} documents after filtering (threshold: {similarity_threshold})")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def get_relevant_context(self, 
                           query: str, 
                           k: int = 5, 
                           max_tokens: int = 4000,
                           use_compression: bool = False,
                           similarity_threshold: float = 0.7) -> str:
        """Get relevant context as a single string with token limit"""
        try:
            documents = self.retrieve_documents(
                query=query,
                k=k,
                use_compression=use_compression,
                similarity_threshold=similarity_threshold
            )
            
            if not documents:
                return "No relevant context found."
            
            # Combine documents into context
            context_parts = []
            total_tokens = 0
            
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                
                # Estimate tokens (roughly 4 characters per token)
                estimated_tokens = len(content) // 4
                
                if total_tokens + estimated_tokens <= max_tokens:
                    context_parts.append(f"Source {i+1}:\n{content}")
                    total_tokens += estimated_tokens
                else:
                    # Truncate the last document to fit within token limit
                    remaining_tokens = max_tokens - total_tokens
                    remaining_chars = remaining_tokens * 4
                    
                    if remaining_chars > 100:  # Only add if meaningful content can fit
                        truncated_content = content[:remaining_chars] + "..."
                        context_parts.append(f"Source {i+1}:\n{truncated_content}")
                    
                    break
            
            context = "\n\n".join(context_parts)
            logger.info(f"Generated context from {len(context_parts)} sources, ~{total_tokens} tokens")
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            raise
    
    def get_document_metadata(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get metadata of retrieved documents"""
        try:
            documents = self.retrieve_documents(query, k)
            
            metadata_list = []
            for doc in documents:
                metadata = doc.metadata.copy()
                metadata['content_preview'] = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                metadata['content_length'] = len(doc.page_content)
                metadata_list.append(metadata)
            
            return metadata_list
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            raise
    
    def search_with_scores_and_metadata(self, 
                                      query: str, 
                                      k: int = 5,
                                      similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search and return documents with scores and metadata"""
        try:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k * 2
            )
            
            # Filter and format results
            filtered_results = []
            for doc, score in results_with_scores:
                if score <= 2.0:  # More lenient distance threshold
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'relevance_score': 1 - score,  # Convert distance to similarity
                        'content_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    filtered_results.append(result)
            
            # Sort by relevance score (descending)
            filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching with scores and metadata: {e}")
            raise
    
    def update_search_params(self, search_kwargs: Dict[str, Any]):
        """Update search parameters for the base retriever"""
        try:
            self.base_retriever = self.vector_store.get_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            
            # Update compression retriever if it exists
            if self.compression_retriever and self.llm:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.base_retriever
                )
            
            logger.info("Search parameters updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating search parameters: {e}")
            raise
