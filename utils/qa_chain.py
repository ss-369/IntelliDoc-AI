import os
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAChain:
    """Question Answering chain using Gemini LLM"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Default prompt template
        self.prompt_template = """You are an AI assistant tasked with answering questions based on provided context. 
Please follow these guidelines:

1. Answer the question using ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say "I cannot answer this question based on the provided context"
3. Be concise but comprehensive in your response
4. Cite specific parts of the context when relevant
5. Do not use any knowledge outside of the provided context

Context:
{context}

Question: {question}

Answer:"""
    
    def set_custom_prompt(self, prompt_template: str):
        """Set a custom prompt template"""
        self.prompt_template = prompt_template
        logger.info("Custom prompt template set")
    
    def generate_answer(self, question: str, context: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Generate answer using Gemini LLM"""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            if not context.strip():
                return {
                    "answer": "No context provided to answer the question.",
                    "context_used": context,
                    "question": question,
                    "model": self.model_name,
                    "error": "No context"
                }
            
            # Format the prompt
            formatted_prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1000
                )
            )
            
            if not response.text:
                raise ValueError("Empty response from model")
            
            result = {
                "answer": response.text.strip(),
                "context_used": context,
                "question": question,
                "model": self.model_name,
                "temperature": temperature,
                "prompt_used": formatted_prompt,
                "success": True
            }
            
            logger.info("Successfully generated answer using Gemini")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context_used": context,
                "question": question,
                "model": self.model_name,
                "error": str(e),
                "success": False
            }
    
    def answer_with_sources(self, question: str, documents: List[Document], temperature: float = 0.3) -> Dict[str, Any]:
        """Generate answer with source attribution"""
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            if not documents:
                return {
                    "answer": "No documents provided to generate answer.",
                    "sources": [],
                    "question": question,
                    "model": self.model_name,
                    "error": "No documents",
                    "success": False
                }
            
            # Create context from documents
            context_parts = []
            sources = []
            
            for i, doc in enumerate(documents):
                # Add document content to context
                context_parts.append(f"Source {i+1}: {doc.page_content}")
                
                # Create source information
                source_info = {
                    "source_id": i + 1,
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using the context
            result = self.generate_answer(question, context, temperature)
            
            # Add sources to the result
            result["sources"] = sources
            result["num_sources"] = len(sources)
            
            logger.info(f"Successfully generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer with sources: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "question": question,
                "model": self.model_name,
                "error": str(e),
                "success": False
            }
    
    def batch_qa(self, questions: List[str], context: str, temperature: float = 0.3) -> List[Dict[str, Any]]:
        """Answer multiple questions with the same context"""
        results = []
        
        for question in questions:
            try:
                result = self.generate_answer(question, context, temperature)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch QA for question '{question}': {e}")
                results.append({
                    "answer": f"Error: {str(e)}",
                    "context_used": context,
                    "question": question,
                    "model": self.model_name,
                    "error": str(e),
                    "success": False
                })
        
        logger.info(f"Completed batch QA for {len(questions)} questions")
        return results
    
    def validate_answer_quality(self, answer: str, question: str, context: str) -> Dict[str, Any]:
        """Basic validation of answer quality"""
        try:
            quality_metrics = {
                "answer_length": len(answer),
                "has_answer": len(answer.strip()) > 0,
                "mentions_context": any(word in answer.lower() for word in context.lower().split()[:50]),
                "says_cannot_answer": "cannot answer" in answer.lower() or "don't know" in answer.lower(),
                "answer_to_question_ratio": len(answer) / len(question) if len(question) > 0 else 0
            }
            
            # Basic quality score
            score = 0
            if quality_metrics["has_answer"]:
                score += 0.3
            if quality_metrics["mentions_context"] and not quality_metrics["says_cannot_answer"]:
                score += 0.4
            if 50 <= quality_metrics["answer_length"] <= 500:
                score += 0.3
            
            quality_metrics["quality_score"] = score
            quality_metrics["is_good_quality"] = score >= 0.7
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error validating answer quality: {e}")
            return {"error": str(e)}
