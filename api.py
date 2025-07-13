from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import logging
from datetime import datetime

# Import your existing modules
from utils.config import Config
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.retriever import AdvancedRetriever
from utils.qa_chain import QAChain
from utils.evaluation import TruLensEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="IntelliDoc AI - Enterprise RAG API",
    description="Production-ready RAG system with document processing and intelligent Q&A capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7
    temperature: Optional[float] = 0.3
    include_sources: Optional[bool] = True

class QuestionResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None
    question: str
    success: bool
    timestamp: str
    processing_time: Optional[float] = None

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: str
    chunks_created: int
    success: bool
    filename: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    api_keys_configured: Dict[str, bool]
    vector_store_status: str
    document_count: int
    timestamp: str

class EvaluationRequest(BaseModel):
    questions: List[str]
    answers: List[str]
    contexts: List[str]

class EvaluationResponse(BaseModel):
    overall_scores: Dict[str, float]
    summary: Dict[str, Any]
    success: bool
    timestamp: str

# Global components (initialize on startup)
vector_store = None
retriever = None
qa_chain = None
document_processor = None
evaluator = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup"""
    global vector_store, retriever, qa_chain, document_processor, evaluator
    
    try:
        logger.info("Initializing IntelliDoc AI components...")
        
        # Validate API keys
        api_keys = Config.validate_api_keys()
        if not api_keys["gemini"]:
            logger.error("Gemini API key not found")
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        # Initialize components
        vector_store = VectorStore(
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_model=Config.EMBEDDING_MODEL
        )
        
        qa_chain = QAChain(
            api_key=Config.GEMINI_API_KEY,
            model_name=Config.GEMINI_MODEL
        )
        
        retriever = AdvancedRetriever(vector_store)
        
        document_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        evaluator = TruLensEvaluator()
        
        logger.info("IntelliDoc AI initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to IntelliDoc AI - Enterprise RAG API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        api_keys = Config.validate_api_keys()
        collection_info = vector_store.get_collection_info()
        
        return HealthResponse(
            status="healthy",
            api_keys_configured=api_keys,
            vector_store_status="connected",
            document_count=collection_info.get("document_count", 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document
        text = document_processor.load_document(tmp_file_path)
        documents = document_processor.create_documents(
            text, 
            metadata={"filename": file.filename, "upload_time": datetime.now().isoformat()}
        )
        
        # Add to vector store
        document_ids = vector_store.add_documents(documents)
        
        # Clean up temporary file
        background_tasks.add_task(os.unlink, tmp_file_path)
        
        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            document_id=document_ids[0] if document_ids else "unknown",
            chunks_created=len(documents),
            success=True,
            filename=file.filename,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        # Clean up on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/questions/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-generated answer"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        start_time = datetime.now()
        
        # Retrieve relevant documents
        documents = retriever.retrieve_documents(
            query=request.question,
            k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        if not documents:
            return QuestionResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                question=request.question,
                success=True,
                timestamp=datetime.now().isoformat(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Generate answer
        if request.include_sources:
            result = qa_chain.answer_with_sources(
                question=request.question,
                documents=documents,
                temperature=request.temperature
            )
        else:
            context = retriever.get_relevant_context(
                query=request.question,
                k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )
            result = qa_chain.generate_answer(
                question=request.question,
                context=context,
                temperature=request.temperature
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QuestionResponse(
            answer=result.get("answer", "Unable to generate answer"),
            sources=result.get("sources", []) if request.include_sources else None,
            question=request.question,
            success=result.get("success", False),
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Question processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.post("/evaluation/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """Run evaluation on provided Q&A pairs"""
    if not (request.questions and request.answers and request.contexts):
        raise HTTPException(status_code=400, detail="Questions, answers, and contexts are required")
    
    if not (len(request.questions) == len(request.answers) == len(request.contexts)):
        raise HTTPException(status_code=400, detail="Questions, answers, and contexts must have the same length")
    
    try:
        # Prepare evaluation data
        qa_pairs = []
        for q, a, c in zip(request.questions, request.answers, request.contexts):
            qa_pairs.append({
                "question": q,
                "answer": a,
                "context": c,
                "success": True
            })
        
        # Run evaluation
        results = evaluator.batch_evaluate(qa_pairs)
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return EvaluationResponse(
            overall_scores=results.get("overall_scores", {}),
            summary=results.get("summary", {}),
            success=True,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/documents/info")
async def get_document_info():
    """Get information about loaded documents"""
    try:
        collection_info = vector_store.get_collection_info()
        return JSONResponse(content={
            "success": True,
            "info": collection_info,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@app.delete("/documents/clear")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        vector_store.reset_collection()
        return JSONResponse(content={
            "message": "All documents cleared successfully",
            "success": True,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
