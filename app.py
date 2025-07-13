import streamlit as st
import tempfile
import os
import json
from typing import List, Dict, Any
import pandas as pd

# Import custom modules
from utils.config import Config
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.retriever import AdvancedRetriever
from utils.qa_chain import QAChain
from utils.evaluation import TruLensEvaluator

# Configure Streamlit page
st.set_page_config(
    page_title="RAG-based Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

def initialize_components():
    """Initialize RAG components"""
    try:
        # Validate API keys
        api_keys = Config.validate_api_keys()
        if not api_keys["gemini"]:
            st.error("⚠️ Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
            st.stop()
        
        # Initialize vector store
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore(
                collection_name=Config.COLLECTION_NAME,
                persist_directory=Config.PERSIST_DIRECTORY,
                embedding_model=Config.EMBEDDING_MODEL
            )
        
        # Initialize QA chain
        if st.session_state.qa_chain is None:
            st.session_state.qa_chain = QAChain(
                api_key=Config.GEMINI_API_KEY,
                model_name=Config.GEMINI_MODEL
            )
        
        # Initialize retriever
        if st.session_state.retriever is None:
            st.session_state.retriever = AdvancedRetriever(st.session_state.vector_store)
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return False

def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.title("⚙️ Configuration")
    
    # API Key status
    st.sidebar.subheader("API Keys Status")
    api_keys = Config.validate_api_keys()
    
    for service, available in api_keys.items():
        if available:
            st.sidebar.success(f"✅ {service.title()} API Key")
        else:
            st.sidebar.error(f"❌ {service.title()} API Key")
    
    st.sidebar.divider()
    
    # Vector store info
    st.sidebar.subheader("Vector Store Info")
    if st.session_state.vector_store:
        info = st.session_state.vector_store.get_collection_info()
        st.sidebar.info(f"📄 Documents: {info['document_count']}")
        st.sidebar.info(f"🧠 Model: {info['embedding_model'].split('/')[-1]}")
    
    st.sidebar.divider()
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    top_k = st.sidebar.slider("Top K Documents", 1, 10, Config.TOP_K)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.1)  # Lower default for better retrieval
    
    # Temperature setting
    st.sidebar.subheader("Generation Settings")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    
    # Reset collection button
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Reset Collection", type="secondary"):
        if st.session_state.vector_store:
            st.session_state.vector_store.reset_collection()
            st.session_state.documents_loaded = False
            st.session_state.qa_history = []
            st.sidebar.success("Collection reset successfully!")
            st.rerun()
    
    return {
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "temperature": temperature
    }

def document_upload_section():
    """Handle document upload and processing"""
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt'],
        help="Upload a PDF or TXT file to create your knowledge base"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        with col2:
            if st.button("🚀 Process Document", type="primary"):
                process_document(uploaded_file)

def process_document(uploaded_file):
    """Process uploaded document"""
    try:
        with st.spinner("Processing document..."):
            # Initialize document processor
            processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            
            # Process uploaded file
            text = processor.process_uploaded_file(uploaded_file)
            
            # Create metadata
            metadata = {
                "filename": uploaded_file.name,
                "file_type": uploaded_file.type,
                "upload_timestamp": pd.Timestamp.now().isoformat()
            }
            
            # Chunk the document
            documents = processor.chunk_text(text, metadata)
            
            # Add to vector store
            ids = st.session_state.vector_store.add_documents(documents)
            
            # Update session state
            st.session_state.documents_loaded = True
            
            # Get statistics
            stats = processor.get_document_stats(documents)
            
            # Show success message with stats
            st.success("✅ Document processed successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunks Created", stats['total_chunks'])
            with col2:
                st.metric("Total Words", f"{stats['total_words']:,}")
            with col3:
                st.metric("Avg Chunk Size", f"{stats['average_chunk_size']:.0f} chars")
            
    except Exception as e:
        st.error(f"❌ Error processing document: {e}")

def qa_section(settings):
    """Handle question answering"""
    st.header("❓ Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please upload and process a document first.")
        return
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about the document?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", disabled=not question.strip())
    
    with col2:
        show_sources = st.checkbox("Show Sources", value=True)
    
    if ask_button and question.strip():
        answer_question(question, settings, show_sources)
    
    # Display QA history
    if st.session_state.qa_history:
        st.subheader("📝 Q&A History")
        
        for i, qa_item in enumerate(reversed(st.session_state.qa_history)):
            with st.expander(f"Q: {qa_item['question'][:100]}..." if len(qa_item['question']) > 100 else f"Q: {qa_item['question']}"):
                st.write("**Answer:**")
                st.write(qa_item['answer'])
                
                if qa_item.get('sources') and show_sources:
                    st.write("**Sources:**")
                    for j, source in enumerate(qa_item['sources']):
                        st.write(f"*Source {j+1}:* {source['content_preview']}")

def answer_question(question: str, settings: dict, show_sources: bool):
    """Process question and generate answer"""
    try:
        with st.spinner("Searching for relevant information..."):
            # Retrieve relevant documents
            documents = st.session_state.retriever.retrieve_documents(
                query=question,
                k=settings['top_k'],
                similarity_threshold=settings['similarity_threshold']
            )
            
        if not documents:
            st.warning("⚠️ No relevant documents found for your question.")
            return
        
        with st.spinner("Generating answer..."):
            # Generate answer
            if show_sources:
                result = st.session_state.qa_chain.answer_with_sources(
                    question=question,
                    documents=documents,
                    temperature=settings['temperature']
                )
            else:
                context = st.session_state.retriever.get_relevant_context(
                    query=question,
                    k=settings['top_k'],
                    similarity_threshold=settings['similarity_threshold']
                )
                result = st.session_state.qa_chain.generate_answer(
                    question=question,
                    context=context,
                    temperature=settings['temperature']
                )
        
        if result['success']:
            # Display answer
            st.write("**Answer:**")
            st.write(result['answer'])
            
            # Display sources if requested
            if show_sources and 'sources' in result:
                st.write("**Sources:**")
                for i, source in enumerate(result['sources']):
                    with st.expander(f"Source {source['source_id']}: {source['content_preview'][:100]}..."):
                        st.write(source['content_preview'])
                        if 'metadata' in source:
                            st.json(source['metadata'])
            
            # Add to history
            st.session_state.qa_history.append(result)
            
        else:
            st.error(f"❌ Error generating answer: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"❌ Error processing question: {e}")

def evaluation_section():
    """Handle RAGAS evaluation"""
    st.header("📊 Evaluation")
    
    if not st.session_state.qa_history:
        st.warning("⚠️ No Q&A history available for evaluation. Ask some questions first.")
        return
    
    # Show TruLens info
    st.info("💡 **TruLens Evaluation**: Advanced RAG evaluation using answer relevance, context relevance, and groundedness metrics. Works with Gemini and provides comprehensive quality analysis.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔬 Run TruLens Evaluation", type="primary"):
            run_evaluation()
    
    with col2:
        if st.button("📈 Show Q&A Statistics", type="secondary"):
            show_qa_statistics()

def run_evaluation():
    """Run TruLens evaluation on Q&A history"""
    try:
        with st.spinner("Running evaluation..."):
            evaluator = TruLensEvaluator()
            
            # Run evaluation
            results = evaluator.batch_evaluate(st.session_state.qa_history)
            
            # Display results
            if "error" not in results:
                st.success("✅ Evaluation completed!")
                
                # Show overall scores
                if "overall_scores" in results:
                    st.subheader("📊 Overall Scores")
                    
                    # Display as a clean table instead of metrics to avoid truncation
                    scores_data = []
                    for metric, score in results["overall_scores"].items():
                        scores_data.append({
                            "Metric": metric.replace("_", " ").title(),
                            "Score": f"{score:.3f}",
                            "Percentage": f"{score*100:.1f}%"
                        })
                    
                    df_scores = pd.DataFrame(scores_data)
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)
                
                # Show summary
                if "summary" in results:
                    st.subheader("📈 Summary")
                    summary = results["summary"]
                    
                    # Create summary table for better readability
                    summary_data = [
                        {"Metric": "Average Score", "Value": f"{summary.get('average_score', 0):.3f}"},
                        {"Metric": "Total Questions", "Value": str(summary.get('total_questions', 0))},
                    ]
                    
                    if 'min_score' in summary and 'max_score' in summary:
                        summary_data.append({
                            "Metric": "Score Range", 
                            "Value": f"{summary['min_score']:.3f} - {summary['max_score']:.3f}"
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Show detailed report
                with st.expander("📄 Detailed Report"):
                    report = evaluator.generate_evaluation_report(results)
                    st.text(report)
                
                # Show metrics info
                with st.expander("ℹ️ Metrics Information"):
                    metrics_info = evaluator.get_metrics_info()
                    st.json(metrics_info)
            
            else:
                st.error(f"❌ Evaluation failed: {results['error']}")
                
    except Exception as e:
        st.error(f"❌ Error running evaluation: {e}")

def show_qa_statistics():
    """Show Q&A statistics"""
    try:
        if not st.session_state.qa_history:
            return
        
        st.subheader("📈 Q&A Statistics")
        
        # Basic statistics
        total_questions = len(st.session_state.qa_history)
        successful_answers = sum(1 for qa in st.session_state.qa_history if qa.get('success', False))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Successful Answers", successful_answers)
        with col3:
            success_rate = (successful_answers / total_questions) * 100 if total_questions > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Answer length distribution
        answer_lengths = [len(qa['answer']) for qa in st.session_state.qa_history]
        
        if answer_lengths:
            st.subheader("📏 Answer Length Distribution")
            df = pd.DataFrame({
                'Question': [qa['question'][:50] + "..." if len(qa['question']) > 50 else qa['question'] 
                           for qa in st.session_state.qa_history],
                'Answer Length': answer_lengths,
                'Success': [qa.get('success', False) for qa in st.session_state.qa_history]
            })
            
            st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error showing statistics: {e}")

def main():
    """Main application function"""
    # Title and description
    st.title("🤖 RAG-based Question Answering System")
    st.markdown("Upload documents, ask questions, and evaluate answers using RAGAS metrics.")
    
    # Initialize components
    if not initialize_components():
        st.stop()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📁 Document Upload", "❓ Question & Answer", "📊 Evaluation"])
    
    with tab1:
        document_upload_section()
    
    with tab2:
        qa_section(settings)
    
    with tab3:
        evaluation_section()
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with Streamlit, LangChain, ChromaDB, and Gemini | 
            Evaluation powered by TruLens
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
