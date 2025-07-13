import logging
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruLensEvaluator:
    """TruLens evaluation for RAG systems"""
    
    def __init__(self):
        self.available_metrics = [
            "answer_relevance",
            "context_relevance", 
            "groundedness",
            "context_recall"
        ]
        
        # Try to import TruLens
        try:
            import trulens_eval
            from trulens_eval import Feedback, feedback
            # Try different import paths for providers
            try:
                from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
                from trulens_eval.feedback.provider.hugs import Huggingface
            except ImportError:
                # Try alternative import paths
                try:
                    from trulens.providers.openai import OpenAI as fOpenAI
                    from trulens.providers.huggingface import Huggingface
                except ImportError:
                    fOpenAI = None
                    Huggingface = None
            
            # Check for API keys
            gemini_key = os.getenv("GEMINI_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            # Initialize feedback provider
            if openai_key and fOpenAI:
                try:
                    self.provider = fOpenAI()
                    logger.info("TruLens initialized with OpenAI provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI provider: {e}")
                    self.provider = self._create_gemini_provider()
                    logger.info("TruLens initialized with custom Gemini provider")
            elif gemini_key:
                # For Gemini, we'll use a custom implementation
                self.provider = self._create_gemini_provider()
                logger.info("TruLens initialized with custom Gemini provider")
            else:
                # Use Huggingface as fallback or custom provider
                try:
                    if Huggingface:
                        self.provider = Huggingface()
                        logger.info("TruLens initialized with Huggingface provider")
                    else:
                        self.provider = self._create_gemini_provider()
                        logger.info("TruLens initialized with custom Gemini provider")
                except:
                    self.provider = self._create_gemini_provider()
                    logger.info("TruLens initialized with custom Gemini provider")
            
            self.trulens_available = True
            self.feedback = feedback
            logger.info("TruLens evaluation framework loaded successfully")
            
        except ImportError as e:
            logger.warning(f"TruLens not available: {e}")
            self.trulens_available = False
            self.provider = None
            self.feedback = None
            
        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}")
            self.ragas_available = False
            self.evaluate = None
            self.metrics_map = {}
    
    def _create_gemini_provider(self):
        """Create a simple provider for Gemini compatibility"""
        class GeminiProvider:
            def __init__(self):
                from utils.qa_chain import QAChain
                self.qa_chain = QAChain()
            
            def relevance(self, prompt: str, response: str) -> float:
                # Simple relevance check using Gemini
                eval_prompt = f"""Rate the relevance of this response to the question on a scale of 0-1:
                Question: {prompt}
                Response: {response}
                
                Return only a number between 0 and 1."""
                
                try:
                    result = self.qa_chain.generate_answer(eval_prompt, "", temperature=0.1)
                    score_text = result.get('answer', '0.5')
                    return float(score_text.strip())
                except:
                    return 0.5
        
        return GeminiProvider()
    
    def prepare_evaluation_data(self, 
                               questions: List[str],
                               answers: List[str], 
                               contexts: List[List[str]],
                               ground_truths: Optional[List[str]] = None) -> pd.DataFrame:
        """Prepare data for TruLens evaluation"""
        try:
            if len(questions) != len(answers) or len(questions) != len(contexts):
                raise ValueError("All input lists must have the same length")
            
            data = {
                "question": questions,
                "answer": answers,
                "contexts": ["\n".join(ctx) if isinstance(ctx, list) else str(ctx) for ctx in contexts]
            }
            
            # Add ground truth if provided
            if ground_truths:
                if len(ground_truths) != len(questions):
                    raise ValueError("Ground truths must have the same length as questions")
                data["ground_truth"] = ground_truths
            
            df = pd.DataFrame(data)
            logger.info(f"Prepared evaluation data with {len(questions)} examples")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing evaluation data: {e}")
            raise
    
    def evaluate_rag_system(self, 
                           questions: List[str],
                           answers: List[str],
                           contexts: List[List[str]],
                           ground_truths: Optional[List[str]] = None,
                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate RAG system using TruLens metrics"""
        
        if not self.trulens_available:
            return self._enhanced_fallback_evaluation(questions, answers, contexts, ground_truths)
        
        try:
            # Prepare data
            data = self.prepare_evaluation_data(questions, answers, contexts, ground_truths)
            
            # Define TruLens feedback functions
            evaluation_results = {
                "overall_scores": {},
                "individual_scores": [],
                "summary": {}
            }
            
            logger.info("Running TruLens evaluation")
            
            # Calculate metrics for each question-answer pair
            answer_relevance_scores = []
            context_relevance_scores = []
            groundedness_scores = []
            
            for i, row in data.iterrows():
                question = row['question']
                answer = row['answer']
                context = row['contexts']
                
                # Answer Relevance: How relevant is the answer to the question
                try:
                    if hasattr(self.provider, 'relevance'):
                        rel_score = self.provider.relevance(question, answer)
                    else:
                        rel_score = self._calculate_relevance(question, answer)
                    answer_relevance_scores.append(rel_score)
                except:
                    answer_relevance_scores.append(0.5)
                
                # Context Relevance: How relevant is the context to the question
                try:
                    ctx_rel_score = self._calculate_context_relevance(question, context)
                    context_relevance_scores.append(ctx_rel_score)
                except:
                    context_relevance_scores.append(0.5)
                
                # Groundedness: How well is the answer supported by the context
                try:
                    ground_score = self._calculate_groundedness(answer, context)
                    groundedness_scores.append(ground_score)
                except:
                    groundedness_scores.append(0.5)
            
            # Calculate overall scores
            evaluation_results["overall_scores"] = {
                "answer_relevance": np.mean(answer_relevance_scores),
                "context_relevance": np.mean(context_relevance_scores),
                "groundedness": np.mean(groundedness_scores)
            }
            
            # Store individual scores
            for i in range(len(questions)):
                evaluation_results["individual_scores"].append({
                    "question": questions[i],
                    "answer_relevance": answer_relevance_scores[i],
                    "context_relevance": context_relevance_scores[i],
                    "groundedness": groundedness_scores[i]
                })
            
            # Calculate summary
            all_scores = [
                np.mean(answer_relevance_scores),
                np.mean(context_relevance_scores),
                np.mean(groundedness_scores)
            ]
            
            evaluation_results["summary"] = {
                "average_score": np.mean(all_scores),
                "min_score": np.min(all_scores),
                "max_score": np.max(all_scores),
                "std_score": np.std(all_scores),
                "total_questions": len(questions)
            }
            
            logger.info("TruLens evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in TruLens evaluation: {e}")
            return self._enhanced_fallback_evaluation(questions, answers, contexts, ground_truths)
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate relevance between question and answer"""
        try:
            from utils.qa_chain import QAChain
            qa_chain = QAChain()
            
            eval_prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0:
            Question: {question}
            Answer: {answer}
            
            Consider:
            - Does the answer directly address the question?
            - Is the answer complete and helpful?
            - Is the answer accurate and relevant?
            
            Respond with only a number between 0.0 and 1.0."""
            
            result = qa_chain.generate_answer(eval_prompt, "", temperature=0.1)
            score_text = result.get('answer', '0.5').strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return min(float(numbers[0]), 1.0)
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.5
    
    def _calculate_context_relevance(self, question: str, context: str) -> float:
        """Calculate how relevant the context is to the question"""
        try:
            from utils.qa_chain import QAChain
            qa_chain = QAChain()
            
            eval_prompt = f"""Rate how relevant this context is for answering the question on a scale of 0.0 to 1.0:
            Question: {question}
            Context: {context[:1000]}...
            
            Consider:
            - Does the context contain information needed to answer the question?
            - Is the context directly related to the question topic?
            - How much of the context is useful for answering?
            
            Respond with only a number between 0.0 and 1.0."""
            
            result = qa_chain.generate_answer(eval_prompt, "", temperature=0.1)
            score_text = result.get('answer', '0.5').strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return min(float(numbers[0]), 1.0)
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating context relevance: {e}")
            return 0.5
    
    def _calculate_groundedness(self, answer: str, context: str) -> float:
        """Calculate how well the answer is grounded in the provided context"""
        try:
            from utils.qa_chain import QAChain
            qa_chain = QAChain()
            
            eval_prompt = f"""Rate how well this answer is supported by the given context on a scale of 0.0 to 1.0:
            Context: {context[:1000]}...
            Answer: {answer}
            
            Consider:
            - Are the facts in the answer supported by the context?
            - Does the answer contain information not found in the context?
            - Is the answer based on the provided context or external knowledge?
            
            1.0 = Fully supported by context
            0.5 = Partially supported
            0.0 = Not supported or contradicts context
            
            Respond with only a number between 0.0 and 1.0."""
            
            result = qa_chain.generate_answer(eval_prompt, "", temperature=0.1)
            score_text = result.get('answer', '0.5').strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return min(float(numbers[0]), 1.0)
            return 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating groundedness: {e}")
            return 0.5
    
    def _enhanced_fallback_evaluation(self, 
                           questions: List[str],
                           answers: List[str], 
                           contexts: List[List[str]],
                           ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enhanced fallback evaluation when RAGAS/OpenAI is not available"""
        logger.info("Using enhanced fallback evaluation metrics (RAGAS requires OpenAI API key)")
        
        try:
            evaluation_results = {
                "overall_scores": {},
                "individual_scores": {},
                "summary": {},
                "note": "Enhanced evaluation metrics (RAGAS requires OpenAI API key for full functionality)"
            }
            
            # Enhanced metrics
            answer_lengths = [len(answer) for answer in answers]
            context_lengths = [sum(len(ctx) for ctx in context_list) for context_list in contexts]
            
            # Quality metrics
            non_empty_answers = sum(1 for answer in answers if answer.strip())
            answers_with_context = sum(1 for i, answer in enumerate(answers) 
                                     if any(word in answer.lower() for ctx in contexts[i] 
                                          for word in ctx.lower().split()[:20]))
            
            # Answer quality analysis
            detailed_answers = sum(1 for answer in answers if len(answer) > 50)
            answers_with_sources = sum(1 for answer in answers if 'source' in answer.lower())
            
            # Context relevance estimation
            avg_context_per_answer = np.mean([len(contexts[i]) for i in range(len(answers))])
            
            evaluation_results["overall_scores"] = {
                "answer_completeness": non_empty_answers / len(answers),
                "context_utilization": answers_with_context / len(answers),
                "answer_detail_score": detailed_answers / len(answers),
                "source_attribution_score": answers_with_sources / len(answers),
                "average_answer_length": np.mean(answer_lengths),
                "average_context_length": np.mean(context_lengths),
                "context_richness": min(avg_context_per_answer / 3.0, 1.0)  # Normalized
            }
            
            # Calculate composite scores
            quality_scores = [
                evaluation_results["overall_scores"]["answer_completeness"],
                evaluation_results["overall_scores"]["context_utilization"],
                evaluation_results["overall_scores"]["answer_detail_score"],
                evaluation_results["overall_scores"]["context_richness"]
            ]
            
            evaluation_results["summary"] = {
                "overall_quality_score": np.mean(quality_scores),
                "total_questions": len(questions),
                "non_empty_answers": non_empty_answers,
                "answers_using_context": answers_with_context,
                "detailed_answers": detailed_answers,
                "answers_with_sources": answers_with_sources
            }
            
            logger.info("Fallback evaluation completed")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in fallback evaluation: {e}")
            return {"error": str(e)}
    
    def evaluate_single_qa(self, 
                          question: str,
                          answer: str, 
                          context: List[str],
                          ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single question-answer pair"""
        try:
            return self.evaluate_rag_system(
                questions=[question],
                answers=[answer],
                contexts=[context],
                ground_truths=[ground_truth] if ground_truth else None
            )
        except Exception as e:
            logger.error(f"Error evaluating single QA: {e}")
            return {"error": str(e)}
    
    def batch_evaluate(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of QA results"""
        try:
            questions = []
            answers = []
            contexts = []
            
            for result in qa_results:
                questions.append(result.get("question", ""))
                answers.append(result.get("answer", ""))
                
                # Extract context from different possible formats
                context = result.get("context_used", "")
                if isinstance(context, str):
                    contexts.append([context])
                elif isinstance(context, list):
                    contexts.append(context)
                else:
                    contexts.append([""])
            
            return self.evaluate_rag_system(questions, answers, contexts)
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return {"error": str(e)}
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        try:
            if "error" in evaluation_results:
                return f"Evaluation Error: {evaluation_results['error']}"
            
            report = ["RAG System Evaluation Report", "=" * 40, ""]
            
            # Overall scores
            if "overall_scores" in evaluation_results:
                report.append("Overall Scores:")
                for metric, score in evaluation_results["overall_scores"].items():
                    report.append(f"  {metric}: {score:.3f}")
                report.append("")
            
            # Summary
            if "summary" in evaluation_results:
                summary = evaluation_results["summary"]
                report.append("Summary:")
                for key, value in summary.items():
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.3f}")
                    else:
                        report.append(f"  {key}: {value}")
                report.append("")
            
            # Notes
            if "note" in evaluation_results:
                report.append(f"Note: {evaluation_results['note']}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return f"Error generating report: {e}"
    
    def get_metrics_info(self) -> Dict[str, str]:
        """Get information about available metrics"""
        metrics_info = {
            "answer_relevance": "Measures how relevant the answer is to the question",
            "context_relevance": "Measures how relevant the retrieved context is to the question", 
            "groundedness": "Measures how well the answer is supported by the provided context",
            "context_recall": "Measures completeness of retrieved context"
        }
        
        return {
            "available_metrics": self.available_metrics,
            "descriptions": metrics_info,
            "trulens_available": self.trulens_available
        }
