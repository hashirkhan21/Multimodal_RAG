"""
Multimodal RAG System - Complete Implementation
Supports text and image queries, advanced prompting, evaluation metrics, and visualization
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Union, Tuple
import torch
from sentence_transformers import SentenceTransformer
import clip
from PIL import Image
import chromadb
from chromadb.errors import ChromaError
import gradio as gr
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64
from io import BytesIO
import inspect

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# LLM Integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import ollama
except ImportError:
    ollama = None

import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the RAG system"""
    CHROMA_PATH = "extracted_data/chroma_db"
    EMBEDDINGS_PATH = "extracted_data/embeddings"
    METADATA_PATH = "extracted_data/metadata"
    IMAGES_PATH = "extracted_data/charts"
    DATA_PATH = "Data"
    
    # Model configurations
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "ViT-B/32"
    
    # Retrieval settings
    DEFAULT_TOP_K = 5
    SIMILARITY_THRESHOLD = 0.3
    
    # LLM settings
    USE_OPENAI = False  # Using Ollama instead
    USE_OLLAMA = True
    OLLAMA_MODEL = "llama3.2"  # Your installed model
    OLLAMA_BASE_URL = "http://localhost:11434"
    OPENAI_MODEL = "gpt-4"  # Fallback if USE_OPENAI is True
    TEMPERATURE = 0.7
    MAX_TOKENS = 1000

config = Config()

# Prompt strategy configuration
PROMPT_STRATEGY_OPTIONS = [
    ("Zero-shot", "zero_shot"),
    ("Few-shot", "few_shot"),
    ("Chain-of-Thought", "cot"),
    ("CoT + Few-shot", "cot_few_shot")
]
PROMPT_LABEL_TO_VALUE = {label: value for label, value in PROMPT_STRATEGY_OPTIONS}
PROMPT_VALUE_TO_LABEL = {value: label for label, value in PROMPT_STRATEGY_OPTIONS}
DEFAULT_PROMPT_STRATEGY_LABEL = PROMPT_STRATEGY_OPTIONS[0][0]
DEFAULT_PROMPT_STRATEGY_VALUE = PROMPT_STRATEGY_OPTIONS[0][1]


def load_test_queries(config: Config) -> List[Dict[str, Any]]:
    """Load test queries with evaluation metadata."""
    test_queries_path = Path("test_queries.json")
    if not test_queries_path.exists():
        return []
    
    try:
        with open(test_queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("test_queries", [])
    except Exception as exc:
        print(f"⚠️ Could not load test queries: {exc}")
        return []

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages all models and embeddings"""
    
    def __init__(self):
        print("🔄 Loading models...")
        
        # Load text embedding model
        self.text_model = SentenceTransformer(config.TEXT_MODEL_NAME)
        
        # Load CLIP model for image embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(config.CLIP_MODEL_NAME, device=self.device)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.text_collection = self.chroma_client.get_collection("text_chunks")
        self.table_collection = self.chroma_client.get_collection("table_chunks")
        self.image_collection = self.chroma_client.get_collection("image_chunks")
        
        # Load metadata
        self.load_metadata()
        
        # Initialize LLM
        self.init_llm()
        
        print("✅ Models loaded successfully!")
    
    def load_metadata(self):
        """Load all metadata files"""
        # Load text metadata
        with open(f"{config.METADATA_PATH}/text_chunks.json", 'r', encoding='utf-8') as f:
            self.text_metadata = json.load(f)
        
        # Load table metadata
        with open(f"{config.METADATA_PATH}/table_chunks.json", 'r', encoding='utf-8') as f:
            self.table_metadata = json.load(f)
        
        # Load image metadata
        with open(f"{config.METADATA_PATH}/image_chunks.json", 'r', encoding='utf-8') as f:
            self.image_metadata = json.load(f)
        
        # Load extraction summary
        with open(f"{config.METADATA_PATH}/extraction_summary.json", 'r', encoding='utf-8') as f:
            self.extraction_summary = json.load(f)
    
    def init_llm(self):
        """Initialize the Language Model"""
        if config.USE_OLLAMA:
            # Try to connect to Ollama
            try:
                if ollama is None:
                    self.llm_available = False
                    print("⚠️ Ollama package not installed. Install with: pip install ollama")
                    return
                
                # Test Ollama connection
                try:
                    ollama.list()
                    self.llm_client = ollama
                    self.llm_available = True
                    print(f"✅ Ollama LLM initialized (model: {config.OLLAMA_MODEL})")
                    print(f"   Using base URL: {config.OLLAMA_BASE_URL}")
                except Exception as e:
                    self.llm_available = False
                    print(f"⚠️ Could not connect to Ollama: {e}")
                    print("   Make sure Ollama is running: ollama serve")
            except Exception as e:
                self.llm_available = False
                print(f"⚠️ Error initializing Ollama: {e}")
        elif config.USE_OPENAI:
            # Check for OpenAI API key
            if OpenAI is None:
                self.llm_available = False
                print("⚠️ OpenAI package not installed. Install with: pip install openai")
                return
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = OpenAI(api_key=api_key)
                self.llm_available = True
                print("✅ OpenAI LLM initialized")
            else:
                self.llm_available = False
                print("⚠️ OpenAI API key not found. LLM features disabled.")
        else:
            self.llm_available = False
            print("ℹ️ No LLM configured. Set USE_OLLAMA=True or USE_OPENAI=True.")

# Initialize global model manager
model_manager = ModelManager()

# ============================================================================
# RETRIEVAL ENGINE
# ============================================================================

class RetrievalEngine:
    """Handles all retrieval operations"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.query_history = []
    
    def _safe_query(self, collection, **kwargs) -> Dict[str, Any]:
        """Query Chroma collection with graceful fallback when index is missing."""
        try:
            return collection.query(**kwargs)
        except (ChromaError, FileNotFoundError) as exc:
            print(f"⚠️ Chroma query failed: {exc}")
        except Exception as exc:
            print(f"⚠️ Unexpected query error: {exc}")
        # Return empty structures compatible with downstream logic
        return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    def search_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Text-to-text search"""
        query_embedding = self.model_manager.text_model.encode([query], normalize_embeddings=True)
        
        results = self._safe_query(
            self.model_manager.text_collection,
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, "text")
    
    def search_tables(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Text-to-table search"""
        query_embedding = self.model_manager.text_model.encode([query], normalize_embeddings=True)
        
        results = self._safe_query(
            self.model_manager.table_collection,
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, "table")
    
    def search_images_by_text(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Text-to-image search using CLIP"""
        # Encode text query with CLIP
        text_tokens = clip.tokenize([query]).to(self.model_manager.device)
        with torch.no_grad():
            query_embedding = self.model_manager.clip_model.encode_text(text_tokens)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        query_embedding_np = query_embedding.cpu().numpy().astype('float32')
        
        results = self._safe_query(
            self.model_manager.image_collection,
            query_embeddings=query_embedding_np.tolist(),
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, "image")
    
    def search_images_by_image(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Image-to-image search using CLIP"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.model_manager.clip_preprocess(image).unsqueeze(0).to(self.model_manager.device)
        
        # Encode image with CLIP
        with torch.no_grad():
            query_embedding = self.model_manager.clip_model.encode_image(image_input)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        query_embedding_np = query_embedding.cpu().numpy().astype('float32')
        
        results = self._safe_query(
            self.model_manager.image_collection,
            query_embeddings=query_embedding_np.tolist(),
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, "image")
    
    def multimodal_search(self, query: str, image_path: str = None, top_k: int = 5) -> Dict[str, Any]:
        """Combined multimodal search across all content types"""
        all_results = {
            'text': [],
            'table': [],
            'image': [],
            'combined': []
        }
        
        if query:
            # Search text chunks
            text_results = self.search_text(query, top_k)
            all_results['text'] = text_results['results']
            
            # Search tables
            table_results = self.search_tables(query, top_k)
            all_results['table'] = table_results['results']
            
            # Search images by text
            image_results = self.search_images_by_text(query, top_k)
            all_results['image'] = image_results['results']
        
        if image_path:
            # Search images by image
            image_results = self.search_images_by_image(image_path, top_k)
            all_results['image'].extend(image_results['results'])
        
        # Combine and rank all results
        combined = []
        for result_type in ['text', 'table', 'image']:
            for result in all_results[result_type]:
                result['type'] = result_type
                combined.append(result)
        
        # Sort by similarity score (lower distance = higher similarity)
        combined.sort(key=lambda x: x['distance'])
        all_results['combined'] = combined[:top_k * 2]
        
        # Log query
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'image_query': image_path is not None,
            'num_results': len(all_results['combined'])
        })
        
        return all_results
    
    def _format_results(self, results: Dict, result_type: str) -> Dict[str, Any]:
        """Format raw ChromaDB results"""
        formatted_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return {'results': [], 'count': 0}
        
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                'type': result_type
            }
            formatted_results.append(result)
        
        return {
            'results': formatted_results,
            'count': len(formatted_results)
        }

# Initialize retrieval engine
retrieval_engine = RetrievalEngine(model_manager)

# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

class PromptEngine:
    """Handles different prompting strategies"""
    
    @staticmethod
    def zero_shot_prompt(query: str, context: str) -> str:
        """Zero-shot prompting"""
        return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    @staticmethod
    def few_shot_prompt(query: str, context: str) -> str:
        """Few-shot prompting with examples"""
        examples = """Example 1:
Question: What was the revenue in 2023?
Context: The company reported total revenue of $150 million in fiscal year 2023.
Answer: The company's revenue in 2023 was $150 million.

Example 2:
Question: What are the computer science programs offered?
Context: The department offers BS Computer Science, MS Computer Science, and PhD programs.
Answer: The department offers three computer science programs: BS Computer Science, MS Computer Science, and PhD.

"""
        return f"""{examples}
Now answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
    
    @staticmethod
    def chain_of_thought_prompt(query: str, context: str) -> str:
        """Chain-of-Thought prompting"""
        return f"""Answer the following question based on the provided context. Think step by step and explain your reasoning.

Context:
{context}

Question: {query}

Let's think step by step:
1. First, identify the relevant information in the context
2. Then, analyze how it relates to the question
3. Finally, provide a clear and comprehensive answer

Answer:"""
    
    @staticmethod
    def cot_few_shot_prompt(query: str, context: str) -> str:
        """Combined Chain-of-Thought and Few-shot prompting"""
        examples = """Example:
Question: What was the percentage increase in revenue?
Context: Revenue was $100M in 2022 and $150M in 2023.
Reasoning: 
1. Identify the values: 2022 = $100M, 2023 = $150M
2. Calculate increase: $150M - $100M = $50M
3. Calculate percentage: ($50M / $100M) × 100 = 50%
Answer: The revenue increased by 50% from 2022 to 2023.

"""
        return f"""{examples}
Now answer the following question using the same step-by-step reasoning approach.

Context:
{context}

Question: {query}

Reasoning:
1. Identify relevant information
2. Analyze the relationships
3. Draw conclusions

Answer:"""
    
    @staticmethod
    def get_prompt(strategy: str, query: str, context: str) -> str:
        """Get prompt based on strategy"""
        strategies = {
            'zero_shot': PromptEngine.zero_shot_prompt,
            'few_shot': PromptEngine.few_shot_prompt,
            'cot': PromptEngine.chain_of_thought_prompt,
            'cot_few_shot': PromptEngine.cot_few_shot_prompt
        }
        
        return strategies.get(strategy, PromptEngine.zero_shot_prompt)(query, context)

prompt_engine = PromptEngine()

# ============================================================================
# LLM GENERATION
# ============================================================================

class LLMGenerator:
    """Handles LLM response generation"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.generation_history = []
    
    def generate_response(self, query: str, retrieved_results: List[Dict], 
                         prompt_strategy: str = "zero_shot") -> Dict[str, Any]:
        """Generate response using LLM"""
        
        if not self.model_manager.llm_available:
            return {
                'answer': "LLM not available. Please configure OpenAI API key or local LLM.",
                'context_used': [],
                'prompt_strategy': prompt_strategy,
                'success': False
            }
        
        # Prepare context from retrieved results
        context_parts = []
        context_sources = []
        
        for i, result in enumerate(retrieved_results[:5], 1):  # Use top 5 results
            content = result['content']
            metadata = result['metadata']
            result_type = result.get('type', 'text')
            
            context_parts.append(f"[Source {i} - {result_type}]: {content}")
            context_sources.append({
                'source_num': i,
                'type': result_type,
                'pdf': _get_metadata_field(metadata, ['pdf', 'source_file']),
                'page': _get_metadata_field(metadata, ['page', 'page_number']),
                'similarity': result.get('similarity', 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate prompt based on strategy
        prompt = prompt_engine.get_prompt(prompt_strategy, query, context)
        
        try:
            # Call LLM API (Ollama or OpenAI)
            if config.USE_OLLAMA:
                # Call Ollama API
                response = self.model_manager.llm_client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be precise, informative, and cite sources when relevant."},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        'temperature': config.TEMPERATURE,
                        'num_predict': config.MAX_TOKENS
                    }
                )
                answer = response['message']['content']
            else:
                # Call OpenAI API
                response = self.model_manager.llm_client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be precise, informative, and cite sources when relevant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                answer = response.choices[0].message.content
            
            # Log generation
            self.generation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'prompt_strategy': prompt_strategy,
                'num_sources': len(context_sources),
                'answer_length': len(answer)
            })
            
            return {
                'answer': answer,
                'context_used': context_sources,
                'prompt_strategy': prompt_strategy,
                'full_prompt': prompt,
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'context_used': context_sources,
                'prompt_strategy': prompt_strategy,
                'success': False
            }
    
    def compare_prompt_strategies(self, query: str, retrieved_results: List[Dict]) -> Dict[str, Any]:
        """Compare different prompting strategies"""
        strategies = ['zero_shot', 'few_shot', 'cot', 'cot_few_shot']
        results = {}
        
        for strategy in strategies:
            response = self.generate_response(query, retrieved_results, strategy)
            results[strategy] = response
        
        return results

llm_generator = LLMGenerator(model_manager)

# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetrics:
    """Evaluation metrics for RAG system"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if len(relevant) == 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Average Precision"""
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                num_relevant += 1
                score += num_relevant / i
        
        return score / len(relevant)
    
    @staticmethod
    def mean_average_precision(queries_results: List[Tuple[List[str], List[str]]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        if len(queries_results) == 0:
            return 0.0
        
        ap_scores = []
        for retrieved, relevant in queries_results:
            ap = EvaluationMetrics.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        try:
            reference_tokens = nltk.word_tokenize(reference.lower())
            candidate_tokens = nltk.word_tokenize(candidate.lower())
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
            return score
        except:
            return 0.0
    
    @staticmethod
    def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    @staticmethod
    def evaluate_retrieval(
        results: List[Dict],
        ground_truth_ids: List[str] = None,
        ground_truth_pdfs: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate retrieval quality"""
        metrics = {
            'num_results': len(results),
            'avg_similarity': np.mean([r['similarity'] for r in results]) if results else 0,
            'min_similarity': np.min([r['similarity'] for r in results]) if results else 0,
            'max_similarity': np.max([r['similarity'] for r in results]) if results else 0,
            'precision@5': None,
            'recall@5': None,
            'precision@10': None,
            'recall@10': None,
            'average_precision': None
        }
        
        labels_for_metrics = None
        if ground_truth_ids:
            labels_for_metrics = {
                'retrieved': [r['id'] for r in results],
                'relevant': ground_truth_ids
            }
        elif ground_truth_pdfs:
            retrieved_pdfs = [
                _get_metadata_field(r['metadata'], ['pdf', 'source_file'])
                for r in results
            ]
            labels_for_metrics = {
                'retrieved': retrieved_pdfs,
                'relevant': ground_truth_pdfs
            }
        
        if labels_for_metrics:
            retrieved_labels = labels_for_metrics['retrieved']
            relevant_labels = labels_for_metrics['relevant']
            metrics['precision@5'] = EvaluationMetrics.precision_at_k(retrieved_labels, relevant_labels, 5)
            metrics['recall@5'] = EvaluationMetrics.recall_at_k(retrieved_labels, relevant_labels, 5)
            metrics['precision@10'] = EvaluationMetrics.precision_at_k(retrieved_labels, relevant_labels, 10)
            metrics['recall@10'] = EvaluationMetrics.recall_at_k(retrieved_labels, relevant_labels, 10)
            metrics['average_precision'] = EvaluationMetrics.average_precision(retrieved_labels, relevant_labels)
        
        return metrics

evaluation_metrics = EvaluationMetrics()

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_retrieval_scores(results: List[Dict]) -> plt.Figure:
        """Plot similarity scores of retrieved results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not results:
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
            return fig
        
        ids = [f"Result {i+1}\n{r['type']}" for i, r in enumerate(results)]
        similarities = [r['similarity'] for r in results]
        colors = ['#2ecc71' if s > 0.7 else '#f39c12' if s > 0.5 else '#e74c3c' for s in similarities]
        
        ax.barh(ids, similarities, color=colors)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Retrieved Results', fontsize=12)
        ax.set_title('Retrieval Similarity Scores', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, v in enumerate(similarities):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_embedding_space(embeddings: np.ndarray, labels: List[str], title: str = "Embedding Space") -> plt.Figure:
        """Plot 2D projection of embedding space using t-SNE"""
        from sklearn.manifold import TSNE
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if len(embeddings) < 2:
            ax.text(0.5, 0.5, 'Not enough data points', ha='center', va='center')
            return fig
        
        # Perform t-SNE
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(embeddings)), cmap='viridis', s=100, alpha=0.6)
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=8, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=ax, label='Result Index')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_prompt_comparison(comparison_results: Dict[str, Any]) -> plt.Figure:
        """Compare different prompting strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        strategies = list(comparison_results.keys())
        
        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            result = comparison_results[strategy]
            
            if result['success']:
                answer_length = len(result['answer'])
                num_sources = len(result['context_used'])
                
                metrics = ['Answer Length', 'Sources Used']
                values = [answer_length / 10, num_sources]  # Normalize length for display
                
                ax.bar(metrics, values, color=['#3498db', '#2ecc71'])
                ax.set_title(f"{strategy.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
                ax.set_ylabel('Value')
                
                # Add value labels
                for i, v in enumerate(values):
                    if i == 0:
                        ax.text(i, v + 5, f'{answer_length} chars', ha='center', va='bottom')
                    else:
                        ax.text(i, v + 0.1, f'{int(v)} sources', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'Generation Failed', ha='center', va='center')
                ax.set_title(f"{strategy.replace('_', ' ').title()}", fontsize=12)
        
        plt.suptitle('Prompt Strategy Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_metrics_dashboard(metrics: Dict[str, float]) -> go.Figure:
        """Create interactive metrics dashboard"""
        fig = go.Figure()
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker=dict(
                color=metric_values,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'{v:.3f}' for v in metric_values],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Evaluation Metrics Dashboard',
            xaxis_title='Metric',
            yaxis_title='Score',
            height=500,
            template='plotly_white'
        )
        
        return fig

visualizer = Visualizer()

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def process_query(query_text: str, query_image, top_k: int, prompt_strategy: str,
                  search_text_enabled: bool, search_tables_enabled: bool, 
                  search_images_enabled: bool):
    """Main function to process user queries"""
    
    strategy_value = PROMPT_LABEL_TO_VALUE.get(prompt_strategy, DEFAULT_PROMPT_STRATEGY_VALUE)
    strategy_label = PROMPT_VALUE_TO_LABEL.get(strategy_value, prompt_strategy)
    
    if not query_text and not query_image:
        return "Please provide a text query or upload an image.", None, None, None, None, None
    
    # Perform retrieval
    image_path = query_image if query_image else None
    results = retrieval_engine.multimodal_search(query_text, image_path, top_k)
    
    # Filter results based on enabled search types
    filtered_results = []
    if search_text_enabled:
        filtered_results.extend(results['text'])
    if search_tables_enabled:
        filtered_results.extend(results['table'])
    if search_images_enabled:
        filtered_results.extend(results['image'])
    
    # Sort by similarity
    filtered_results.sort(key=lambda x: x['distance'])
    filtered_results = filtered_results[:top_k]
    
    if not filtered_results:
        return "No results found.", None, None, None, None, None
    
    # Generate LLM response
    llm_response = llm_generator.generate_response(query_text, filtered_results, strategy_value)
    response_strategy_label = PROMPT_VALUE_TO_LABEL.get(llm_response['prompt_strategy'], strategy_label)
    
    # Prepare response text
    response_text = f"**Answer ({response_strategy_label} prompting):**\n\n{llm_response['answer']}\n\n"
    response_text += "---\n\n**Retrieved Sources:**\n\n"
    
    for i, source in enumerate(llm_response['context_used'], 1):
        response_text += (
            f"{i}. **{source['type'].title()}** from {source['pdf']} "
            f"(Page {source['page']}) - Similarity: {source['similarity']:.3f}\n"
        )
    
    # Evaluate retrieval
    eval_metrics = evaluation_metrics.evaluate_retrieval(filtered_results)
    
    def _human_metric(value: float) -> str:
        return f"{value:.3f}" if value is not None else "N/A (needs labeled ground truth)"
    
    metrics_text = "**Retrieval Metrics:**\n\n"
    metrics_text += f"- Number of Results: {eval_metrics['num_results']}\n"
    metrics_text += f"- Average Similarity: {eval_metrics['avg_similarity']:.3f}\n"
    metrics_text += f"- Max Similarity: {eval_metrics['max_similarity']:.3f}\n"
    metrics_text += f"- Min Similarity: {eval_metrics['min_similarity']:.3f}\n"
    metrics_text += f"- Precision@5: {_human_metric(eval_metrics['precision@5'])}\n"
    metrics_text += f"- Recall@5: {_human_metric(eval_metrics['recall@5'])}\n"
    metrics_text += f"- Precision@10: {_human_metric(eval_metrics['precision@10'])}\n"
    metrics_text += f"- Recall@10: {_human_metric(eval_metrics['recall@10'])}\n"
    metrics_text += f"- Mean Average Precision (MAP): {_human_metric(eval_metrics['average_precision'])}\n"
    
    # Create visualizations
    similarity_plot = visualizer.plot_retrieval_scores(filtered_results[:10])
    
    # Prepare detailed results
    detailed_results = []
    for i, result in enumerate(filtered_results[:10], 1):
        metadata = result['metadata']
        pdf_name = _get_metadata_field(metadata, ['pdf', 'source_file'])
        page_num = _get_metadata_field(metadata, ['page', 'page_number'])
        detailed_results.append({
            'Rank': i,
            'Type': result['type'].title(),
            'PDF': pdf_name,
            'Page': page_num,
            'Similarity': f"{result['similarity']:.3f}",
            'Content': result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        })
    
    results_df = pd.DataFrame(detailed_results)
    
    # Get image results if any
    image_results = []
    for result in filtered_results:
        if result['type'] == 'image':
            image_file = result['metadata'].get('image_file')
            if not image_file:
                continue  # Skip entries that do not point to a concrete file
            img_path = os.path.join(config.IMAGES_PATH, image_file)
            if os.path.isfile(img_path):
                image_results.append((img_path, f"Similarity: {result['similarity']:.3f}"))
    
    # Append evaluation summary note
    metrics_text += "\nRetrieval quality is evaluated using Precision@K, Recall@K, and Mean Average Precision (MAP)."
    
    return response_text, metrics_text, similarity_plot, results_df, image_results if image_results else None, llm_response['full_prompt']

def compare_prompts(query_text: str, top_k: int):
    """Compare different prompting strategies"""
    
    if not query_text:
        return "Please provide a query.", None, None
    
    # Perform retrieval
    results = retrieval_engine.multimodal_search(query_text, None, top_k)
    filtered_results = results['combined'][:top_k]
    
    if not filtered_results:
        return "No results found.", None, None
    
    # Compare strategies
    comparison = llm_generator.compare_prompt_strategies(query_text, filtered_results)
    
    # Prepare comparison text
    comparison_text = "**Prompt Strategy Comparison:**\n\n"
    
    for strategy, result in comparison.items():
        strategy_label = PROMPT_VALUE_TO_LABEL.get(strategy, strategy.replace('_', ' ').title())
        comparison_text += f"### {strategy_label}\n\n"
        if result['success']:
            comparison_text += f"{result['answer']}\n\n"
            comparison_text += f"*Sources used: {len(result['context_used'])}*\n\n"
        else:
            comparison_text += f"Failed: {result['answer']}\n\n"
        comparison_text += "---\n\n"
    
    # Create comparison plot
    comparison_plot = visualizer.plot_prompt_comparison(comparison)
    
    # Prepare prompts log
    prompts_log = "**Full Prompts Used:**\n\n"
    for strategy, result in comparison.items():
        strategy_label = PROMPT_VALUE_TO_LABEL.get(strategy, strategy.replace('_', ' ').title())
        prompts_log += f"### {strategy_label}\n\n"
        prompts_log += f"```\n{result['full_prompt']}\n```\n\n"
        prompts_log += "---\n\n"
    
    return comparison_text, comparison_plot, prompts_log

def show_statistics():
    """Show system statistics"""
    
    stats_text = "## System Statistics\n\n"
    stats_text += f"**Data Summary:**\n"
    stats_text += f"- Total PDFs Processed: {model_manager.extraction_summary['total_pdfs_processed']}\n"
    stats_text += f"- Total Text Chunks: {model_manager.extraction_summary['total_text_chunks']}\n"
    stats_text += f"- Total Table Chunks: {model_manager.extraction_summary['total_table_chunks']}\n"
    stats_text += f"- Total Image Chunks: {model_manager.extraction_summary['total_image_chunks']}\n"
    stats_text += f"- Total Chunks: {model_manager.extraction_summary['total_chunks']}\n\n"
    
    stats_text += f"**Query History:**\n"
    stats_text += f"- Total Queries: {len(retrieval_engine.query_history)}\n"
    stats_text += f"- Total Generations: {len(llm_generator.generation_history)}\n\n"
    
    stats_text += f"**Models:**\n"
    stats_text += f"- Text Embedding: {config.TEXT_MODEL_NAME}\n"
    stats_text += f"- Image Embedding: CLIP {config.CLIP_MODEL_NAME}\n"
    
    if config.USE_OLLAMA:
        stats_text += f"- LLM: Ollama ({config.OLLAMA_MODEL})\n"
    elif config.USE_OPENAI:
        stats_text += f"- LLM: OpenAI ({config.OPENAI_MODEL})\n"
    else:
        stats_text += f"- LLM: Not configured\n"
    
    stats_text += f"- LLM Available: {'✅ Yes' if model_manager.llm_available else '❌ No'}\n"
    
    return stats_text

def evaluate_on_test_set():
    """Evaluate system on a test set of queries"""
    
    test_queries = load_test_queries(config)
    if not test_queries:
        return ("Unable to run evaluation because `test_queries.json` was not found "
                "or is empty. Please add labeled test queries first."), None
    
    results_summary = "## Evaluation Results\n\n"
    all_metrics = []
    
    for test_case in test_queries:
        query = test_case['query']
        results = retrieval_engine.multimodal_search(query, None, 10)
        filtered_results = results['combined'][:10]
        relevant_pdf = test_case.get('relevant_pdf')
        relevant_ids = test_case.get('relevant_ids', [])
        
        metrics = evaluation_metrics.evaluate_retrieval(
            filtered_results,
            ground_truth_ids=relevant_ids if relevant_ids else None,
            ground_truth_pdfs=[relevant_pdf] if relevant_pdf else None
        )
        
        all_metrics.append(metrics)
        
        results_summary += f"**Query:** {query}\n"
        results_summary += f"- Avg Similarity: {metrics['avg_similarity']:.3f}\n"
        results_summary += f"- Max Similarity: {metrics['max_similarity']:.3f}\n"
        if metrics['precision@5'] is not None:
            results_summary += f"- Precision@5: {metrics['precision@5']:.3f}\n"
            results_summary += f"- Recall@5: {metrics['recall@5']:.3f}\n"
            results_summary += f"- Precision@10: {metrics['precision@10']:.3f}\n"
            results_summary += f"- Recall@10: {metrics['recall@10']:.3f}\n"
            results_summary += f"- Average Precision: {metrics['average_precision']:.3f}\n"
        else:
            results_summary += "- Precision/Recall metrics require labeled ground truth for this query.\n"
        results_summary += "\n"
    
    # Calculate average metrics
    avg_metrics = {}
    all_keys = set()
    for metric in all_metrics:
        all_keys.update(metric.keys())
    
    for key in all_keys:
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            avg_metrics[key] = np.mean(values)
    
    results_summary += "\n**Average Metrics Across All Queries:**\n"
    for key, value in avg_metrics.items():
        pretty_name = key.replace('_', ' ').title()
        if key == 'num_results':
            results_summary += f"- Average {pretty_name}: {value:.1f}\n"
        else:
            results_summary += f"- Average {pretty_name}: {value:.3f}\n"
    
    # Create metrics dashboard
    dashboard = visualizer.create_metrics_dashboard(avg_metrics)
    
    return results_summary, dashboard

def _get_metadata_field(metadata: Dict[str, Any], keys: List[str], default: str = "Unknown") -> Any:
    """Return first available value from metadata for the provided keys."""
    for key in keys:
        if key in metadata and metadata[key] not in (None, "", []):
            return metadata[key]
    return default


# Build Gradio Interface
def _get_blocks_kwargs():
    """Return kwargs compatible with the installed Gradio version."""
    kwargs = {"title": "Multimodal RAG System"}
    try:
        signature = inspect.signature(gr.Blocks.__init__)
        if "theme" in signature.parameters and hasattr(gr, "themes"):
            soft_theme = getattr(getattr(gr, "themes", None), "Soft", None)
            if callable(soft_theme):
                kwargs["theme"] = soft_theme()
    except Exception:
        # Silently ignore incompatibilities to keep UI loading
        pass
    return kwargs


with gr.Blocks(**_get_blocks_kwargs()) as app:
    gr.Markdown("# 🚀 Multimodal Retrieval-Augmented Generation (RAG) System")
    gr.Markdown("Query documents using text or images, powered by advanced AI models and prompt engineering.")
    
    with gr.Tabs():
        # Main Query Tab
        with gr.Tab("🔍 Query Interface"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    query_input = gr.Textbox(
                        label="Enter your question",
                        placeholder="e.g., What are the computer science programs offered?",
                        lines=3
                    )
                    image_input = gr.Image(label="Or upload an image query", type="filepath")
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Number of results (top-k)")
                    
                    prompt_strategy_radio = gr.Radio(
                        choices=[label for label, _ in PROMPT_STRATEGY_OPTIONS],
                        value=DEFAULT_PROMPT_STRATEGY_LABEL,
                        label="Prompting Strategy"
                    )
                    
                    gr.Markdown("### Search Options")
                    search_text_check = gr.Checkbox(label="Search Text", value=True)
                    search_tables_check = gr.Checkbox(label="Search Tables", value=True)
                    search_images_check = gr.Checkbox(label="Search Images", value=True)
                    
                    search_btn = gr.Button("🔍 Search", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    response_output = gr.Markdown(label="Answer")
                    metrics_output = gr.Markdown(label="Metrics")
            
            with gr.Row():
                with gr.Column():
                    similarity_plot = gr.Plot(label="Similarity Scores")
                with gr.Column():
                    image_gallery = gr.Gallery(label="Retrieved Images", columns=3, height=400)
            
            with gr.Row():
                results_table = gr.Dataframe(label="Detailed Results")
            
            with gr.Accordion("View Full Prompt", open=False):
                prompt_display = gr.Textbox(label="Prompt sent to LLM", lines=10)
            
            search_btn.click(
                fn=process_query,
                inputs=[query_input, image_input, top_k_slider, prompt_strategy_radio,
                       search_text_check, search_tables_check, search_images_check],
                outputs=[response_output, metrics_output, similarity_plot, results_table, 
                        image_gallery, prompt_display]
            )
        
        # Prompt Comparison Tab
        with gr.Tab("📊 Prompt Strategy Comparison"):
            gr.Markdown("### Compare Different Prompting Strategies")
            gr.Markdown("Test how Zero-shot, Few-shot, CoT, and CoT+Few-shot prompting affect response quality.")
            
            with gr.Row():
                with gr.Column():
                    comparison_query = gr.Textbox(
                        label="Enter query to compare",
                        placeholder="e.g., What was the revenue in 2023?",
                        lines=2
                    )
                    comparison_top_k = gr.Slider(1, 10, value=5, step=1, label="Number of results")
                    compare_btn = gr.Button("🔄 Compare Strategies", variant="primary")
            
            comparison_output = gr.Markdown(label="Comparison Results")
            comparison_plot = gr.Plot(label="Strategy Comparison")
            
            with gr.Accordion("View All Prompts", open=False):
                prompts_log_output = gr.Textbox(label="Prompts Log", lines=20)
            
            compare_btn.click(
                fn=compare_prompts,
                inputs=[comparison_query, comparison_top_k],
                outputs=[comparison_output, comparison_plot, prompts_log_output]
            )
        
        # Evaluation Tab
        with gr.Tab("📈 Evaluation & Metrics"):
            gr.Markdown("### System Evaluation")
            gr.Markdown("Evaluate retrieval quality using Precision@K, Recall@K, and other metrics.")
            
            eval_btn = gr.Button("🧪 Run Evaluation", variant="primary")
            eval_output = gr.Markdown(label="Evaluation Results")
            eval_dashboard = gr.Plot(label="Metrics Dashboard")
            
            eval_btn.click(
                fn=evaluate_on_test_set,
                inputs=[],
                outputs=[eval_output, eval_dashboard]
            )
        
        # Statistics Tab
        with gr.Tab("📊 System Statistics"):
            gr.Markdown("### System Overview")
            
            stats_btn = gr.Button("📊 Show Statistics", variant="primary")
            stats_output = gr.Markdown(label="Statistics")
            
            stats_btn.click(
                fn=show_statistics,
                inputs=[],
                outputs=[stats_output]
            )
        
        # Documentation Tab
        with gr.Tab("📚 Documentation"):
            gr.Markdown("""
            ## System Architecture
            
            This Multimodal RAG system consists of the following components:
            
            ### 1. Data Processing Pipeline
            - **PDF Parsing**: Extracts text, tables, and images from PDF documents
            - **Chunking**: Splits content into manageable chunks with metadata
            - **OCR**: Extracts text from images and charts
            
            ### 2. Embedding Generation
            - **Text Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
            - **Image Embeddings**: CLIP (ViT-B/32)
            - **Normalization**: L2-normalized embeddings for cosine similarity
            
            ### 3. Vector Database
            - **Storage**: ChromaDB with persistent storage
            - **Collections**: Separate collections for text, tables, and images
            - **Indexing**: HNSW indexing for fast similarity search
            
            ### 4. Retrieval Engine
            - **Text-to-Text**: Query text chunks using semantic similarity
            - **Text-to-Image**: Query images using CLIP text encoder
            - **Image-to-Image**: Query images using CLIP image encoder
            - **Multimodal**: Combined search across all modalities
            
            ### 5. LLM Integration
            - **Model**: OpenAI GPT-4 (or local LLMs like LLaMA2, Mistral)
            - **Context**: Top-K retrieved results as context
            - **Prompting**: Multiple strategies (Zero-shot, Few-shot, CoT)
            
            ### 6. Evaluation Metrics
            - **Retrieval**: Precision@K, Recall@K, MAP
            - **Generation**: BLEU, ROUGE, Cosine Similarity
            - **Performance**: Query latency, response time
            
            ### 7. Advanced Features
            - **Prompt Engineering**: Zero-shot, Few-shot, Chain-of-Thought
            - **Source Linking**: References to original PDF pages
            - **Visualization**: Embedding space, similarity scores
            - **Evaluation**: Comprehensive metrics dashboard
            
            ## Usage Examples
            
            ### Text Query
            ```
            Query: "What are the admission requirements for MS programs?"
            Strategy: CoT + Few-shot
            ```
            
            ### Image Query
            ```
            Upload a chart image to find similar financial visualizations
            ```
            
            ### Multimodal Query
            ```
            Query: "Show me revenue trends"
            + Upload a sample chart
            Strategy: Zero-shot
            ```
            
            ## Configuration
            
            To use OpenAI LLM, set your API key:
            ```bash
            export OPENAI_API_KEY="your-api-key"
            ```
            
            Or modify `config.USE_OPENAI = False` to use local LLMs.
            """)
    
    gr.Markdown("---")
    gr.Markdown("*Developed as part of Multimodal RAG System Assignment | Powered by ChromaDB, CLIP, and GPT-4*")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting Multimodal RAG System")
    print("="*80 + "\n")
    
    print("📊 System Information:")
    print(f"  - Text Model: {config.TEXT_MODEL_NAME}")
    print(f"  - Image Model: CLIP {config.CLIP_MODEL_NAME}")
    
    if config.USE_OLLAMA:
        print(f"  - LLM: Ollama ({config.OLLAMA_MODEL})")
    elif config.USE_OPENAI:
        print(f"  - LLM: OpenAI ({config.OPENAI_MODEL})")
    else:
        print(f"  - LLM: Not configured")
    
    print(f"  - Device: {model_manager.device}")
    print(f"  - Total Documents: {model_manager.extraction_summary['total_chunks']}")
    print(f"  - LLM Available: {'✅' if model_manager.llm_available else '❌'}")
    
    print("\n" + "="*80)
    print("🌐 Launching Web Interface...")
    print("="*80 + "\n")
    
    # Launch Gradio app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
