"""
Ranker and Indexer Module for Document Information Retrieval System

Implements TF-IDF indexing from scratch and Vector Space Model ranking with 
Cosine Similarity. No external machine learning libraries are used.

Author: Senior IRS Algorithm Expert
Version: 1.0.0
"""

import json
import math
import logging
from typing import Any, Dict, List, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict
import re

from core.preprocessor import DocumentProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/ranker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Professional TF-IDF Search Engine using Vector Space Model with Cosine Similarity.
    
    Features:
    - Inverted index structure for efficient term lookups
    - TF-IDF weight calculation from scratch (no scikit-learn)
    - Vector Space Model with cosine similarity ranking
    - JSON persistence for index storage and recovery
    - Dynamic text snippet extraction for result previews
    
    Attributes:
        document_metadata (Dict): Stores document titles and raw text
        term_idf (Dict): Stores IDF weights for all terms
        document_vectors (Dict): Stores TF-IDF vectors for all documents
        document_count (int): Total number of indexed documents
        processor (DocumentProcessor): Text preprocessing pipeline
    """
    
    def __init__(self) -> None:
        """
        Initialize the SearchEngine.
        
        Creates empty data structures and initializes the document preprocessor.
        """
        self.document_metadata: Dict[int, Dict[str, any]] = {}
        self.term_idf: Dict[str, float] = {}
        self.document_vectors: Dict[int, Dict[str, float]] = {}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.document_count: int = 0
        self.processor: DocumentProcessor = DocumentProcessor()
        logger.info("SearchEngine initialized successfully")
    
    def build_index(self, documents: Dict[int, Any]) -> None:
        """
        Build complete TF-IDF index from a collection of documents.
        
        This method orchestrates the indexing pipeline:
        1. Preprocesses all documents using DocumentProcessor
        2. Builds inverted index with term frequencies
        3. Calculates IDF scores for all terms
        4. Computes TF-IDF vectors for each document
        
        Args:
            documents (Dict[int, Any]): 
                Mapping of {doc_id: (title, raw_text)} or
                {doc_id: {'title': title, 'raw_text': raw_text, ...}}
                
        Raises:
            ValueError: If documents dictionary is empty
            RuntimeError: If indexing pipeline fails
        """
        try:
            if not documents:
                raise ValueError("Documents dictionary cannot be empty")
            
            logger.info(f"Starting index build for {len(documents)} documents")
            self.document_count = len(documents)
            self.document_metadata = {}
            self.term_idf = {}
            self.document_vectors = {}
            
            # Phase 1: Preprocess documents and collect statistics
            doc_tokens_map: Dict[int, List[str]] = {}
            inverted_index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            
            for doc_id, value in documents.items():
                try:
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        title, raw_text = value
                        extra_metadata = {}
                    elif isinstance(value, dict):
                        title = value.get('title')
                        raw_text = value.get('raw_text')
                        extra_metadata = {k: v for k, v in value.items() if k not in ('title', 'raw_text')}
                    else:
                        raise ValueError(f"Unsupported document format for doc_id {doc_id}")
                    
                    if title is None or raw_text is None:
                        raise ValueError(f"Document {doc_id} must include title and raw_text")
                    
                    # Preprocess document text
                    tokens = self.processor.preprocess_text(raw_text)
                    doc_tokens_map[doc_id] = tokens
                    
                    # Store document metadata
                    self.document_metadata[doc_id] = {
                        'title': title,
                        'raw_text': raw_text,
                        'token_count': len(tokens),
                        **extra_metadata
                    }
                    
                    # Build inverted index: term -> {doc_id: frequency}
                    for token in tokens:
                        inverted_index[token][doc_id] += 1
                    
                    logger.debug(f"Processed document {doc_id}: '{title}' ({len(tokens)} tokens)")
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    raise
            
            logger.info(f"Phase 1 complete: {len(inverted_index)} unique terms identified")
            
            # Phase 2: Calculate IDF (Inverse Document Frequency)
            for term, doc_dict in inverted_index.items():
                # IDF = log(N / df) where N = total documents, df = document frequency
                document_frequency = len(doc_dict)
                idf = math.log(self.document_count / document_frequency) if document_frequency > 0 else 0
                self.term_idf[term] = idf
            
            logger.info(f"Phase 2 complete: IDF calculated for {len(self.term_idf)} terms")
            
            # Phase 3: Calculate TF-IDF vectors for each document
            for doc_id, tokens in doc_tokens_map.items():
                doc_vector: Dict[str, float] = {}
                token_count = self.document_metadata[doc_id]['token_count']
                
                if token_count > 0:
                    # Count term frequencies in document
                    term_frequencies: Dict[str, int] = {}
                    for token in tokens:
                        term_frequencies[token] = term_frequencies.get(token, 0) + 1
                    
                    # Calculate TF-IDF for each unique term
                    for term, freq in term_frequencies.items():
                        # TF = term_count / total_tokens_in_doc
                        tf = freq / token_count
                        # IDF = pre-calculated
                        idf = self.term_idf.get(term, 0)
                        # TF-IDF = TF * IDF
                        tfidf = tf * idf
                        
                        doc_vector[term] = tfidf
                
                self.document_vectors[doc_id] = doc_vector

            self.inverted_index = {term: dict(postings) for term, postings in inverted_index.items()}
            logger.info(f"Phase 3 complete: TF-IDF vectors built for all {self.document_count} documents")
            logger.info(f"Index build complete. Index size: {len(self.term_idf)} terms, "
                       f"{len(self.document_vectors)} documents")
            
        except ValueError as e:
            logger.error(f"ValueError during index build: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during index build: {str(e)}")
            raise RuntimeError(f"Index build failed: {str(e)}")
    
    def search(self, query: str, top_k: int = 10, search_type: str = 'vector') -> List[Dict]:
        """
        Execute a search query and return ranked results.
        
        Processing pipeline:
        1. Preprocess query using DocumentProcessor
        2. Calculate query TF-IDF vector
        3. Compute cosine similarity with selected documents
        4. Rank results by relevance score
        5. Extract text snippets for preview
        
        Args:
            query (str): User search query
            top_k (int): Number of top results to return (default: 10)
            search_type (str): Search mode, either 'vector' or 'boolean'
            
        Returns:
            List[Dict]: List of ranked search results, each containing:
                - rank (int): Result rank position
                - doc_id (int): Document identifier
                - title (str): Document title
                - score (float): Cosine similarity score [0, 1]
                - snippet (str): Text preview with query context
                
        Raises:
            ValueError: If query is empty or index not built
            RuntimeError: If search pipeline fails
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            if not self.document_vectors:
                raise ValueError("Search index not built. Call build_index() first.")
            
            if search_type not in {'vector', 'boolean'}:
                raise ValueError("search_type must be 'vector' or 'boolean'")

            logger.info(f"Search initiated: '{query}' [type={search_type}]")

            if search_type == 'boolean':
                matched_doc_ids = self._boolean_retrieve(query)
                logger.debug(f"Boolean query matched documents: {matched_doc_ids}")
                if not matched_doc_ids:
                    logger.info(f"No documents matched boolean query: '{query}'")
                    return []

                query_tokens = self._extract_query_terms(query)
                if not query_tokens:
                    logger.warning(f"Boolean query contained no searchable terms after preprocessing: '{query}'")
                    return []

                query_vector = self._calculate_query_vector(query_tokens)
                logger.debug(f"Boolean query tokens: {query_tokens}")

                similarity_scores: Dict[int, float] = {}
                for doc_id in matched_doc_ids:
                    doc_vector = self.document_vectors.get(doc_id, {})
                    similarity = self._cosine_similarity(query_vector, doc_vector)
                    if similarity > 0:
                        similarity_scores[doc_id] = similarity

                if not similarity_scores:
                    logger.info(f"Boolean query matched documents but no positive TF-IDF score found: '{query}'")
                    return []
            else:
                query_tokens = self.processor.preprocess_text(query)
                
                if not query_tokens:
                    logger.warning(f"Query preprocessing returned no tokens: '{query}'")
                    return []
                
                logger.debug(f"Query tokens: {query_tokens}")
                
                query_vector = self._calculate_query_vector(query_tokens)
                logger.debug(f"Query vector size: {len(query_vector)} terms")
                
                similarity_scores: Dict[int, float] = {}
                for doc_id, doc_vector in self.document_vectors.items():
                    similarity = self._cosine_similarity(query_vector, doc_vector)
                    if similarity > 0:
                        similarity_scores[doc_id] = similarity

                if not similarity_scores:
                    logger.info("No matching documents found")
                    return []
            
            logger.info(f"Computed similarities for {len(similarity_scores)} relevant documents")
            
            results = self._rank_results(similarity_scores, query_tokens, top_k)
            logger.info(f"Search complete: {len(results)} results returned (requested top {top_k})")
            
            return results
            
        except ValueError as e:
            logger.error(f"ValueError during search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")

    def search_vector(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using the Vector Space Model."""
        return self.search(query, top_k=top_k, search_type='vector')

    def search_boolean(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search using Boolean retrieval on the inverted index."""
        return self.search(query, top_k=top_k, search_type='boolean')
    
    def _calculate_query_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Calculate TF-IDF vector representation of the query.
        
        Applies same TF-IDF calculation as documents:
        - TF = term_frequency / query_length
        - IDF = pre-calculated corpus IDF
        - TF-IDF = TF * IDF
        
        Args:
            query_tokens (List[str]): Preprocessed query tokens
            
        Returns:
            Dict[str, float]: Query vector {term: tfidf_weight}
        """
        try:
            query_vector: Dict[str, float] = {}
            
            # Count term frequencies in query
            term_frequencies: Dict[str, int] = {}
            for token in query_tokens:
                term_frequencies[token] = term_frequencies.get(token, 0) + 1
            
            # Calculate TF-IDF for each term
            query_length = len(query_tokens)
            for term, freq in term_frequencies.items():
                tf = freq / query_length if query_length > 0 else 0
                idf = self.term_idf.get(term, 0)
                query_vector[term] = tf * idf
            
            return query_vector
            
        except Exception as e:
            logger.error(f"Error calculating query vector: {str(e)}")
            raise RuntimeError(f"Query vector calculation failed: {str(e)}")

    def _contains_boolean_operator(self, query: str) -> bool:
        """Check whether the raw query contains Boolean operators."""
        return bool(re.search(r"\b(?:AND|OR)\b", query, flags=re.IGNORECASE))

    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract searchable terms from a query, removing Boolean operators."""
        # Preprocess the raw query text; stop words and operators are removed automatically.
        return self.processor.preprocess_text(query)

    def _boolean_retrieve(self, query: str) -> Set[int]:
        """Parse a Boolean query and compute exact document matches from posting lists."""
        operands, operators = self._parse_boolean_query(query)
        if not operands:
            return set()

        posting_sets = [self._posting_set_for_operand(op) for op in operands]
        if any(not postings for postings in posting_sets):
            return set()

        return self._evaluate_boolean_expression(posting_sets, operators)

    def _parse_boolean_query(self, query: str) -> Tuple[List[str], List[str]]:
        """Split query into operand text fragments and Boolean operators."""
        raw_tokens = re.findall(r"\b(?:AND|OR)\b|[^\s]+", query, flags=re.IGNORECASE)
        operands: List[str] = []
        operators: List[str] = []
        current_operand: List[str] = []

        for token in raw_tokens:
            upper_token = token.upper()
            if upper_token in {"AND", "OR"}:
                if not current_operand:
                    raise ValueError(f"Malformed Boolean query: '{query}'")
                operands.append(" ".join(current_operand))
                operators.append(upper_token)
                current_operand = []
            else:
                current_operand.append(token)

        if current_operand:
            operands.append(" ".join(current_operand))

        return operands, operators

    def _posting_set_for_operand(self, operand: str) -> Set[int]:
        """Return the set of documents matching a Boolean operand."""
        tokens = self.processor.preprocess_text(operand)
        if not tokens:
            return set()

        posting_set: Set[int] = set(self.inverted_index.get(tokens[0], {}).keys())
        for token in tokens[1:]:
            posting_set &= set(self.inverted_index.get(token, {}).keys())
            if not posting_set:
                break

        return posting_set

    def _evaluate_boolean_expression(self, posting_sets: List[Set[int]], operators: List[str]) -> Set[int]:
        """Evaluate a Boolean expression with precedence: AND before OR."""
        if not posting_sets:
            return set()

        precedence = {"AND": 2, "OR": 1}
        values: List[Set[int]] = [posting_sets[0]]
        op_stack: List[str] = []

        for operator, posting_set in zip(operators, posting_sets[1:]):
            while op_stack and precedence[op_stack[-1]] >= precedence[operator]:
                right = values.pop()
                left = values.pop()
                op = op_stack.pop()
                values.append(self._apply_boolean_operator(left, right, op))
            op_stack.append(operator)
            values.append(posting_set)

        while op_stack:
            right = values.pop()
            left = values.pop()
            op = op_stack.pop()
            values.append(self._apply_boolean_operator(left, right, op))

        return values[0] if values else set()

    def _apply_boolean_operator(self, left: Set[int], right: Set[int], operator: str) -> Set[int]:
        """Apply a Boolean operator to two posting sets."""
        if operator == "AND":
            return left & right
        if operator == "OR":
            return left | right
        raise ValueError(f"Unsupported Boolean operator: {operator}")
    
    def _cosine_similarity(self, vector1: Dict[str, float], vector2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors.
        
        Formula: cos(θ) = (A · B) / (||A|| * ||B||)
        Where:
        - A · B = dot product of vectors
        - ||A||, ||B|| = magnitudes (L2 norm) of vectors
        
        Result is normalized to [0, 1] range where:
        - 0 = completely dissimilar (orthogonal)
        - 1 = identical vectors
        
        Args:
            vector1 (Dict[str, float]): First TF-IDF vector
            vector2 (Dict[str, float]): Second TF-IDF vector
            
        Returns:
            float: Cosine similarity score in range [0, 1]
        """
        try:
            # Calculate dot product
            dot_product = 0.0
            for term in vector1:
                if term in vector2:
                    dot_product += vector1[term] * vector2[term]
            
            # Calculate magnitude of vector1 (L2 norm)
            magnitude1 = math.sqrt(sum(weight ** 2 for weight in vector1.values()))
            
            # Calculate magnitude of vector2 (L2 norm)
            magnitude2 = math.sqrt(sum(weight ** 2 for weight in vector2.values()))
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)
            
            # Clamp to [0, 1] to handle floating point errors
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def _rank_results(
        self,
        similarity_scores: Dict[int, float],
        query_tokens: List[str],
        top_k: int
    ) -> List[Dict]:
        """
        Rank search results and format for presentation.
        
        Operations:
        1. Sort documents by similarity score (descending)
        2. Select top-k results
        3. Extract relevant text snippets
        4. Format result metadata
        
        Args:
            similarity_scores (Dict[int, float]): {doc_id: similarity_score}
            query_tokens (List[str]): Query tokens for snippet extraction
            top_k (int): Number of results to return
            
        Returns:
            List[Dict]: Ranked results with the following structure:
                {
                    'rank': int,           # Result rank (1, 2, 3, ...)
                    'doc_id': int,         # Document ID
                    'title': str,          # Document title
                    'score': float,        # Similarity score (rounded to 4 decimals)
                    'snippet': str         # Text preview with context
                }
        """
        try:
            # Sort by score descending
            ranked_scores = sorted(
                similarity_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            results: List[Dict] = []
            
            for rank, (doc_id, score) in enumerate(ranked_scores, 1):
                metadata = self.document_metadata[doc_id]
                snippet = self._extract_snippet(metadata['raw_text'], query_tokens)
                
                results.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'title': metadata['title'],
                    'score': round(score, 4),
                    'snippet': snippet
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error ranking results: {str(e)}")
            return []
    
    def _extract_snippet(
        self,
        text: str,
        query_tokens: List[str],
        window_size: int = 150
    ) -> str:
        """
        Extract a contextual text snippet around query term matches.
        
        Strategy:
        1. Search for first occurrence of any query term in document
        2. Extract window of characters before and after match
        3. Add ellipsis to indicate truncation
        4. Fallback to document beginning if no match found
        
        Args:
            text (str): Full document text
            query_tokens (List[str]): Query terms to locate
            window_size (int): Characters to include around match (default: 150)
            
        Returns:
            str: Text snippet with ellipsis indicators
        """
        try:
            if not text or not query_tokens:
                return text[:300] if text else ""
            
            # Find first occurrence of any query term
            text_lower = text.lower()
            earliest_position = len(text)
            found_term = None
            
            for token in query_tokens:
                position = text_lower.find(token)
                if position != -1 and position < earliest_position:
                    earliest_position = position
                    found_term = token
            
            # No match found - return document beginning
            if found_term is None:
                if len(text) > window_size:
                    return text[:window_size] + "..."
                return text
            
            # Extract window around matched term
            term_length = len(found_term)
            start_pos = max(0, earliest_position - (window_size // 2))
            end_pos = min(len(text), earliest_position + term_length + (window_size // 2))
            
            snippet = text[start_pos:end_pos]
            
            # Add ellipsis indicators
            if start_pos > 0:
                snippet = "..." + snippet
            if end_pos < len(text):
                snippet = snippet + "..."
            
            return snippet
            
        except Exception as e:
            logger.error(f"Error extracting snippet: {str(e)}")
            return text[:200] if text else ""
    
    def save_index(self, filepath: str) -> None:
        """
        Persist the search index to a JSON file.
        
        Serializes all index components for recovery:
        - Document metadata (titles, raw text)
        - IDF weights for all terms
        - TF-IDF vectors for all documents
        - Document count for IDF recalculation
        
        Args:
            filepath (str): Destination file path for index storage
            
        Raises:
            ValueError: If index is empty or filepath is invalid
            IOError: If file writing fails
        """
        try:
            if not self.document_vectors:
                raise ValueError("Index is empty. Build index first using build_index()")
            
            # Prepare data for JSON serialization
            index_data = {
                'document_metadata': self.document_metadata,
                'term_idf': self.term_idf,
                'document_vectors': self.document_vectors,
                'document_count': self.document_count,
                'inverted_index': self.inverted_index
            }
            
            # Create parent directories if needed
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Write index to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            file_size = Path(filepath).stat().st_size
            logger.info(f"Index successfully saved to {filepath} ({file_size} bytes)")
            
        except ValueError as e:
            logger.error(f"ValueError during index save: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"IOError during index save: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving index: {str(e)}")
            raise RuntimeError(f"Failed to save index: {str(e)}")
    
    def load_index(self, filepath: str) -> None:
        """
        Restore a previously saved search index from JSON file.
        
        Loads all index components:
        - Document metadata and text
        - IDF weights
        - TF-IDF vectors
        - Document count
        
        After loading, the search engine is ready for queries without rebuilding.
        
        Args:
            filepath (str): Path to saved index JSON file
            
        Raises:
            FileNotFoundError: If index file does not exist
            IOError: If file reading fails
            ValueError: If JSON format is invalid
        """
        try:
            filepath_obj = Path(filepath)
            
            if not filepath_obj.exists():
                raise FileNotFoundError(f"Index file not found: {filepath}")
            
            # Read index from JSON file
            with open(filepath, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Restore all index components
            # Convert string keys back to integers (JSON converts integer keys to strings)
            self.document_metadata = {int(k): v for k, v in index_data['document_metadata'].items()}
            self.term_idf = index_data['term_idf']
            
            # Convert document_vectors keys from strings to integers
            self.document_vectors = {int(k): v for k, v in index_data['document_vectors'].items()}
            self.document_count = index_data['document_count']
            self.inverted_index = {
                term: {int(doc_id): frequency for doc_id, frequency in postings.items()}
                for term, postings in index_data.get('inverted_index', {}).items()
            }

            if not self.inverted_index:
                logger.info("Reconstructing inverted index from loaded document vectors")
                self.inverted_index = {}
                for doc_id, vector in self.document_vectors.items():
                    for term in vector.keys():
                        self.inverted_index.setdefault(term, {})[doc_id] = 1

            # Repair inconsistent index state if metadata and vector storage differ
            metadata_ids = set(self.document_metadata.keys())
            vector_ids = set(self.document_vectors.keys())
            if metadata_ids != vector_ids:
                logger.warning(
                    f"Index data inconsistency detected: {len(metadata_ids)} metadata docs, "
                    f"{len(vector_ids)} vector docs. Rebuilding index from metadata."
                )
                if metadata_ids:
                    documents_to_rebuild = {
                        doc_id: (meta['title'], meta['raw_text'])
                        for doc_id, meta in self.document_metadata.items()
                    }
                    self.build_index(documents_to_rebuild)
                    try:
                        self.save_index(filepath)
                        logger.info(f"Repaired and persisted consistent index to {filepath}")
                    except Exception as e:
                        logger.warning(f"Failed to persist repaired index: {str(e)}")
                else:
                    self.document_vectors = {doc_id: vec for doc_id, vec in self.document_vectors.items() if doc_id in metadata_ids}
                    self.document_count = len(self.document_metadata)
            
            file_size = filepath_obj.stat().st_size
            logger.info(f"Index successfully loaded from {filepath} ({file_size} bytes)")
            logger.info(f"Loaded {len(self.document_vectors)} documents with "
                       f"{len(self.term_idf)} unique terms")
            
        except FileNotFoundError as e:
            logger.error(f"Index file not found: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in index file: {str(e)}")
            raise ValueError(f"Index file is corrupted: {str(e)}")
        except IOError as e:
            logger.error(f"IOError reading index file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading index: {str(e)}")
            raise RuntimeError(f"Failed to load index: {str(e)}")
    
    def get_index_stats(self) -> Dict:
        """
        Get comprehensive statistics about the current index.
        
        Returns:
            Dict: Statistics including document count, unique terms, average
                  vector size, and other metrics for monitoring
        """
        try:
            total_terms = len(self.term_idf)
            avg_vector_size = (
                sum(len(v) for v in self.document_vectors.values()) / len(self.document_vectors)
                if self.document_vectors else 0
            )
            
            stats = {
                'document_count': self.document_count,
                'unique_terms': total_terms,
                'average_vector_size': round(avg_vector_size, 2),
                'indexed_documents': len(self.document_vectors),
                'indexed': len(self.document_vectors) == self.document_count
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating index statistics: {str(e)}")
            return {}
    
    def get_document_details(self, doc_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific document including term weights.
        
        Returns IR metrics for the document:
        - Title and raw text
        - Token count and document length
        - Top weighted terms with TF-IDF scores
        - IDF values for all terms in document
        
        Args:
            doc_id (int): Document identifier
            
        Returns:
            Dict with document details or None if not found
        """
        try:
            if doc_id not in self.document_metadata:
                logger.warning(f"Document {doc_id} not found in index")
                return None
            
            metadata = self.document_metadata[doc_id]
            doc_vector = self.document_vectors.get(doc_id, {})
            
            # Get top 20 terms by TF-IDF weight
            top_terms = sorted(
                [(term, round(weight, 4)) for term, weight in doc_vector.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            # Get IDF values for document terms
            term_idf_weights = {}
            for term in doc_vector.keys():
                term_idf_weights[term] = round(self.term_idf.get(term, 0), 4)
            
            details = {
                'doc_id': doc_id,
                'title': metadata['title'],
                'text_length': len(metadata['raw_text']),
                'token_count': metadata['token_count'],
                'unique_terms': len(doc_vector),
                'top_terms': top_terms,
                'term_idf_map': term_idf_weights,
                'raw_text': metadata['raw_text']
            }
            
            logger.info(f"Retrieved details for document {doc_id}")
            return details
            
        except Exception as e:
            logger.error(f"Error retrieving document details for {doc_id}: {str(e)}")
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get list of all indexed documents with basic metadata.
        
        Returns:
            List of dictionaries with document information:
            - doc_id: Document identifier
            - title: Document title
            - text_length: Length of raw text
            - token_count: Number of preprocessed tokens
            - unique_terms: Count of unique terms in document
        """
        try:
            documents = []
            for doc_id, metadata in self.document_metadata.items():
                doc_vector = self.document_vectors.get(doc_id, {})
                documents.append({
                    'doc_id': doc_id,
                    'title': metadata['title'],
                    'text_length': len(metadata['raw_text']),
                    'token_count': metadata['token_count'],
                    'unique_terms': len(doc_vector)
                })
            
            # Sort by doc_id for consistent ordering
            documents.sort(key=lambda x: x['doc_id'], reverse=True)
            logger.info(f"Retrieved list of {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving document list: {str(e)}")
            return []
    
    def get_term_statistics(self, term: str) -> Optional[Dict]:
        """
        Get statistics about a specific term across the corpus.
        
        Returns IR metrics for the term:
        - IDF weight
        - Document frequency (how many documents contain this term)
        - Posting list (documents containing term with frequency)
        
        Args:
            term (str): Term to analyze (should be preprocessed)
            
        Returns:
            Dict with term statistics or None if not found
        """
        try:
            if term not in self.term_idf:
                logger.info(f"Term '{term}' not found in index")
                return None
            
            # Find documents containing this term
            documents_with_term = []
            for doc_id, doc_vector in self.document_vectors.items():
                if term in doc_vector:
                    documents_with_term.append({
                        'doc_id': doc_id,
                        'title': self.document_metadata[doc_id]['title'],
                        'tfidf_weight': round(doc_vector[term], 4)
                    })
            
            stats = {
                'term': term,
                'idf': round(self.term_idf[term], 4),
                'document_frequency': len(documents_with_term),
                'documents': documents_with_term
            }
            
            logger.info(f"Retrieved statistics for term '{term}'")
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving term statistics for '{term}': {str(e)}")
            return None
    
    def get_top_terms_by_idf(self, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        Get the top-k terms by IDF weight (rarest/most discriminative terms).
        
        IDF measures term rarity: higher IDF means term appears in fewer documents
        and is more useful for discrimination between documents.
        
        Args:
            top_k (int): Number of top terms to return
            
        Returns:
            List of (term, idf_weight) tuples sorted by IDF descending
        """
        try:
            sorted_terms = sorted(
                [(term, round(weight, 4)) for term, weight in self.term_idf.items()],
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            logger.info(f"Retrieved top {top_k} terms by IDF")
            return sorted_terms
            
        except Exception as e:
            logger.error(f"Error retrieving top terms: {str(e)}")
            return []
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete a document from the index and rebuild.
        
        This operation:
        1. Removes document metadata
        2. Removes TF-IDF vectors
        3. Rebuilds the entire index
        
        Args:
            doc_id (int): Document identifier to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if doc_id not in self.document_metadata:
                logger.warning(f"Document {doc_id} not found in index")
                return False
            
            logger.info(f"Deleting document {doc_id}")
            
            # Remove document metadata
            del self.document_metadata[doc_id]
            
            # If we have remaining documents, rebuild the index
            if self.document_metadata:
                # Rebuild index with remaining documents
                documents_to_rebuild = {
                    doc_id: (meta['title'], meta['raw_text'])
                    for doc_id, meta in self.document_metadata.items()
                }
                self.build_index(documents_to_rebuild)
                logger.info(f"Index rebuilt after deletion. Remaining documents: {len(self.document_metadata)}")
            else:
                # No documents left, clear index
                self.term_idf = {}
                self.document_vectors = {}
                self.document_count = 0
                logger.info("All documents deleted. Index cleared.")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def update_document(self, doc_id: int, title: str, raw_text: str) -> bool:
        """
        Update a document's content and rebuild the index.
        
        This operation:
        1. Updates document metadata (title and text)
        2. Rebuilds the index to recalculate TF-IDF
        
        Args:
            doc_id (int): Document identifier
            title (str): New document title
            raw_text (str): New document text
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if doc_id not in self.document_metadata:
                logger.warning(f"Document {doc_id} not found in index")
                return False
            
            logger.info(f"Updating document {doc_id}: '{title}'")
            
            # Update metadata
            existing_metadata = self.document_metadata[doc_id]
            extra_metadata = {k: v for k, v in existing_metadata.items() if k not in ('title', 'raw_text', 'token_count')}
            self.document_metadata[doc_id] = {
                'title': title,
                'raw_text': raw_text,
                **extra_metadata
            }
            
            # Rebuild index with updated documents
            documents_to_rebuild = {
                doc_id: meta
                for doc_id, meta in self.document_metadata.items()
            }
            self.build_index(documents_to_rebuild)
            
            logger.info(f"Document {doc_id} updated and index rebuilt")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {str(e)}")
            return False
