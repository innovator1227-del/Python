"""
Search Engine module for Document Information Retrieval System.

This module implements a complete TF-IDF search engine from scratch using
an inverted index and the Vector Space Model with cosine similarity.

Author: IRS Expert
Version: 1.0.0
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

from core.document_processor import DocumentProcessor


# Configure logging for search engine operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/search_engine.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SearchEngine:
    """
    SearchEngine builds an inverted index, calculates TF-IDF weights from scratch,
    and ranks documents against a query using cosine similarity.
    """

    def __init__(self) -> None:
        """Initialize the search engine and core data structures."""
        self.processor = DocumentProcessor()
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.document_frequencies: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.document_vectors: Dict[int, Dict[str, float]] = {}
        self.document_norms: Dict[int, float] = {}
        self.document_count: int = 0
        logger.info("SearchEngine initialized")

    def build_index(self, documents: Dict[int, Tuple[str, str]]) -> None:
        """
        Build the inverted index and TF-IDF vectors from a collection of documents.

        Args:
            documents: Mapping from document ID to (title, raw_text).

        Raises:
            ValueError: If the document collection is empty.
            RuntimeError: If index building fails.
        """
        if not documents:
            logger.error("build_index() called with empty document collection")
            raise ValueError("Document collection must not be empty")

        try:
            self.inverted_index = {}
            self.document_frequencies = {}
            self.idf = {}
            self.documents = {}
            self.document_vectors = {}
            self.document_norms = {}
            self.document_count = len(documents)

            for doc_id, (title, raw_text) in documents.items():
                if not isinstance(doc_id, int):
                    raise ValueError("Document ID must be an integer")
                if not title or not isinstance(title, str):
                    raise ValueError(f"Invalid title for document {doc_id}")

                tokens = self.processor.preprocess_text(raw_text)
                if not tokens:
                    logger.warning(f"Document {doc_id} contains no preprocessed tokens")

                term_counts: Dict[str, int] = {}
                for token in tokens:
                    term_counts[token] = term_counts.get(token, 0) + 1

                self.documents[doc_id] = {
                    'title': title,
                    'raw_text': raw_text,
                    'token_count': len(tokens)
                }

                for term, frequency in term_counts.items():
                    if term not in self.inverted_index:
                        self.inverted_index[term] = {}
                    self.inverted_index[term][doc_id] = frequency

            self.document_frequencies = {
                term: len(postings)
                for term, postings in self.inverted_index.items()
            }

            total_docs = float(self.document_count)
            for term, df in self.document_frequencies.items():
                if df <= 0:
                    self.idf[term] = 0.0
                else:
                    self.idf[term] = math.log(total_docs / df)

            for doc_id, metadata in self.documents.items():
                tokens = self.processor.preprocess_text(metadata['raw_text'])
                term_counts: Dict[str, int] = {}
                for token in tokens:
                    term_counts[token] = term_counts.get(token, 0) + 1

                vector: Dict[str, float] = {}
                token_total = float(metadata['token_count']) if metadata['token_count'] > 0 else 1.0
                for term, frequency in term_counts.items():
                    tf = frequency / token_total
                    tfidf = tf * self.idf.get(term, 0.0)
                    if tfidf > 0.0:
                        vector[term] = tfidf

                self.document_vectors[doc_id] = vector
                self.document_norms[doc_id] = math.sqrt(
                    sum(weight * weight for weight in vector.values())
                )

            logger.info(
                f"Built index for {self.document_count} documents with "
                f"{len(self.document_vectors)} document vectors"
            )

        except ValueError:
            raise
        except Exception as exc:
            logger.error(f"Failed to build index: {exc}")
            raise RuntimeError(f"Index build error: {exc}")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index and return ranked documents for the query.

        Args:
            query: User query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dictionaries containing title, score, and snippet.

        Raises:
            ValueError: If the query is invalid or the index is empty.
            RuntimeError: If search execution fails.
        """
        if not query or not isinstance(query, str):
            logger.error("search() called with invalid query")
            raise ValueError("Query must be a non-empty string")
        if self.document_count == 0 or not self.document_vectors:
            logger.error("search() called before index was built")
            raise ValueError("Search index is empty. Build the index before searching.")

        try:
            query_tokens = self.processor.preprocess_text(query)
            if not query_tokens:
                logger.warning("Query preprocessing returned no tokens")
                return []

            query_vector = self._build_query_vector(query_tokens)
            query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
            if query_norm == 0.0:
                logger.warning("Query vector norm is zero")
                return []

            scores: List[Tuple[int, float]] = []
            for doc_id, doc_vector in self.document_vectors.items():
                score = self._cosine_similarity(query_vector, query_norm, doc_vector, self.document_norms.get(doc_id, 0.0))
                if score > 0.0:
                    scores.append((doc_id, score))

            scores.sort(key=lambda item: item[1], reverse=True)
            top_results = scores[:top_k]

            results: List[Dict[str, Any]] = []
            for rank, (doc_id, score) in enumerate(top_results, start=1):
                metadata = self.documents[doc_id]
                snippet = self._build_snippet(metadata['raw_text'], query_tokens)
                results.append({
                    'rank': rank,
                    'doc_id': doc_id,
                    'title': metadata['title'],
                    'score': round(score, 4),
                    'snippet': snippet
                })

            logger.info(f"Search returned {len(results)} results for query '{query}'")
            return results

        except ValueError:
            raise
        except Exception as exc:
            logger.error(f"Search failed for query '{query}': {exc}")
            raise RuntimeError(f"Search error: {exc}")

    def _build_query_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Construct a TF-IDF vector for the search query."""
        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        total_terms = float(len(tokens)) if tokens else 1.0
        query_vector: Dict[str, float] = {}
        for term, frequency in term_counts.items():
            tf = frequency / total_terms
            idf_value = self.idf.get(term, 0.0)
            query_vector[term] = tf * idf_value
        return query_vector

    def _cosine_similarity(
        self,
        vector_a: Dict[str, float],
        norm_a: float,
        vector_b: Dict[str, float],
        norm_b: float
    ) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        dot_product = 0.0
        for term, weight_a in vector_a.items():
            weight_b = vector_b.get(term)
            if weight_b is not None:
                dot_product += weight_a * weight_b

        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, min(similarity, 1.0))

    def _build_snippet(self, raw_text: str, terms: List[str], window: int = 150) -> str:
        """Generate a short snippet around the first matching query term."""
        if not raw_text:
            return ''

        text_lower = raw_text.lower()
        best_position = len(text_lower)
        best_term = ''
        for term in terms:
            position = text_lower.find(term)
            if position != -1 and position < best_position:
                best_position = position
                best_term = term

        if best_term == '':
            snippet = raw_text[:window].strip()
            return f"{snippet}..." if len(raw_text) > window else snippet

        start = max(0, best_position - window // 2)
        end = min(len(raw_text), best_position + window // 2)
        snippet = raw_text[start:end].strip()
        if start > 0:
            snippet = '...' + snippet
        if end < len(raw_text):
            snippet = snippet + '...'
        return snippet

    def save_index(self, file_path: str) -> None:
        """Persist the search index and metadata to a JSON file."""
        if self.document_count == 0 or not self.document_vectors:
            logger.error("save_index() called with empty index")
            raise ValueError("Search index is empty. Build the index before saving.")

        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            serializable_data = {
                'documents': {
                    str(doc_id): {
                        'title': metadata['title'],
                        'raw_text': metadata['raw_text'],
                        'token_count': metadata['token_count']
                    }
                    for doc_id, metadata in self.documents.items()
                },
                'inverted_index': {
                    term: {str(doc_id): frequency for doc_id, frequency in postings.items()}
                    for term, postings in self.inverted_index.items()
                },
                'idf': self.idf,
                'document_vectors': {
                    str(doc_id): vector for doc_id, vector in self.document_vectors.items()
                },
                'document_norms': self.document_norms,
                'document_count': self.document_count
            }

            with path.open('w', encoding='utf-8') as file_handle:
                json.dump(serializable_data, file_handle, indent=2, ensure_ascii=False)

            logger.info(f"Saved search index to '{file_path}'")

        except Exception as exc:
            logger.error(f"Failed to save index to '{file_path}': {exc}")
            raise RuntimeError(f"Save index failed: {exc}")

    def load_index(self, file_path: str) -> None:
        """Load a previously saved search index from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Index file not found: '{file_path}'")
            raise FileNotFoundError(f"Index file not found: {file_path}")

        try:
            with path.open('r', encoding='utf-8') as file_handle:
                data = json.load(file_handle)

            self.documents = {
                int(doc_id): {
                    'title': metadata['title'],
                    'raw_text': metadata['raw_text'],
                    'token_count': metadata['token_count']
                }
                for doc_id, metadata in data.get('documents', {}).items()
            }
            self.inverted_index = {
                term: {int(doc_id): frequency for doc_id, frequency in postings.items()}
                for term, postings in data.get('inverted_index', {}).items()
            }
            self.idf = {term: float(value) for term, value in data.get('idf', {}).items()}
            self.document_vectors = {
                int(doc_id): {term: float(weight) for term, weight in vector.items()}
                for doc_id, vector in data.get('document_vectors', {}).items()
            }
            self.document_norms = {
                int(doc_id): float(norm) for doc_id, norm in data.get('document_norms', {}).items()
            }
            self.document_count = int(data.get('document_count', len(self.documents)))

            logger.info(f"Loaded search index from '{file_path}'")

        except Exception as exc:
            logger.error(f"Failed to load index from '{file_path}': {exc}")
            raise RuntimeError(f"Load index failed: {exc}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Return index metadata for monitoring and diagnostics."""
        return {
            'document_count': self.document_count,
            'indexed_documents': len(self.document_vectors),
            'unique_terms': len(self.inverted_index),
            'average_vector_size': (
                sum(len(vector) for vector in self.document_vectors.values()) / len(self.document_vectors)
                if self.document_vectors else 0.0
            )
        }
