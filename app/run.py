"""
Flask Web Application for Document Information Retrieval System

Provides REST API endpoints for document upload, search, and administration.
Implements full CORS support, error handling, and logging.

Author: Full-Stack Web Developer
Version: 1.0.0
"""

import os
import sys
import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.preprocessor import DocumentProcessor
from core.ranker import SearchEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/flask_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Application Configuration
class Config:
    """Flask application configuration."""
    UPLOAD_FOLDER = 'data/raw_documents'
    INDEX_FILE = 'data/processed_indexes/index.json'
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max file size
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False


def create_app() -> Flask:
    """
    Application factory for Flask app creation.
    
    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)
    
    # Enable CORS for all routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Create necessary directories
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['INDEX_FILE']).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Flask application created and configured")
    return app


# Initialize Flask app
app = create_app()

# Initialize core components
try:
    processor = DocumentProcessor()
    search_engine = SearchEngine()
    
    # Load existing index if available
    if Path(app.config['INDEX_FILE']).exists():
        search_engine.load_index(app.config['INDEX_FILE'])
        logger.info("Existing index loaded successfully")
    else:
        logger.info("No existing index found. Fresh index will be created on first document upload.")
        
except Exception as e:
    logger.error(f"Failed to initialize core components: {str(e)}")
    processor = None
    search_engine = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has allowed extension.
    
    Args:
        filename (str): Filename to validate
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_document_id() -> int:
    """
    Generate unique document ID based on timestamp and random number.
    
    Returns:
        int: Unique document identifier
    """
    import time
    import random
    return int(time.time() * 1000) + random.randint(1, 999)


def resolve_document_path(file_path: str) -> Path:
    """Resolve a raw document path to an absolute filesystem path."""
    path_obj = Path(file_path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parents[1] / path_obj
    return path_obj


def write_document_file(original_path: Optional[str], text: str, doc_id: int, title: str) -> str:
    """Write updated document text back to a raw_documents file and return its stored path."""
    workspace_root = Path(__file__).resolve().parents[1]
    fallback_name = f"{secure_filename(title)}_{doc_id}.txt"
    if original_path:
        source_path = resolve_document_path(original_path)
    else:
        source_path = None

    if source_path and source_path.exists() and source_path.suffix.lower() == '.txt':
        source_path.write_text(text, encoding='utf-8')
        return str(source_path.relative_to(workspace_root))

    if source_path and source_path.exists():
        updated_path = source_path.with_suffix('.txt')
        if updated_path.exists() and updated_path != source_path:
            updated_path = source_path.with_name(f"{source_path.stem}_{doc_id}.txt")
        updated_path.parent.mkdir(parents=True, exist_ok=True)
        updated_path.write_text(text, encoding='utf-8')
        return str(updated_path.relative_to(workspace_root))

    fallback_path = Path(app.config['UPLOAD_FOLDER']) / fallback_name
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(text, encoding='utf-8')
    return str(fallback_path.relative_to(workspace_root))


def delete_document_file(file_path: Optional[str]) -> None:
    """Delete the raw document file from disk if it exists."""
    if not file_path:
        return
    try:
        path_obj = resolve_document_path(file_path)
        if path_obj.exists():
            path_obj.unlink()
            logger.info(f"Deleted raw document file: {path_obj}")
    except Exception as e:
        logger.warning(f"Unable to delete raw document file '{file_path}': {str(e)}")


# ============================================================================
# FLASK ROUTES - FRONTEND
# ============================================================================

@app.route('/')
def index() -> str:
    """
    Render main dashboard page.
    
    Returns:
        str: Rendered HTML template
    """
    try:
        logger.info("Dashboard page requested")
        # Provide initial documents to the template to avoid UI flash on refresh
        try:
            initial_documents = []
            if search_engine is not None:
                initial_documents = search_engine.get_all_documents()
        except Exception:
            initial_documents = []

        return render_template('index.html', initial_docs=json.dumps(initial_documents))
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        return f"Error loading page: {str(e)}", 500


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_document() -> Tuple[Dict[str, Any], int]:
    """
    Handle document upload and indexing.
    
    Process:
    1. Validate file (extension, size)
    2. Save file to disk
    3. Extract and preprocess text
    4. Add to search index
    5. Persist index to JSON
    
    Returns:
        Tuple[Dict, int]: JSON response and HTTP status code
        
    Response format:
    {
        'success': bool,
        'message': str,
        'doc_id': int (on success),
        'error': str (on failure)
    }
    """
    # Enhanced logging to help diagnose upload failures
    try:
        logger.info("Document upload initiated")

        logger.info(f"Request content length: {request.content_length}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request mimetype: {request.mimetype}")
        logger.info(f"Request headers Content-Type: {request.headers.get('Content-Type')}")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form keys: {list(request.form.keys())}")

        # Validate request
        file = request.files.get('file')
        if file is None:
            logger.warning("Upload request missing file field")
            return jsonify({
                'success': False,
                'error': 'No file provided in request',
                'debug': {
                    'files': list(request.files.keys()),
                    'form': list(request.form.keys()),
                    'content_type': request.content_type,
                    'mimetype': request.mimetype
                }
            }), 400

        logger.info(f"Uploaded filename: {getattr(file, 'filename', None)}, content_type: {getattr(file, 'content_type', None)}")

        if not file or file.filename == '':
            logger.warning("Upload request with empty filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.warning(f"Upload attempted with disallowed file type: {file.filename}")
            return jsonify({'success': False, 'error': f'File type not allowed. Supported: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'}), 400

        # Check if processors are initialized
        if processor is None or search_engine is None:
            logger.error("Core processors not initialized")
            return jsonify({'success': False, 'error': 'System not ready. Please try again later.'}), 500

        # Build destination path and save
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            unique_filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # Ensure upload folder exists
            Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

            file.save(filepath)
            logger.info(f"File saved: {filepath}")
        except Exception:
            logger.exception("Failed to save uploaded file")
            return jsonify({'success': False, 'error': 'Failed to save uploaded file on server'}), 500

        # Extract and preprocess document
        try:
            raw_text, tokens = processor.process_document(filepath)
        except Exception:
            logger.exception("Document processing failed")
            try:
                os.remove(filepath)
            except Exception:
                logger.warning(f"Failed to remove file after processing failure: {filepath}")
            return jsonify({'success': False, 'error': 'Failed to process document'}), 400

        if not tokens:
            logger.warning(f"No content extracted from file: {filepath}")
            try:
                os.remove(filepath)
            except Exception:
                logger.warning(f"Failed to remove empty-content file: {filepath}")
            return jsonify({'success': False, 'error': 'File contains no extractable text content'}), 400

        # Generate unique document ID
        doc_id = get_document_id()
        doc_title = os.path.splitext(filename)[0]

        # Merge with existing documents metadata (if any)
        try:
            # use attribute document_metadata if present (SearchEngine may have different storage)
            existing_docs = {}
            if hasattr(search_engine, 'document_metadata') and isinstance(search_engine.document_metadata, dict):
                existing_docs = {did: meta.copy() for did, meta in search_engine.document_metadata.items()}
            elif hasattr(search_engine, 'documents') and isinstance(search_engine.documents, dict):
                # support alternative storage used by different SearchEngine implementations
                existing_docs = {did: {'title': meta.get('title'), 'raw_text': meta.get('raw_text')} for did, meta in search_engine.documents.items()}

            file_path_metadata = os.path.relpath(filepath, Path(__file__).resolve().parents[1])
            existing_docs[doc_id] = {
                'title': doc_title,
                'raw_text': raw_text,
                'file_path': file_path_metadata,
                'uploaded_at': datetime.now().isoformat()
            }

            # Rebuild index
            search_engine.build_index(existing_docs)
            logger.info(f"Index rebuilt with document {doc_id}")
        except Exception:
            logger.exception("Failed to rebuild index after upload")
            try:
                os.remove(filepath)
            except Exception:
                logger.warning(f"Failed to remove file after index rebuild failure: {filepath}")
            return jsonify({'success': False, 'error': 'Failed to update search index'}), 500

        # Persist index
        try:
            search_engine.save_index(app.config['INDEX_FILE'])
            logger.info("Index persisted to disk")
        except Exception:
            logger.exception("Failed to save index to disk")
            return jsonify({'success': False, 'error': 'Failed to persist index'}), 500

        logger.info(f"Document upload successful: {doc_id} - {doc_title}")
        return jsonify({'success': True, 'message': f'Document "{doc_title}" uploaded and indexed successfully', 'doc_id': doc_id, 'doc_title': doc_title, 'token_count': len(tokens)}), 201

    except Exception:
        logger.exception("Unexpected error during upload")
        return jsonify({'success': False, 'error': 'Server error during upload'}), 500


@app.route('/api/search', methods=['POST'])
def search_documents() -> Tuple[Dict[str, Any], int]:
    """
    Execute search query against indexed documents.
    
    Process:
    1. Extract and validate query from request
    2. Execute search using SearchEngine
    3. Format results for frontend
    
    Request JSON:
    {
        'query': str,
        'top_k': int (optional, default: 10)
    }
    
    Returns:
        Tuple[Dict, int]: JSON response and HTTP status code
        
    Response format (success):
    {
        'success': True,
        'query': str,
        'results': [
            {
                'rank': int,
                'doc_id': int,
                'title': str,
                'score': float (0-1),
                'score_percent': int (0-100),
                'snippet': str
            }
        ],
        'total_results': int,
        'execution_time_ms': float
    }
    
    Response format (failure):
    {
        'success': False,
        'error': str
    }
    """
    try:
        import time
        start_time = time.time()
        
        logger.info("Search request initiated")
        
        # Parse request
        data = request.get_json()
        
        if not data or 'query' not in data:
            logger.warning("Search request missing query field")
            return jsonify({
                'success': False,
                'error': 'Query field is required'
            }), 400
        
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)
        
        if not query:
            logger.warning("Search request with empty query")
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            top_k = 10
        
        # Check if search engine is ready
        if search_engine is None or not search_engine.document_vectors:
            logger.warning("Search attempted but no documents indexed")
            return jsonify({
                'success': True,
                'query': query,
                'results': [],
                'total_results': 0,
                'message': 'No documents indexed yet. Please upload documents first.'
            }), 200
        
        # Execute search
        try:
            results = search_engine.search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Search execution failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Search failed: {str(e)}'
            }), 500
        
        # Format results for frontend
        formatted_results = [
            {
                'rank': result['rank'],
                'doc_id': result['doc_id'],
                'title': result['title'],
                'score': result['score'],
                'score_percent': int(result['score'] * 100),
                'snippet': result['snippet']
            }
            for result in results
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Search completed: '{query}' - {len(formatted_results)} results "
                   f"({execution_time:.2f}ms)")
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results),
            'execution_time_ms': round(execution_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats() -> Tuple[Dict[str, Any], int]:
    """
    Get system statistics and index information.
    
    Returns:
        Tuple[Dict, int]: Statistics and HTTP status code
    """
    try:
        if search_engine is None:
            stats = {
                'document_count': 0,
                'indexed_documents': 0,
                'unique_terms': 0,
                'status': 'Not initialized'
            }
        else:
            stats = search_engine.get_index_stats()
            stats['status'] = 'Ready' if search_engine.document_vectors else 'Empty'
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/documents', methods=['GET'])
def get_documents_list() -> Tuple[Dict[str, Any], int]:
    """
    Get list of all indexed documents with metadata.
    
    Returns:
        Tuple[Dict, int]: JSON response with document list
        
    Response format:
    {
        'success': True,
        'documents': [
            {
                'doc_id': int,
                'title': str,
                'text_length': int,
                'token_count': int,
                'unique_terms': int
            }
        ],
        'total_documents': int
    }
    """
    try:
        if search_engine is None or not search_engine.document_vectors:
            return jsonify({
                'success': True,
                'documents': [],
                'total_documents': 0
            }), 200
        
        documents = search_engine.get_all_documents()
        
        return jsonify({
            'success': True,
            'documents': documents,
            'total_documents': len(documents)
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving documents list: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/document/<int:doc_id>', methods=['GET'])
def get_document_details(doc_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Get detailed information about a specific document including term weights.
    
    Returns IR metrics:
    - Document title and text length
    - Token count and unique terms
    - Top 20 terms by TF-IDF weight
    - IDF weights for all terms
    
    Args:
        doc_id (int): Document identifier
        
    Returns:
        Tuple[Dict, int]: JSON response with document details
        
    Response format:
    {
        'success': True,
        'document': {
            'doc_id': int,
            'title': str,
            'text_length': int,
            'token_count': int,
            'unique_terms': int,
            'top_terms': [[term, tfidf_weight], ...],
            'term_idf_map': {term: idf_weight, ...}
        }
    }
    """
    try:
        if search_engine is None:
            return jsonify({
                'success': False,
                'error': 'Search engine not initialized'
            }), 400
        
        details = search_engine.get_document_details(doc_id)
        
        if details is None:
            return jsonify({
                'success': False,
                'error': f'Document {doc_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'document': details
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving document details for {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/document/<int:doc_id>/view', methods=['GET'])
def view_document(doc_id: int) -> Tuple[Dict[str, Any], int]:
    """
    View the full text content of a specific document.
    
    Args:
        doc_id (int): Document identifier
        
    Returns:
        Tuple[Dict, int]: JSON response with document text
        
    Response format:
    {
        'success': True,
        'doc_id': int,
        'title': str,
        'text': str,
        'length': int
    }
    """
    try:
        if search_engine is None:
            return jsonify({
                'success': False,
                'error': 'Search engine not initialized'
            }), 400
        
        if doc_id not in search_engine.document_metadata:
            return jsonify({
                'success': False,
                'error': f'Document {doc_id} not found'
            }), 404
        
        metadata = search_engine.document_metadata[doc_id]
        
        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'title': metadata['title'],
            'text': metadata['raw_text'],
            'length': len(metadata['raw_text'])
        }), 200
        
    except Exception as e:
        logger.error(f"Error viewing document {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/term-stats/<term>', methods=['GET'])
def get_term_stats(term: str) -> Tuple[Dict[str, Any], int]:
    """
    Get statistics about a specific term in the corpus.
    
    Returns IR metrics for the term:
    - IDF weight (discriminative value)
    - Document frequency
    - List of documents containing the term
    
    Args:
        term (str): Term to analyze (will be preprocessed)
        
    Returns:
        Tuple[Dict, int]: JSON response with term statistics
    """
    try:
        if search_engine is None or not search_engine.document_vectors:
            return jsonify({
                'success': False,
                'error': 'No documents indexed'
            }), 400
        
        # Preprocess the term
        try:
            preprocessed_terms = processor.preprocess_text(term)
            if not preprocessed_terms:
                return jsonify({
                    'success': False,
                    'error': 'Term preprocessing resulted in no tokens'
                }), 400
            preprocessed_term = preprocessed_terms[0]
        except Exception as e:
            logger.error(f"Error preprocessing term '{term}': {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Term preprocessing failed: {str(e)}'
            }), 400
        
        stats = search_engine.get_term_statistics(preprocessed_term)
        
        if stats is None:
            return jsonify({
                'success': True,
                'message': f'Term "{term}" not found in any documents'
            }), 200
        
        return jsonify({
            'success': True,
            'term_stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving term statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/document/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Delete a document from the index.
    
    This operation:
    1. Removes document from index
    2. Rebuilds index to recalculate TF-IDF
    3. Persists updated index to disk
    
    Args:
        doc_id (int): Document identifier
        
    Returns:
        Tuple[Dict, int]: JSON response and HTTP status code
    """
    try:
        if search_engine is None:
            return jsonify({
                'success': False,
                'error': 'Search engine not initialized'
            }), 400
        
        if doc_id not in search_engine.document_metadata:
            return jsonify({
                'success': False,
                'error': f'Document {doc_id} not found'
            }), 404
        
        metadata = search_engine.document_metadata[doc_id]
        doc_title = metadata['title']
        file_path = metadata.get('file_path')
        
        # Delete document and rebuild index
        if search_engine.delete_document(doc_id):
            # Delete raw document file from disk
            delete_document_file(file_path)

            # Persist updated index
            try:
                search_engine.save_index(app.config['INDEX_FILE'])
                logger.info(f"Document {doc_id} deleted and index persisted")
            except Exception as e:
                logger.error(f"Failed to persist index after deletion: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to save index after deletion'
                }), 500
            
            return jsonify({
                'success': True,
                'message': f'Document "{doc_title}" deleted successfully',
                'doc_id': doc_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to delete document {doc_id}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/document/<int:doc_id>', methods=['PUT'])
def update_document(doc_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Update a document's content and rebuild the index.
    
    Request JSON:
    {
        'title': str (optional),
        'text': str (optional)
    }
    
    Returns:
        Tuple[Dict, int]: JSON response and HTTP status code
    """
    try:
        if search_engine is None:
            return jsonify({
                'success': False,
                'error': 'Search engine not initialized'
            }), 400
        
        if doc_id not in search_engine.document_metadata:
            return jsonify({
                'success': False,
                'error': f'Document {doc_id} not found'
            }), 404
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body is empty'
            }), 400
        
        # Get current values or use new ones
        old_metadata = search_engine.document_metadata[doc_id]
        new_title = data.get('title', old_metadata['title']).strip()
        new_text = data.get('text', old_metadata['raw_text']).strip()
        
        if not new_title or not new_text:
            return jsonify({
                'success': False,
                'error': 'Title and text cannot be empty'
            }), 400
        
        # Persist the edited document back to raw_documents storage
        updated_file_path = write_document_file(old_metadata.get('file_path'), new_text, doc_id, new_title)
        old_metadata['file_path'] = updated_file_path

        # Update document and rebuild index
        if search_engine.update_document(doc_id, new_title, new_text):
            # Ensure the updated file path is preserved in metadata
            if 'file_path' in old_metadata:
                search_engine.document_metadata[doc_id]['file_path'] = updated_file_path

            # Persist updated index
            try:
                search_engine.save_index(app.config['INDEX_FILE'])
                logger.info(f"Document {doc_id} updated and index persisted")
            except Exception as e:
                logger.error(f"Failed to persist index after update: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to save index after update'
                }), 500
            
            return jsonify({
                'success': True,
                'message': f'Document updated successfully',
                'doc_id': doc_id,
                'title': new_title
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to update document {doc_id}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating document {doc_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error) -> Tuple[Dict, int]:
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path}")
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404


@app.errorhandler(500)
def internal_error(error) -> Tuple[Dict, int]:
    """Handle 500 errors."""
    logger.error(f"500 error: {str(error)}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


@app.errorhandler(413)
def request_entity_too_large(error) -> Tuple[Dict, int]:
    """Handle file too large errors."""
    logger.warning(f"413 error: File too large")
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: 50 MB'
    }), 413


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Starting Document IRS Flask Application")
    logger.info("=" * 80)
    
    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )

