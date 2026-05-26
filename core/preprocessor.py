"""
Preprocessor Module for Document Information Retrieval System

This module provides text extraction and NLP preprocessing capabilities for various
document formats (TXT, PDF). It includes tokenization, normalization, stop-word removal,
and lemmatization using NLTK.

Author: Senior Enterprise Developer
Version: 1.0.0
"""

import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/preprocessor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Download required NLTK data on module import
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK tokenizers/punkt...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords corpus...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK wordnet corpus...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    logger.info("Downloading NLTK omw corpus...")
    nltk.download('omw-1.4', quiet=True)


class DocumentProcessor:
    """
    Professional document processor for text extraction and NLP preprocessing.
    
    Supports multiple document formats (.txt, .pdf) and provides comprehensive
    text preprocessing including tokenization, normalization, stop-word removal,
    and lemmatization.
    
    Attributes:
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer instance
        stop_words (set): English stop words from NLTK corpus
        max_file_size (int): Maximum file size in bytes (50 MB default)
    """
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024) -> None:
        """
        Initialize the DocumentProcessor.
        
        Args:
            max_file_size (int): Maximum allowed file size in bytes. Default is 50 MB.
            
        Raises:
            RuntimeError: If NLTK resources cannot be loaded.
        """
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.max_file_size = max_file_size
            logger.info("DocumentProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentProcessor: {str(e)}")
            raise RuntimeError(f"DocumentProcessor initialization failed: {str(e)}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract text from a .txt file with robust error handling.
        
        Args:
            file_path (str): Path to the .txt file
            
        Returns:
            str: Extracted text content, empty string on failure
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If file is not a .txt file or exceeds size limit
        """
        try:
            # Validate file path and extension
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if path.suffix.lower() != '.txt':
                raise ValueError(f"Expected .txt file, got {path.suffix}")
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)"
                )
            
            # Read file with multiple encoding attempts
            text = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    logger.info(f"Successfully read TXT file: {file_path} (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise RuntimeError("Could not decode file with any supported encoding")
            
            # Remove null bytes and control characters
            text = text.replace('\x00', '')
            text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
            
            return text.strip()
            
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError while reading TXT: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"ValueError while reading TXT: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading TXT file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text from TXT: {str(e)}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a .pdf file with graceful error handling.
        
        Args:
            file_path (str): Path to the .pdf file
            
        Returns:
            str: Extracted text content, empty string on failure
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If file is not a .pdf file or exceeds size limit
        """
        try:
            # Validate file path and extension
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if path.suffix.lower() != '.pdf':
                raise ValueError(f"Expected .pdf file, got {path.suffix}")
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File size ({file_size} bytes) exceeds maximum ({self.max_file_size} bytes)"
                )
            
            # Extract text from PDF
            text_content = []
            try:
                pdf_reader = PdfReader(file_path)
                num_pages = len(pdf_reader.pages)
                
                if num_pages == 0:
                    logger.warning(f"PDF file {file_path} contains no pages")
                    return ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
                
                extracted_text = "\n".join(text_content)
                logger.info(f"Successfully extracted text from PDF: {file_path} ({num_pages} pages)")
                
            except Exception as e:
                logger.error(f"PyPDF error reading {file_path}: {str(e)}")
                raise RuntimeError(f"Failed to read PDF file: {str(e)}")
            
            # Clean extracted text
            extracted_text = extracted_text.replace('\x00', '')
            extracted_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', extracted_text)
            
            return extracted_text.strip()
            
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError while reading PDF: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"ValueError while reading PDF: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading PDF file {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text(self, file_path: str) -> str:
        """
        Unified text extraction method that handles multiple file formats.
        
        Args:
            file_path (str): Path to the document file (.txt or .pdf)
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If the file does not exist
        """
        try:
            path = Path(file_path)
            file_extension = path.suffix.lower()
            
            if file_extension == '.txt':
                return self.extract_text_from_txt(file_path)
            elif file_extension == '.pdf':
                return self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Supported: .txt, .pdf")
                
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"File extraction error for {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during text extraction from {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text: {str(e)}")
    
    def preprocess_text(self, raw_text: str) -> List[str]:
        """
        Comprehensive NLP preprocessing pipeline.
        
        Performs the following operations:
        1. Lowercase conversion
        2. Tokenization with punctuation/special character removal
        3. English stop-words removal
        4. Lemmatization
        
        Args:
            raw_text (str): Raw text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens
            
        Raises:
            ValueError: If input text is empty or None
        """
        try:
            # Validate input
            if not raw_text or not isinstance(raw_text, str):
                raise ValueError("Input text must be a non-empty string")
            
            # Step 1: Lowercase conversion
            text = raw_text.lower()
            logger.debug("Lowercase conversion completed")
            
            # Step 2: Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Step 3: Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Step 4: Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Step 5: Remove special characters and keep only alphanumeric and spaces
            text = re.sub(r"[^a-z0-9\s]", ' ', text)
            
            # Step 6: Tokenization using regex to avoid external NLTK tokenizer dependencies
            tokens = re.findall(r"\b[a-z0-9]+\b", text)
            logger.debug(f"Tokenization completed. Tokens generated: {len(tokens)}")
            
            # Step 7: Filter out short tokens (single characters) and stop words
            filtered_tokens = [
                token for token in tokens
                if len(token) > 1 and token not in self.stop_words
            ]
            logger.debug(f"Stop-words removal completed. Remaining tokens: {len(filtered_tokens)}")
            
            # Step 8: Lemmatization
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in filtered_tokens
            ]
            logger.debug(f"Lemmatization completed. Final tokens: {len(lemmatized_tokens)}")
            
            return lemmatized_tokens
            
        except ValueError as e:
            logger.error(f"ValueError during preprocessing: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during text preprocessing: {str(e)}")
            raise RuntimeError(f"Text preprocessing failed: {str(e)}")
    
    def process_document(self, file_path: str) -> Tuple[str, List[str]]:
        """
        Complete document processing pipeline: extraction + preprocessing.
        
        Extracts text from a document file and returns both the raw and
        preprocessed versions.
        
        Args:
            file_path (str): Path to the document file (.txt or .pdf)
            
        Returns:
            Tuple[str, List[str]]: (raw_text, preprocessed_tokens)
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If file format is unsupported
            RuntimeError: If processing fails
        """
        try:
            logger.info(f"Starting document processing for: {file_path}")
            
            # Extract text
            raw_text = self.extract_text(file_path)
            
            if not raw_text:
                logger.warning(f"No text extracted from {file_path}")
                return raw_text, []
            
            # Preprocess text
            preprocessed_tokens = self.preprocess_text(raw_text)
            
            logger.info(
                f"Document processing completed. "
                f"Raw tokens: {len(raw_text.split())}, "
                f"Preprocessed tokens: {len(preprocessed_tokens)}"
            )
            
            return raw_text, preprocessed_tokens
            
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Document processing failed for {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during document processing: {str(e)}")
            raise RuntimeError(f"Document processing pipeline failed: {str(e)}")
    
    def batch_process_documents(
        self, 
        file_paths: List[str],
        skip_errors: bool = True
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Process multiple documents in batch mode.
        
        Args:
            file_paths (List[str]): List of file paths to process
            skip_errors (bool): If True, skip files that fail; if False, raise exception
            
        Returns:
            List[Tuple[str, str, List[str]]]: List of (file_path, raw_text, tokens)
            
        Raises:
            RuntimeError: If skip_errors is False and any file fails
        """
        try:
            results = []
            failed_files = []
            
            logger.info(f"Starting batch processing of {len(file_paths)} documents")
            
            for idx, file_path in enumerate(file_paths, 1):
                try:
                    raw_text, tokens = self.process_document(file_path)
                    results.append((file_path, raw_text, tokens))
                    logger.info(f"[{idx}/{len(file_paths)}] Successfully processed: {file_path}")
                except Exception as e:
                    logger.error(f"[{idx}/{len(file_paths)}] Failed to process {file_path}: {str(e)}")
                    failed_files.append((file_path, str(e)))
                    if not skip_errors:
                        raise RuntimeError(f"Batch processing failed at {file_path}: {str(e)}")
            
            if failed_files:
                logger.warning(f"Batch processing completed with {len(failed_files)} failures")
                for file_path, error in failed_files:
                    logger.warning(f"  - {file_path}: {error}")
            else:
                logger.info(f"Batch processing completed successfully for all {len(results)} documents")
            
            return results
            
        except RuntimeError as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch processing: {str(e)}")
            raise RuntimeError(f"Batch processing failed: {str(e)}")
