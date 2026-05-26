"""
Document Processor for the Document Information Retrieval System.

This module implements a professional document processing pipeline that can
read .txt and .pdf files, extract raw text, and normalize text for search.

The processing pipeline includes:
- Text extraction from supported file formats
- Lowercase conversion
- Regex-based tokenization with punctuation removal
- Stop-word filtering (English)
- Lemmatization using NLTK

Author: Senior Python Developer
Version: 1.0.0
"""

import logging
import re
from pathlib import Path
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/document_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are not already available."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


class DocumentProcessor:
    """
    DocumentProcessor extracts text from supported documents and prepares
    tokens for indexing and search.

    Attributes:
        stop_words (set): English stop words from the NLTK corpus.
        lemmatizer (WordNetLemmatizer): Lemmatizer instance.
        max_file_size (int): Maximum file size accepted for extraction.
    """

    def __init__(self, max_file_size: int = 50 * 1024 * 1024) -> None:
        """
        Initialize the document processor.

        Args:
            max_file_size (int): Maximum file size allowed for uploaded documents.
        """
        try:
            _ensure_nltk_resources()
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.max_file_size = max_file_size
            logger.info("DocumentProcessor initialized successfully")
        except Exception as exc:
            logger.error(f"DocumentProcessor initialization failed: {exc}")
            raise RuntimeError(f"Failed to initialize DocumentProcessor: {exc}")

    def extract_text_from_txt(self, file_path: str) -> str:
        """
        Extract raw text from a .txt file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Extracted file content.

        Raises:
            FileNotFoundError: When the file is missing.
            ValueError: When file extension is invalid or file is too large.
            RuntimeError: When text extraction fails.
        """
        path = Path(file_path)
        try:
            if not path.exists():
                raise FileNotFoundError(f"TXT file not found: {file_path}")
            if path.suffix.lower() != '.txt':
                raise ValueError(f"Expected .txt file, got '{path.suffix}'")

            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File size exceeds limit: {file_size} bytes > {self.max_file_size} bytes"
                )

            text = None
            for encoding in ('utf-8', 'latin-1', 'iso-8859-1', 'cp1252'):
                try:
                    with path.open('r', encoding=encoding) as file_handle:
                        text = file_handle.read()
                    logger.info(f"Read TXT file '{file_path}' with encoding {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise RuntimeError("Unable to decode TXT file using supported encodings")

            cleaned_text = text.replace('\x00', '')
            cleaned_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned_text)
            return cleaned_text.strip()

        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error(f"TXT extraction failed for '{file_path}': {exc}")
            raise RuntimeError(f"Failed to extract text from TXT file: {exc}")

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract raw text from a .pdf file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from all readable pages.

        Raises:
            FileNotFoundError: When the file is missing.
            ValueError: When file extension is invalid or file is too large.
            RuntimeError: When PDF extraction fails.
        """
        path = Path(file_path)
        try:
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            if path.suffix.lower() != '.pdf':
                raise ValueError(f"Expected .pdf file, got '{path.suffix}'")

            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(
                    f"File size exceeds limit: {file_size} bytes > {self.max_file_size} bytes"
                )

            reader = PdfReader(str(path))
            if not reader.pages:
                logger.warning(f"PDF file '{file_path}' contains zero pages")
                return ''

            content_parts: List[str] = []
            for page_index, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ''
                    content_parts.append(page_text)
                except Exception as page_exc:
                    logger.warning(
                        f"Failed to extract page {page_index} from '{file_path}': {page_exc}"
                    )
                    continue

            extracted = '\n'.join(content_parts)
            extracted = extracted.replace('\x00', '')
            extracted = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', extracted)
            logger.info(f"Extracted text from PDF '{file_path}' ({len(content_parts)} pages)")
            return extracted.strip()

        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            logger.error(f"PDF extraction failed for '{file_path}': {exc}")
            raise RuntimeError(f"Failed to extract text from PDF file: {exc}")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a supported document format.

        Args:
            file_path (str): Path to the document file.

        Returns:
            str: Extracted text.

        Raises:
            ValueError: When the file extension is unsupported.
            FileNotFoundError: When the file is missing.
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == '.txt':
            return self.extract_text_from_txt(file_path)
        if extension == '.pdf':
            return self.extract_text_from_pdf(file_path)

        logger.error(f"Unsupported file extension '{extension}' for file '{file_path}'")
        raise ValueError(f"Unsupported file format '{extension}'. Supported: .txt, .pdf")

    def preprocess_text(self, raw_text: str) -> List[str]:
        """
        Preprocess raw text for indexing and search.

        Steps:
            1. Lowercase conversion
            2. Regex tokenization removing punctuation
            3. Stop-word removal
            4. Lemmatization

        Args:
            raw_text (str): Raw text content.

        Returns:
            List[str]: Preprocessed token list.

        Raises:
            ValueError: When the input text is empty or invalid.
            RuntimeError: When preprocessing fails.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            logger.error("Preprocessing received invalid or empty text")
            raise ValueError("Input text must be a non-empty string")

        try:
            normalized = raw_text.lower()
            normalized = normalized.replace('\n', ' ').replace('\r', ' ')
            normalized = re.sub(r'\s+', ' ', normalized).strip()

            tokens = re.findall(r"[a-z0-9]+", normalized)
            logger.debug(f"Tokenized {len(tokens)} raw tokens")

            filtered = [token for token in tokens if token not in self.stop_words]
            logger.debug(f"Filtered to {len(filtered)} tokens after stop-word removal")

            lemmatized = [self.lemmatizer.lemmatize(token) for token in filtered]
            logger.info(f"Preprocessed text into {len(lemmatized)} tokens")
            return lemmatized

        except Exception as exc:
            logger.error(f"Preprocessing failed: {exc}")
            raise RuntimeError(f"Text preprocessing failed: {exc}")

    def process_document(self, file_path: str) -> List[str]:
        """
        Extract and preprocess text from a document file.

        Args:
            file_path (str): Path to the document file.

        Returns:
            List[str]: Preprocessed tokens.

        Raises:
            Exception: When extraction or preprocessing fails.
        """
        try:
            raw_text = self.extract_text(file_path)
            if not raw_text:
                logger.warning(f"No text found in document '{file_path}'")
                return []
            return self.preprocess_text(raw_text)
        except Exception:
            logger.exception(f"Failed to process document '{file_path}'")
            raise
