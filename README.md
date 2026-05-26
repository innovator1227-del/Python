# Document Information Retrieval and Storage (IRS) System

A professional enterprise-grade document retrieval system built with Python and Flask.

## Architecture

### Core Components

- **Core Engine**: TF-IDF-based document indexing and cosine similarity ranking
- **Web Application**: Flask REST API with HTML/CSS/JS frontend
- **Data Storage**: Raw documents and processed indexes management

### Project Structure

```
IRS_System/
├── config/          # Configuration and constants
├── core/            # TF-IDF engine and retrieval logic
├── app/             # Flask web application
├── data/            # Document and index storage
└── tests/           # Unit and integration tests
```

## Installation

1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the application:

```bash
python main.py
```

## Features

- Document upload and preprocessing
- TF-IDF indexing
- Cosine similarity-based ranking
- REST API endpoints
- Web UI for search and upload
- Admin statistics

## Technologies

- Python 3.8+
- Flask 2.3
- NumPy, Scikit-learn
- HTML5, CSS3, JavaScript

## License

Corporate Proprietary
