# Bilingual RAG System

A Retrieval-Augmented Generation (RAG) system for Bengali literature question answering, specifically designed to handle Bengali PDF documents with MCQ and story content separation.

## Setup Guide

### Prerequisites
- Python 3.8+
- Google API key for Gemini model

### Installation

1. Clone the repository and navigate to the project directory

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Place your Bengali PDF file in the project directory and update the `PDF_PATH` in `ingest.py`

5. Run the data ingestion pipeline:
```bash
python ingest.py
```

6. Start the application:
```bash
python app.py
```

## Used Tools, Libraries, and Packages

### Core Dependencies
- **pdfplumber**: PDF text extraction with Bengali character support
- **langchain**: Text processing, chunking, and vector store management
- **langchain-community**: Community embeddings and vector store implementations
- **google-generativeai**: Google Gemini model integration
- **chromadb**: Vector database for semantic search
- **sentence-transformers**: Multilingual embedding model backend
- **fastapi**: Modern, fast web framework for building APIs
- **uvicorn**: ASGI server for FastAPI
- **python-dotenv**: Environment variable management

### Key Components
- **Embedding Model**: `paraphrase-multilingual-mpnet-base-v2`
- **Vector Store**: Chroma with persistent storage
- **LLM**: Google Gemini 2.0 Flash
- **Text Splitter**: RecursiveCharacterTextSplitter with Bengali-friendly separators

## Sample Queries and Outputs

### Bengali Queries

**Query 1:**
```
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
```
**Output:**
```
অনুপমের ভাষায় শম্ভুনাথকে সুপুরুষ বলা হয়েছে।
```

**Query 2:**
```
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
```
**Output:**
```
অনুপমের মামাকে ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে।
```

**Query 3:**
```
বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
```
**Output:**
```
দুঃখিত, এই প্রশ্নের উত্তর প্রদত্ত প্রসঙ্গে খুঁজে পাওয়া যায়নি।
```

### English Queries

**Query:**
```
Who is described as a good man in Anupam's language?
```
**Output:**
```
In Anupam's language, Shambhunath is described as a good man (সুপুরুষ).
```

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### POST /query
Submit a question to the RAG system.

**Request Body:**
```json
{
  "query": "Your question in Bengali or English"
}
```

**Response:**
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "অনুপমের ভাষায় শম্ভুনাথকে সুপুরুষ বলা হয়েছে।"
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "Bilingual RAG API is running"
}
```

**Example cURL Request:**
```bash
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

### Starting the API Server
```python
from app import start_api
start_api()
```

### Interactive API Documentation
FastAPI provides automatic interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Technical Implementation Details

### Text Extraction Method

**Library Used:** pdfplumber

**Why pdfplumber?**
- Superior Bengali character handling compared to alternatives like PyMuPDF
- Word-by-word extraction capability that preserves text structure
- Better handling of complex PDF layouts with mixed content types

**Formatting Challenges Faced:**
- Bengali character encoding issues in PDF extraction
- Mixed content types (story text vs MCQ format)
- Inconsistent spacing and line breaks in PDF structure
- Page boundary handling for continuous text flow

**Solution Implemented:**
Used word-by-word extraction with `page.extract_words()` method, which provides better control over text reconstruction and handles Bengali Unicode characters more reliably than standard text extraction methods.

### Chunking Strategy

**Method:** Hybrid approach combining content-type separation and character-limit chunking

**Strategy Details:**
1. **Content Separation**: Distinguish between story content (first 25 pages) and MCQ content (remaining pages)
2. **Character-limit Based**: 1000 characters per chunk with 100 character overlap
3. **Bengali-friendly Separators**: Custom separator hierarchy including Bengali sentence endings (।), punctuation, and whitespace

**Why This Works Well:**
- **Semantic Coherence**: Story and MCQ content have different structures and answer patterns
- **Context Preservation**: 100-character overlap ensures context continuity across chunk boundaries
- **Language-appropriate Splitting**: Bengali-specific separators respect linguistic boundaries
- **Balanced Chunk Size**: 1000 characters provide sufficient context without overwhelming the embedding model

### Embedding Model

**Model:** `paraphrase-multilingual-mpnet-base-v2`

**Why This Model:**
- **Multilingual Support**: Specifically trained on multiple languages including Bengali
- **Semantic Understanding**: MPNet architecture captures deep semantic relationships
- **Proven Performance**: Strong performance on paraphrase and semantic similarity tasks
- **Balanced Size**: Good trade-off between model size and performance

**How It Captures Meaning:**
- **Cross-lingual Embeddings**: Maps Bengali and English text to shared semantic space
- **Context-aware Encoding**: Considers word relationships and sentence structure
- **Semantic Similarity**: Enables meaningful comparison between questions and document chunks

### Similarity Comparison and Storage

**Vector Store:** ChromaDB with persistent storage

**Similarity Method:** Cosine similarity in high-dimensional embedding space

**Why This Setup:**
- **Efficient Retrieval**: ChromaDB optimized for similarity search operations
- **Persistent Storage**: Avoids re-embedding on each application restart
- **Scalable**: Can handle large document collections efficiently
- **Semantic Matching**: Cosine similarity captures semantic relationships better than keyword matching

**Comparison Process:**
1. Query is embedded using the same multilingual model
2. Cosine similarity computed between query embedding and all chunk embeddings
3. Top-k most similar chunks retrieved as context
4. Context provided to LLM for answer generation

### Query-Document Comparison Meaningfulness

**Ensuring Meaningful Comparison:**
- **Shared Embedding Space**: Both queries and documents use identical embedding model
- **Language Consistency**: Multilingual model handles Bengali-Bengali and English-Bengali comparisons
- **Content-type Awareness**: MCQ and story chunks both available for retrieval
- **Context Enrichment**: Multiple chunks provide comprehensive context to LLM

**Handling Vague or Missing Context Queries:**
- **Fallback Retrieval**: System retrieves best available matches even for vague queries
- **LLM Reasoning**: Gemini model can indicate when context is insufficient
- **Multiple Chunk Context**: Increases chance of finding relevant information
- **Graceful Degradation**: System provides partial answers when full context unavailable

### Results Relevance Assessment

**Current Performance:**
- **Story Questions**: Good performance when answers exist in narrative chunks
- **MCQ Questions**: Variable performance depending on chunk boundary alignment
- **Cross-reference Questions**: Challenges when answers span multiple content types

**Potential Improvements:**
1. **Better Chunking**: Implement question-aware chunking for MCQ content
2. **Hybrid Retrieval**: Combine semantic search with keyword matching for specific terms
3. **Larger Context Window**: Increase number of retrieved chunks for complex questions
4. **Fine-tuned Embeddings**: Train embeddings specifically on Bengali literature domain
5. **Answer Validation**: Implement confidence scoring and answer verification
6. **Document Expansion**: Include more diverse Bengali literature for better coverage

**Success Metrics:**
- Successful retrieval of relevant context for target questions
- Accurate answer generation for factual queries
- Proper handling of Bengali language nuances
- Consistent performance across different question types

### Current Challenges and Limitations

**MCQ Answer Table Retrieval Issue:**
One significant challenge is that some answers, particularly for MCQ questions like "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" (What was Kalyani's actual age at the time of marriage?), are present in tabular format on separate pages from the question content. 

**Why Semantic Search Fails:**
- **Context Separation**: The question appears in narrative/MCQ format while the answer exists in a structured table
- **Semantic Gap**: The table format ("৫৪. (খ) ১৫ বছর") has minimal semantic similarity to the question phrasing
- **Page Boundaries**: Current chunking treats table content as separate chunks with insufficient context linkage
- **Format Mismatch**: Semantic embeddings struggle to connect question intent with tabular answer format

**Impact:**
This results in the system being unable to retrieve relevant answer chunks for questions whose answers are stored in table format, leading to "cannot find answer" responses even when the information exists in the document.

## Project Structure

```
Bilingual-RAG/
├── app.py                 # Main application with chat interface and API
├── ingest.py             # Data ingestion and vector store creation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── chroma_db/           # Vector database storage (auto-created)
├── HSC26-Bangla1st-Paper.pdf  # Source document
└── README.md            # This documentation
```

## Usage Instructions

### Interactive Chat Mode
```bash
python app.py
```
Then ask questions in Bengali or English. Type 'exit' to quit.

### API Mode
```python
from app import start_api
start_api()
```
Then use HTTP requests to interact with the system.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **API Key Issues**: Verify your Google API key is correctly set in the `.env` file
3. **PDF Not Found**: Check that the PDF path in `ingest.py` matches your file location
4. **Encoding Issues**: Ensure your terminal supports UTF-8 for Bengali text display
5. **Vector Store Issues**: Delete the `chroma_db` folder and re-run `ingest.py` if needed

### Performance Optimization

- **First Run**: Initial embedding creation takes time; subsequent runs are faster
- **Memory Usage**: Large PDFs may require more RAM for processing
- **API Response Time**: First API call may be slower due to model loading

## Contributing

To extend this system:
1. Modify chunking strategy in `ingest.py` for different document types
2. Experiment with different embedding models in the configuration
3. Add evaluation metrics for systematic performance assessment
4. Implement additional API endpoints for specific use cases
