import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Function to load and extract text from PDF with proper Bengali handling
def load_and_extract_text(file_path):
    print(f"Loading and extracting text from {file_path}...")
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                try:
                    # Try word-by-word extraction for better Bengali handling
                    words = page.extract_words()
                    if words:
                        page_text = " ".join([word['text'] for word in words])
                        if page_text.strip():
                            full_text += f"\n=== PAGE {i+1} ===\n{page_text}\n"
                except Exception as word_error:
                    # Fallback to regular text extraction
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            full_text += f"\n=== PAGE {i+1} ===\n{page_text}\n"
                    except Exception as text_error:
                        print(f"Failed to extract text from page {i+1}: {text_error}")
                        continue
        
        print(f"Text extraction successful. Total characters: {len(full_text)}")
        
        # Save extracted text for debugging
        try:
            with open("extracted_text_debug.txt", "w", encoding="utf-8") as f:
                f.write(full_text)
            print("Debug: Extracted text saved to 'extracted_text_debug.txt'")
        except Exception as save_error:
            print(f"Warning: Could not save debug file: {save_error}")
        
        return full_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

# Function to split text into chunks with MCQ and story separation
def chunk_text(text):
    print("Chunking text with MCQ and story separation...")
    
    # Split text by pages first
    pages = text.split("=== PAGE")
    
    story_chunks = []
    mcq_chunks = []
    
    # Identify story pages (typically first 20-25 pages) and MCQ pages (rest)
    for i, page_content in enumerate(pages):
        if not page_content.strip():
            continue
            
        page_num = i  # Approximate page number
        
        # Story content is typically in first 25 pages
        if page_num <= 25:
            # This is likely story content
            if len(page_content.strip()) > 100:  # Only chunk substantial content
                story_chunks.append(f"[STORY] {page_content.strip()}")
        else:
            # This is likely MCQ content
            if len(page_content.strip()) > 50:  # MCQs can be shorter
                mcq_chunks.append(f"[MCQ] {page_content.strip()}")
    
    # Now chunk the story and MCQ content separately
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", "?", "!", ";", ":", ",", " ", ""]
    )
    
    # Chunk story content
    all_story_text = "\n\n".join(story_chunks)
    story_text_chunks = text_splitter.split_text(all_story_text)
    
    # Chunk MCQ content
    all_mcq_text = "\n\n".join(mcq_chunks)
    mcq_text_chunks = text_splitter.split_text(all_mcq_text)
    
    # Combine all chunks
    all_chunks = story_text_chunks + mcq_text_chunks
    
    print(f"Created {len(story_text_chunks)} story chunks and {len(mcq_text_chunks)} MCQ chunks")
    print(f"Total chunks: {len(all_chunks)}")
    
    return all_chunks

# Function to create a vector store from text chunks
def create_vector_store(chunks):
    print("Initializing embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Creating and persisting vector store...")
    db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    print("Vector store created successfully.")
    return db

# Main function to run the data ingestion pipeline
def main():
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at '{PDF_PATH}'")
        print("Please make sure the file 'HSC26 Bangla 1st paper.pdf' is in the same directory.")
        return

    document_text = load_and_extract_text(PDF_PATH)
    if document_text:
        text_chunks = chunk_text(document_text)
        create_vector_store(text_chunks)
        print("\n--- Ingestion Complete ---")
        print(f"The knowledge base has been created and saved in the '{PERSIST_DIRECTORY}' folder.")

if __name__ == "__main__":
    main()
