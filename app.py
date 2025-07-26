import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
PERSIST_DIRECTORY = "chroma_db"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

def configure_gemini():
    """Loads API key and configures the Gemini model."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please create a .env file and add it.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def get_retriever():
    """Initializes the vector store retriever."""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Vector store not found at '{PERSIST_DIRECTORY}'. Please run ingest.py first.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 chunks

def get_rag_response(query, model, retriever):
    """Generates a response using the RAG pipeline."""
    print(f"\nRetrieving context for query: '{query}'")
    retrieved_docs = retriever.get_relevant_documents(query)

    # --- DEBUG: Print Retrieved Context ---
    print("\n--- Retrieved Context ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk {i+1}:\n{doc.page_content}\n")
    print("-------------------------")
    # -------------------------------------
    
    # Format the context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create a prompt
    template = """
Based on the following context, please provide a clear and direct answer to the question. Synthesize the information from the text to formulate your response. Answer in the same language as the question. The answer shoould be short and stragiht to the point.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=context, question=query)
    
    print("Generating answer...")
    response = model.generate_content(formatted_prompt)
    return response.text

# --- Main Interactive Chat Loop ---
def main():
    """Runs the interactive command-line chat application."""
    try:
        model = configure_gemini()
        retriever = get_retriever()
    except (ValueError, FileNotFoundError) as e:
        print(f"Error initializing the application: {e}")
        return

    print("\n--- Bilingual RAG System Ready ---")
    print("You can ask questions in English or Bengali.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'exit':
            break
        if user_query:
            answer = get_rag_response(user_query, model, retriever)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()

# --- Bonus Task: FastAPI Implementation ---
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    # FastAPI app instance
    api_app = FastAPI(title="Bilingual RAG API", description="Bengali Literature Q&A System")
    
    # Request model
    class QueryRequest(BaseModel):
        query: str
    
    # Response model
    class QueryResponse(BaseModel):
        query: str
        answer: str
    
    @api_app.post("/query", response_model=QueryResponse)
    async def api_query(request: QueryRequest):
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            model = configure_gemini()
            retriever = get_retriever()
            answer = get_rag_response(request.query.strip(), model, retriever)
            return QueryResponse(query=request.query, answer=answer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_app.get("/health")
    async def health_check():
        return {"status": "healthy", "message": "Bilingual RAG API is running"}
    
    def start_api():
        print("Starting FastAPI server...")
        print("API Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print("Query Endpoint: POST http://localhost:8000/query")
        uvicorn.run(api_app, host="0.0.0.0", port=8000)
        
except ImportError:
    def start_api():
        print("Install FastAPI: pip install fastapi uvicorn")


