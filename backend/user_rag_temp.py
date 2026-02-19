from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool
import sqlite3
from backend.config import settings
import os

# RAG components
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

# Import Custom Error Handlers
from .error_handlers import (
    setup_logger, 
    DocumentProcessingError, 
    RAGError, 
    DatabaseError
)

# Initialize Logger
logger = setup_logger("Core_RAG")

vectorstore_cache = {}
# ========================================
# Setup
# ========================================

@lru_cache(maxsize=1)
def get_llm():
    logger.info("Loading LLM...")
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL_NAME,
        api_key=settings.GEMINI_API_KEY,
        temperature=0.3 )  # faster + more stable

@lru_cache(maxsize=1)
def get_embeddings():
    logger.info("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL  # use MiniLM for speed
    )


@lru_cache(maxsize=1)
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=500,     # ⚡ smaller = faster
        chunk_overlap=50
    )


def preload_models():
    logger.info("Preloading models...")
    get_llm()
    get_embeddings()
    get_text_splitter()
    logger.info("Models ready!")
# ========================================
# User-Specific Vector Store Manager
# ========================================

class UserVectorStoreManager:
    """Manages separate vector stores for each user"""

    def __init__(self, base_dir: str = settings.CHROMA_DB_PATH):
        self.base_dir = base_dir
        self._stores = {}

    def get_collection_name(self, user_id: str) -> str:
        return f"user_{user_id}_documents"

    def get_vectorstore(self, user_id: str) -> Chroma:
        """Get or create vector store for specific user"""

        if user_id in self._stores:
            return self._stores[user_id]

        collection_name = self.get_collection_name(user_id)
        persist_dir = f"{self.base_dir}/{user_id}"

        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),  # ✅ FIXED
            persist_directory=persist_dir
        )

        self._stores[user_id] = vectorstore
        logger.info(f"Vector store loaded for user {user_id}")

        return vectorstore

    def get_retriever(self, user_id: str):
        """MMR retriever (optimized)"""

        vectorstore = self.get_vectorstore(user_id)

        return vectorstore.as_retriever(
            search_type="mmr",   # ✅ KEEP MMR
            search_kwargs={
                "k": 3,                 # ⚡ reduce final results
                "fetch_k": 8,           # ⚡ lower = faster
                "lambda_mult": 0.6      # ⚖️ balance relevance/diversity
            }
        )

    
vector_manager = UserVectorStoreManager()

def get_cached_vectorstore(user_id):
    global vectorstore_cache
    
    if user_id not in vectorstore_cache:
        vectorstore_cache[user_id] = vector_manager.get_vectorstore(user_id)
    
    return vectorstore_cache[user_id]


# ========================================
# Document Management Functions
# ========================================

# def ingest_document(user_id:str,file_path: str) -> dict:
#     """Ingest document into RAG system with proper validation and error handling"""
#     logger.info(f"Starting ingestion for user {user_id}: {file_path}")
    
#     if not os.path.exists(file_path):
#         logger.error(f"File not found: {file_path}")
#         raise DocumentProcessingError(f"File not found: {file_path}")

#     try:
#         if file_path.endswith('.pdf'):
#             loader = PyPDFLoader(file_path)
#         elif file_path.endswith('.txt'):
#             loader = TextLoader(file_path)
#         else:
#             logger.warning(f"Unsupported file type: {file_path}")
#             return {"status": "error", "message": "Unsupported file type. Use PDF or TXT."}
        
#         documents = loader.load()
#         if not documents:
#             logger.error("No text extracted from document")
#             raise DocumentProcessingError("No text could be extracted from the document")

#         chunks = text_splitter.split_documents(documents)
#         if not chunks:
#             logger.error("Document yielded 0 chunks")
#             raise DocumentProcessingError("Document is empty or could not be chunked")

#         for chunk in chunks:
#             chunk.metadata["source"] = file_path
#             chunk.metadata["user_id"] = user_id


#         vectorstore=vector_manager.get_vectorstore(user_id)
        
#         BATCH_SIZE = 4000  # safe under Chroma limit

#         for i in range(0, len(chunks), BATCH_SIZE):
#             batch = chunks[i:i + BATCH_SIZE]
#             vectorstore.add_documents(batch)

        
#         logger.info(f"Successfully ingested {len(chunks)} chunks from {os.path.basename(file_path)} with user {user_id}")
        
#         return {
#             "status": "success",
#             "message": f"Successfully processed {len(chunks)} chunks",
#             "chunks": len(chunks),
#             "filename": os.path.basename(file_path)
#         }

#     except DocumentProcessingError:
#         raise
#     except Exception as e:
#         logger.exception(f"Unexpected error during ingestion of {file_path}")
#         raise DocumentProcessingError(f"Failed to process document: {str(e)}")

def ingest_document(user_id: str, file_path: str) -> dict:
    logger.info(f"Ingesting: {file_path}")

    if not os.path.exists(file_path):
        raise DocumentProcessingError("File not found")

    try:
        # Loader
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            return {"status": "error", "message": "Unsupported file type"}

        documents = loader.load()
        if not documents:
            raise DocumentProcessingError("Empty document")

        splitter = get_text_splitter()
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata.update({
                "source": file_path,
                "user_id": user_id
            })

        vs = get_cached_vectorstore(user_id)

            
        BATCH_SIZE = 100

        for i in range(0, len(chunks), BATCH_SIZE):
            vs.add_documents(chunks[i:i + BATCH_SIZE])

        return {
            "status": "success",
            "chunks": len(chunks),
            "filename": os.path.basename(file_path)
        }

    except Exception as e:
        logger.exception("Ingestion failed")
        raise DocumentProcessingError(str(e))


def has_documents(user_id: str) -> bool:
    try:
        return get_cached_vectorstore(user_id)._collection.count() > 0
    except:
        return False


# def list_documents(user_id=str) -> list:
#     try:
#         vectorstore = vector_manager.get_vectorstore(user_id)
#         collection = vectorstore._collection
#         all_docs = collection.get()
#         sources = set()
#         for meta in all_docs.get("metadatas", []):
#             if meta and "source" in meta and meta.get('user_id')==user_id:
#                 sources.add(os.path.basename(meta["source"]))
#         return sorted(sources)
#     except:
#         return []
def list_documents(user_id: str) -> list:
    try:
        data = get_cached_vectorstore(user_id)._collection.get()
        return list({
            os.path.basename(m["source"])
            for m in data.get("metadatas", [])
            if m and m.get("user_id") == user_id
        })
    except:
        return []


# def delete_document(user_id:str,filename: str) -> dict:
#     """Delete all chunks for a single uploaded document"""
#     logger.info(f"Attempting to delete document: {filename}")
#     try:
#         vectorstore = vector_manager.get_vectorstore(user_id)
#         collection = vectorstore._collection
#         all_docs = collection.get()
#         if not all_docs or 'metadatas' not in all_docs:
#             logger.warning("Delete failed: No documents in database")
#             raise RAGError("No documents found in the database")

#         ids_to_delete = [
#             all_docs['ids'][i]
#             for i, metadata in enumerate(all_docs['metadatas'])
#             if metadata.get("source") and os.path.basename(metadata["source"]) == filename and metadata.get('user_id')==user_id
            
#         ]

#         if not ids_to_delete:
#             logger.warning(f"Delete failed: No chunks found for {filename}")
#             raise RAGError(f"File not found in database: {filename}")

#         vectorstore._collection.delete(ids=ids_to_delete)
#         logger.info(f"Successfully deleted {len(ids_to_delete)} chunks for {filename} from user {user_id}")
        
#         return {"status": "success", "message": f"Deleted {len(ids_to_delete)} chunks"}

#     except RAGError:
#         raise
#     except Exception as e:
#         logger.exception(f"Error deleting document {filename}")
#         raise RAGError(f"Failed to delete document: {str(e)}")

def delete_document(user_id: str, filename: str) -> dict:
    logger.info(f"Deleting document: {filename} for user {user_id}")
    try:
        vectorstore = get_cached_vectorstore(user_id)

        # ✅ Direct DB filtering (NO full load)
        vectorstore._collection.delete(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"source": {"$contains": filename}}
                ]
            }
        )

        logger.info(f"Deleted document: {filename}")
        return {"status": "success", "message": f"Deleted document: {filename}"}

    except Exception as e:
        logger.exception("Delete failed")
        raise RAGError(f"Failed to delete document: {str(e)}")



# def clear_documents(user_id:str) -> dict:
#     """Safely clear all documents from Chroma (Windows-safe)"""
#     logger.info("Attempting to clear all documents")
#     try:
#         vectorstore = vector_manager.get_vectorstore(user_id)
#         collection = vectorstore._collection

#         count = collection.count()
#         if count == 0:
#             logger.info("Clear skipped: No documents to clear")
#             return {"status": "success", "message": "No documents to clear"}

#         all_ids = collection.get()["ids"]

#         if all_ids:
#             collection.delete(ids=all_ids)
#             logger.info(f"Successfully cleared {len(all_ids)} documents")

#         return {
#             "status": "success",
#             "message": f"Cleared {len(all_ids)} documents"
#         }

#     except Exception as e:
#         logger.exception("Error clearing all documents")
#         raise RAGError(f"Failed to clear documents: {str(e)}")

def clear_documents(user_id: str) -> dict:
    logger.info(f"Clearing documents for user {user_id}")
    try:
        vectorstore  = get_cached_vectorstore(user_id)

        # ✅ Direct delete using filter
        vectorstore._collection.delete(
            where={"user_id": user_id}
        )

        logger.info("All documents cleared")
        return {"status": "success", "message": "All documents cleared"}

    except Exception as e:
        logger.exception("Clear failed")
        raise RAGError(f"Failed to clear documents: {str(e)}")



# ========================================
# RAG Tool
# ========================================

# def create_search_tool(user_id: str):
#     @tool
#     def search_documents(query: str) -> str:
#         """
#         Search through PDF and TXT documents to find specific information.
#         Use this tool ONLY when user asks about uploaded files.
#         """
#         try:
#             logger.info(f"Tool 'search_documents' called with query: '{query}' by user {user_id}")
            
#             if not query.strip():
#                 return "Please provide a valid search query."

#             if not has_documents(user_id):
#                 logger.info("Search skipped: No documents available")
#                 return "No documents have been uploaded yet."
            
#             try:
#                 #results = vectorstore.similarity_search(query, k=3)
#                 retriever = vector_manager.get_retriever(user_id)
#                 results=retriever.invoke(query)
#             except Exception as e:
#                 logger.error(f"Vector store search faileduvicorn backend.main:app --host 127.0.0.1 --port 8000: {e}")
#                 return "I encountered an error searching the documents."
            
#             if not results:
#                 logger.info(f"No results found for query: {query}")
#                 return f"I searched but couldn't find relevant information about '{query}'."
            
#             response_parts = []
#             citations = []

#             for i, doc in enumerate(results, start=1):
#                 filename = os.path.basename(doc.metadata.get("source", "unknown"))
#                 page = doc.metadata.get("page", "N/A")
#                 response_parts.append(
#                     f"[{i}] From **{filename}** (page {page}):\n{doc.page_content}"
#                 )
#                 citations.append(f"[{i}] {filename}, page {page}")

#             response = "\n\n".join(response_parts)
#             sources = "\n".join(citations)

#             return f"**Found in uploaded documents:**\n\n{response}\n\n**Sources:**\n{sources}"

#         except Exception as e:
#             logger.error(f"Unexpected error in search tool: {e}")
#             return f"Error searching documents: {str(e)}"
    
#     return search_documents 
def create_search_tool(user_id: str):
    @tool
    def search_documents(query: str) -> str:

        if not query.strip():
            return "Please enter a valid query."

        if not has_documents(user_id):
            return "No documents uploaded."

        try:
            retriever =  vector_manager.get_retriever(user_id)
            results = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return "Error searching documents."

        if not results:
            return "I cannot find this information in the provided documents."

        response, sources = [], []

        for i, doc in enumerate(results, 1):
            fname = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")

            response.append(f"[{i}] {doc.page_content}")
            sources.append(f"[{i}] {fname}, page {page}")

        return (
            "**Found in uploaded documents:**\n\n"
            + "\n\n".join(response)
            + "\n\n**Sources:**\n"
            + "\n".join(sources)
        )

    return search_documents

# ========================================
# Agent with System Instructions
# ========================================

SYSTEM_PROMPT = """
You are an AI assistant for a document-based question answering system.

STRICT RULES (DO NOT VIOLATE):

1. If you use the tool `search_documents`, you MUST:
   - Use ONLY the information returned by the tool
   - Return the tool response EXACTLY as it is
   - DO NOT rewrite, summarize, or paraphrase
   - DO NOT remove filenames, page numbers, or sources
   - ALWAYS include the **Sources** section

2. If the answer comes from your general knowledge:
   - Do NOT mention documents
   - Do NOT include sources

3. NEVER say:
   - "Based on the documents" WITHOUT sources
   - "The documents mention" WITHOUT citations

4. If the tool says no relevant information was found:
   - Say: "I cannot find this information in the provided documents."
   - Do NOT use general knowledge.

IMPORTANT:
If a tool response contains a "Sources" section, it MUST appear verbatim in your final answer.
"""  
# Setup Sync Checkpointer (Works with Global Object)
# conn = sqlite3.connect(database='chat_database.db', check_same_thread=False)
# conn = sqlite3.connect(
#     database=settings.SQLITE_DB_PATH,
#     check_same_thread=False
# )
# checkpointer = SqliteSaver(conn=conn)

# def create_user_agent(user_id: str):
#     search_tool = create_search_tool(user_id)
#     tools = [search_tool]
    
#     agent = create_agent(
#         model=model,
#         tools=tools,
#         checkpointer=checkpointer
#     )
    
#     return agent

_agent_cache = {}

def create_user_agent(user_id: str):
    if user_id in _agent_cache:
        return _agent_cache[user_id]

    agent = create_agent(
        model=get_llm(),
        tools=[create_search_tool(user_id)],
        checkpointer=get_checkpointer()
    )

    _agent_cache[user_id] = agent
    return agent

@lru_cache(maxsize=1)
def get_checkpointer():
    conn = sqlite3.connect(
        database=settings.SQLITE_DB_PATH,
        check_same_thread=False
    )
    return SqliteSaver(conn=conn)

# ========================================
# Thread Management
# ========================================

# def get_user_thread_id(user_id: str, thread_id: str) -> str:
#     """Create namespaced thread ID for user isolation"""
#     return f"{user_id}:{thread_id}"


# def retrieve_user_threads(user_id: str) -> list:
#     """Get all thread IDs safely"""
#     all_threads = set()
#     user_prefix = f"{user_id}:"
#     try:
#         # Checkpointer.list returns generator of CheckpointTuple
#         for checkpoint in checkpointer.list(None):
#             try:
#                 # config is inside the checkpoint tuple usually
#                 # structure: CheckpointTuple(config, checkpoint, metadata, ...)
#                 if hasattr(checkpoint, 'config') and 'configurable' in checkpoint.config:
#                     thread_id = checkpoint.config['configurable'].get('thread_id')
#                     if thread_id and thread_id.startswith(user_prefix):
#                         clean_id = thread_id[len(user_prefix):]
#                         all_threads.add(clean_id)
#             except Exception:
#                 continue
#     except Exception:
#         pass
#     return list(all_threads)


# def get_thread_history_safe(user_id:str,thread_id: str) -> list:
#     """Get thread history safely"""
#     try:
#         namespaced_id = get_user_thread_id(user_id, thread_id)
#         agent = create_user_agent(user_id)
#         config = {"configurable": {"thread_id": namespaced_id}}
#         state = agent.get_state(config)
        
#         if not state or not state.values:
#             return []
        
#         messages = state.values.get("messages", [])
#         return messages if messages else []
#     except:
#         return []

def get_user_thread_id(user_id: str, thread_id: str) -> str:
    return f"{user_id}:{thread_id}"


def retrieve_user_threads(user_id: str) -> list:
    try:
        cp = get_checkpointer()
        prefix = f"{user_id}:"

        return list({
            chk.config["configurable"]["thread_id"][len(prefix):]
            for chk in cp.list(None)
            if hasattr(chk, "config")
            and "configurable" in chk.config
            and chk.config["configurable"].get("thread_id", "").startswith(prefix)
        })
    except:
        return []


def get_thread_history_safe(user_id: str, thread_id: str) -> list:
    try:
        agent = create_user_agent(user_id)
        config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
        state = agent.get_state(config)
        return state.values.get("messages", []) if state else []
    except:
        return []