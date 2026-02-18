from fastapi import FastAPI, HTTPException, UploadFile, Request, File, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field,EmailStr
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import shutil, os, uuid, time
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .user_db_file import user_db
from fastapi import BackgroundTasks
from pypdf import PdfReader
from .auth_utiles import (
      create_access_token,
      create_refresh_token,
      Token,
      verify_token,
      UserCreate,
      UserLogin, TokenData
)



# Import Core Logic and Error Handlers
# from .user_rag_temp import (
#     create_user_agent,
#     retrieve_user_threads,
#     get_thread_history_safe,
#     get_user_thread_id,
#     vector_manager,
#     ingest_document,
#     has_documents,
#     clear_documents,
#     SYSTEM_PROMPT,
#     list_documents,
#     delete_document
# )

from .error_handlers import setup_logger, AppError, DocumentProcessingError, RAGError, DatabaseError
from .config import settings

# ========================================
# Initialize FastAPI & Logger
# ========================================
logger = setup_logger("FastAPI")

app = FastAPI(
    title="RAG Based Chatbot API",
    description="Backend API for AI chatbot with conversation threads",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#limiter = Limiter(key_func=get_remote_address)
# limiter = Limiter(key_func=lambda request: "global")

# app.state.limiter = limiter
# #app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# @app.exception_handler(RateLimitExceeded)
# async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
#     return JSONResponse(
#         status_code=429,
#         content={"detail": "Rate limit exceeded"},
#         headers={
#             "Retry-After": str(settings.RATE_LIMIT_SECONDS) # MUST match your limiter window
#         },
#     )
# app.add_middleware(SlowAPIMiddleware)


import asyncio

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_rag_modules())
    print("🚀 FastAPI started (RAG loading in background)")


#=================================================
# security 
# ===============================================
security = HTTPBearer()



vector_manager = None
create_user_agent = None
retrieve_user_threads = None
get_thread_history_safe = None
get_user_thread_id = None
ingest_document = None
has_documents = None
clear_documents = None
SYSTEM_PROMPT = None
list_documents = None
delete_document = None

# -----------------------------
# 5️⃣ Startup event
# -----------------------------



async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    token_data = verify_token(token, token_type="access")
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify user still exists and is active
    user = user_db.get_user_by_id(token_data.user_id)
    if not user or not user.get("is_active"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return token_data

# ========================================
# Exception Handlers
# ========================================
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    logger.error(f"AppError: {exc.message} (Status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "type": exc.__class__.__name__}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled Exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "type": "UnhandledException"}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

async def load_rag_modules():
    global vector_manager, create_user_agent
    global retrieve_user_threads, get_thread_history_safe, get_user_thread_id
    global ingest_document, has_documents, clear_documents
    global SYSTEM_PROMPT, list_documents, delete_document

    try:
        from .user_rag_temp import (
            create_user_agent as cua,
            retrieve_user_threads as rut,
            get_thread_history_safe as gths,
            get_user_thread_id as guthi,
            vector_manager as vm,
            ingest_document as ing_doc,
            has_documents as hd,
            clear_documents as cd,
            SYSTEM_PROMPT as sp,
            list_documents as ld,
            delete_document as dd
        )

        vector_manager = vm
        create_user_agent = cua
        retrieve_user_threads = rut
        get_thread_history_safe = gths
        get_user_thread_id = guthi
        ingest_document = ing_doc
        has_documents = hd
        clear_documents = cd
        SYSTEM_PROMPT = sp
        list_documents = ld
        delete_document = dd

        print("✅ RAG modules loaded successfully")

    except Exception as e:
        print(f"❌ RAG load failed: {e}")


# ========================================
# Pydantic Models
# ========================================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    thread_id: str = Field(..., min_length=32, max_length=36)

class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    info: Optional[str] = None

class ThreadsResponse(BaseModel):
    threads: List[str]

class MessageItem(BaseModel):
    role: str
    content: str

class HistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageItem]

class NewThreadResponse(BaseModel):
    thread_id: str
    message: str

class DocumentsResponse(BaseModel):
    documents: List[str]
    count: int

class ThreadTitleUpdate(BaseModel):
    title: str
    
class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    is_active: bool
    
    
# ======================================
# auth endpoints 
# =====================================

@app.post("/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate):
    """Register new user"""
    logger.info(f"Registration attempt: {user_data.username}")
    
    try:
        # Validate password strength
        if len(user_data.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )
        
        # Create user
        user = user_db.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        
        # Generate tokens
        access_token = create_access_token(data={"sub": user.user_id, "username": user.username})
        refresh_token = create_refresh_token(data={"sub": user.user_id, "username": user.username})
        
        logger.info(f"User registered successfully: {user.username}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token
        )
    
    except DatabaseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    
    

@app.post("/auth/login", response_model=Token)
def login(credentials: UserLogin):
    """Login user"""
    logger.info(f"Login attempt: {credentials.username}")
    
    user = user_db.authenticate_user(credentials.username, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user["user_id"], "username": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["user_id"], "username": user["username"]})
    
    logger.info(f"User logged in: {credentials.username}")
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )


@app.post("/auth/refresh", response_model=Token)
def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh access token using refresh token"""
    token = credentials.credentials
    token_data = verify_token(token, token_type="refresh")
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Generate new tokens
    access_token = create_access_token(data={"sub": token_data.user_id, "username": token_data.username})
    new_refresh_token = create_refresh_token(data={"sub": token_data.user_id, "username": token_data.username})
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token
    )
    
    
@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """Get current user information"""
    user = user_db.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        user_id=user["user_id"],
        username=user["username"],
        email=user["email"],
        is_active=user["is_active"]
    )






# ========================================
# Health Check
# ========================================
@app.get("/")
def health():
    return {"status": "ok", "service": "chatbot-api"}

@app.get("/metrics")
def metrics(currrent_user : TokenData = Depends(get_current_user)):
    from datetime import datetime
    # Basic metrics
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "threads_active": len(retrieve_user_threads(currrent_user.user_id)),
        "documents_loaded": len(list_documents(currrent_user.user_id)) if has_documents(currrent_user.user_id) else 0,
        "limiter_info": "Rate limiting active"
    }

@app.post("/system/backup")
def trigger_backup():
    """Create a backup of critical data (DBs)"""
    import zipfile
    import datetime
    
    backup_dir = "./backups"
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{backup_dir}/backup_{timestamp}.zip"
    
    try:
        with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Backup SQLite DB
            if os.path.exists(settings.SQLITE_DB_PATH):
                zipf.write(settings.SQLITE_DB_PATH, os.path.basename(settings.SQLITE_DB_PATH))
            
            # Backup ChromaDB (recursive)
            if os.path.exists(settings.CHROMA_DB_PATH):
                for root, dirs, files in os.walk(settings.CHROMA_DB_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(settings.CHROMA_DB_PATH))
                        zipf.write(file_path, arcname)
                        
        logger.info(f"Backup created at {backup_file}")
        return {"status": "success", "backup_file": backup_file}
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")

# ========================================
# Thread Management
# ========================================
@app.post("/threads/new", response_model=NewThreadResponse)
def create_new_thread(current_user: TokenData = Depends(get_current_user)):
    new_thread_id = str(uuid.uuid4())
    logger.info(f"New thread created for user {current_user.user_id}: {new_thread_id}")

    return NewThreadResponse(thread_id=new_thread_id, message="New thread created successfully")

@app.get("/threads", response_model=ThreadsResponse)
def get_all_threads(current_user: TokenData = Depends(get_current_user)):
    try:
        all_threads = retrieve_user_threads(current_user.user_id)
        return ThreadsResponse(threads=all_threads)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/exists")
def check_thread_exists(thread_id: str,current_user: TokenData = Depends(get_current_user)):
    existing_threads =  retrieve_user_threads(current_user.user_id)
    return {"exists": thread_id in existing_threads, "thread_id": thread_id}

@app.post("/threads/{thread_id}/title")
def update_thread_title(thread_id: str, data: ThreadTitleUpdate,current_user : TokenData = Depends(get_current_user)):
    try:
        agent=create_user_agent(current_user.user_id)
        namespaced_thread_id = get_user_thread_id(current_user.user_id, thread_id)
        state = agent.get_state({"configurable": {"thread_id": namespaced_thread_id}})
        #state = agent.get_state({"configurable": {"thread_id": thread_id}})
        if not state.values:
            state.values = {}
        state.values["title"] = data.title
        agent.set_state({"configurable": {"thread_id":  namespaced_thread_id}}, state)
        return {"success": True, "title": data.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")

# ========================================
# Chat Endpoint
# ========================================
# @app.post("/chat", response_model=ChatResponse)
# @limiter.limit(settings.MAX_REQUESTS_PER_MINUTE)
# def chat(request: Request,req: ChatRequest):
#     logger.info(f"Chat request for thread {req.thread_id}")
#     config = {"configurable": {"thread_id": req.thread_id}}

#     try:
#         existing_messages = get_thread_history_safe(req.thread_id)
#         if not existing_messages:
#             messages = [
#                 SystemMessage(content=SYSTEM_PROMPT),
#                 HumanMessage(content=req.message)
#             ]
#         else:
#             messages = [HumanMessage(content=req.message)]

#         result = chatmodel.invoke({"messages": messages}, config=config)

#         # Normalize AI response
#         def normalize_message_content(content):
#             if isinstance(content, str):
#                 return content
#             elif isinstance(content, list):
#                 return " ".join(item.get("text", "") for item in content if isinstance(item, dict) and "text" in item)
#             else:
#                 return str(content)

#         ai_reply = normalize_message_content(result["messages"][-1].content)

#         # Lock citations
#         if "SOURCES (DO NOT MODIFY):" in ai_reply or "DOCUMENT_ANSWER (DO NOT MODIFY):" in ai_reply:
#             return ChatResponse(reply=ai_reply, thread_id=req.thread_id, info="RAG response with citations")

#         info = "RAG-enabled" if has_documents() else "Direct response"
#         return ChatResponse(reply=ai_reply, thread_id=req.thread_id, info=info)
#     except Exception as e:
#         logger.error(f"Chat processing failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# @app.post("/chat", response_model=ChatResponse)
# #@limiter.limit(settings.MAX_REQUESTS_PER_MINUTE)
# def chat(request: Request, req: ChatRequest, current_user: TokenData = Depends(get_current_user)):
#     user_db.check_and_increment_daily_limit(current_user.user_id)
#     logger.info(f"Chat request from user {current_user.user_id}")
#     logger.info(f"Chat request from user {current_user.user_id}, thread {req.thread_id}")
#     namespaced_thread_id = get_user_thread_id(current_user.user_id, req.thread_id)
#     config = {"configurable": {"thread_id": namespaced_thread_id}}
    
#     try:
#         # -------------------------------
#         # Thread / Memory handling
#         # -------------------------------
#         agent = create_user_agent(current_user.user_id)

#         existing_messages = get_thread_history_safe(current_user.user_id,req.thread_id)

#         messages = []
#         if not existing_messages:
#             messages.append(SystemMessage(content=SYSTEM_PROMPT))

#         messages.append(HumanMessage(content=req.message))

#         # -------------------------------
#         # Explicit RAG retrieval
#         # -------------------------------
#         retrieved_docs = []
#         if has_documents(current_user.user_id):
#             try:
#                 retriever = vector_manager.get_retriever(current_user.user_id)

#                 retrieved_docs = retriever.invoke(req.message)
#             except Exception as e:
#                 logger.error("Vector store retrieval failed", exc_info=True)
#                 retrieved_docs = []

#         # -------------------------------
#         # Inject retrieved context (TRUE RAG)
#         # -------------------------------
#         if retrieved_docs:
#             context_text = "\n\n".join(
#                 f"[DOC {i+1}]\n{doc.page_content}"
#                 for i, doc in enumerate(retrieved_docs)
#             )

#             messages.insert(
#                 0,
#                 SystemMessage(
#                     content=(
#                         f"{SYSTEM_PROMPT}\n\n"
#                         "Use the following documents to answer. "
#                         "If the answer is not present, say you don't know.\n\n"
#                         f"{context_text}"
#                     )
#                 )
#             )
#         else:
#             logger.warning("RAG returned 0 documents — LLM-only response")

#         # -------------------------------
#         # Model invocation
#         # -------------------------------
#         result = agent.invoke({"messages": messages}, config=config)

#         # -------------------------------
#         # Normalize model output
#         # -------------------------------
#         def normalize_message_content(content):
#             if isinstance(content, str):
#                 return content
#             elif isinstance(content, list):
#                 return " ".join(
#                     item.get("text", "")
#                     for item in content
#                     if isinstance(item, dict) and "text" in item
#                 )
#             return str(content)

#         ai_reply = normalize_message_content(result["messages"][-1].content)

#         # -------------------------------
#         # Citation lock (if present)
#         # -------------------------------
#         if (
#             "SOURCES (DO NOT MODIFY):" in ai_reply
#             or "DOCUMENT_ANSWER (DO NOT MODIFY):" in ai_reply
#         ):
#             return ChatResponse(
#                 reply=ai_reply,
#                 thread_id=req.thread_id,
#                 info="RAG response with citations"
#             )

#         info = "RAG-enabled" if retrieved_docs else "Direct response"
#         return ChatResponse(
#             reply=ai_reply,
#             thread_id=req.thread_id,
#             info=info
#         )

#     except Exception as e:
#         logger.error("Chat processing failed", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Chat failed: {str(e)}"
#         )

@app.post("/chat", response_model=ChatResponse)
# @limiter.limit(settings.MAX_REQUESTS_PER_MINUTE)
def chat(
    request: Request,
    req: ChatRequest,
    current_user: TokenData = Depends(get_current_user)
):
    # ---------------------------------------
    # Per-user daily limit (your DB limit)
    # ---------------------------------------
    user_db.check_and_increment_daily_limit(current_user.user_id)

    logger.info(f"Chat request from user {current_user.user_id}, thread {req.thread_id}")

    namespaced_thread_id = get_user_thread_id(current_user.user_id, req.thread_id)
    config = {"configurable": {"thread_id": namespaced_thread_id}}

    try:
        # ---------------------------------------
        # Thread / Memory handling
        # ---------------------------------------
        agent = create_user_agent(current_user.user_id)
        existing_messages = get_thread_history_safe(
            current_user.user_id,
            req.thread_id
        )

        messages = []

        if not existing_messages:
            messages.append(SystemMessage(content=SYSTEM_PROMPT))

        messages.append(HumanMessage(content=req.message))

        # ---------------------------------------
        # Explicit RAG retrieval
        # ---------------------------------------
        retrieved_docs = []

        if has_documents(current_user.user_id):
            try:
                retriever = vector_manager.get_retriever(current_user.user_id)
                retrieved_docs = retriever.invoke(req.message)
            except Exception:
                logger.error("Vector store retrieval failed", exc_info=True)
                retrieved_docs = []

        # ---------------------------------------
        # Inject retrieved context (TRUE RAG)
        # ---------------------------------------
        if retrieved_docs:
            context_text = "\n\n".join(
                f"[DOC {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(retrieved_docs)
            )

            messages.insert(
                0,
                SystemMessage(
                    content=(
                        f"{SYSTEM_PROMPT}\n\n"
                        "Use the following documents to answer. "
                        "If the answer is not present, say you don't know.\n\n"
                        f"{context_text}"
                    )
                )
            )
        else:
            logger.warning("RAG returned 0 documents — LLM-only response")

        # ---------------------------------------
        # Model invocation (Gemini quota-safe)
        # ---------------------------------------
        try:
            result = agent.invoke({"messages": messages}, config=config)

        except Exception as model_error:
            error_str = str(model_error)

            # ✅ Catch Gemini daily quota exceeded
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                logger.warning("Gemini daily quota exceeded")

                raise HTTPException(
                    status_code=503,
                    detail=(
                        "AI service temporarily unavailable "
                        "(daily quota reached). Please try again tomorrow."
                    )
                )

            logger.error("Model invocation failed", exc_info=True)

            raise HTTPException(
                status_code=500,
                detail="AI service error."
            )

        # ---------------------------------------
        # Normalize model output
        # ---------------------------------------
        def normalize_message_content(content):
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and "text" in item
                )
            return str(content)

        ai_reply = normalize_message_content(
            result["messages"][-1].content
        )

        # ---------------------------------------
        # Citation lock (if present)
        # ---------------------------------------
        if (
            "SOURCES (DO NOT MODIFY):" in ai_reply
            or "DOCUMENT_ANSWER (DO NOT MODIFY):" in ai_reply
        ):
            return ChatResponse(
                reply=ai_reply,
                thread_id=req.thread_id,
                info="RAG response with citations"
            )

        info = "RAG-enabled" if retrieved_docs else "Direct response"

        return ChatResponse(
            reply=ai_reply,
            thread_id=req.thread_id,
            info=info
        )

    except HTTPException:
        # Let handled HTTP errors pass through cleanly
        raise

    except Exception as e:
        logger.error("Chat processing failed", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail="Chat failed due to internal error."
        )




# ========================================
# Thread History
# ========================================
# @app.get("/threads/{thread_id}/history", response_model=HistoryResponse)
# def get_thread_history(thread_id: str,current_user: TokenData = Depends(get_current_user)):
    
#     existing_threads = retrieve_user_threads(current_user.user_id)
#     if thread_id not in existing_threads:
#         raise HTTPException(status_code=404, detail="Thread not found")
#     try:
#         namespaced_thread_id = get_user_thread_id(current_user.user_id, thread_id)
#         config = {"configurable": {"thread_id": namespaced_thread_id}}
#        # config = {"configurable": {"thread_id": thread_id}}
#         agent=create_user_agent(current_user.user_id)
#         state = agent.get_state(config)
#         messages = state.values.get("messages", [])

#         def normalize_message_content(content):
#             if isinstance(content, str):
#                 return content
#             elif isinstance(content, list):
#                 return " ".join(item.get("text", "") for item in content if isinstance(item, dict) and "text" in item)
#             else:
#                 return str(content)

#         history = []
#         for msg in messages:
#             if isinstance(msg, (SystemMessage, ToolMessage)):
#                 continue
#             if not hasattr(msg, "content") or not msg.content:
#                 continue

#             if isinstance(msg, HumanMessage):
#                 role = "user"
#             elif isinstance(msg, AIMessage):
#                 role = "assistant"
#             else:
#                 continue

#             content = normalize_message_content(msg.content)
#             if not content.strip():
#                 continue
#             history.append(MessageItem(role=role, content=content))

#         logger.info(f"Loaded {len(history)} messages for thread {thread_id}")
#         return HistoryResponse(thread_id=thread_id, messages=history)
#     except Exception as e:
#         logger.error(f"Failed to load history for {thread_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/history", response_model=HistoryResponse)
def get_thread_history(thread_id: str, current_user: TokenData = Depends(get_current_user)):

    try:
        namespaced_thread_id = get_user_thread_id(current_user.user_id, thread_id)
        config = {"configurable": {"thread_id": namespaced_thread_id}}

        agent = create_user_agent(current_user.user_id)
        state = agent.get_state(config)
        messages = state.values.get("messages", [])

        def normalize_message_content(content):
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and "text" in item
                )
            else:
                return str(content)

        history = []

        for msg in messages:
            if isinstance(msg, (SystemMessage, ToolMessage)):
                continue
            if not hasattr(msg, "content") or not msg.content:
                continue

            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue

            content = normalize_message_content(msg.content)
            if not content.strip():
                continue

            history.append(MessageItem(role=role, content=content))

        logger.info(f"Loaded {len(history)} messages for thread {thread_id}")

        return HistoryResponse(
            thread_id=thread_id,
            messages=history  # ✅ will return empty list if no messages
        )

    except Exception as e:
        logger.error(f"Failed to load history for {thread_id}: {e}")
        raise HTTPException(status_code=404, detail="Thread not found")


# ========================================
# Document Management
# ========================================
@app.get("/documents", response_model=DocumentsResponse)
def get_documents(current_user: TokenData = Depends(get_current_user)):
    try:
        docs = list_documents(current_user.user_id)
        return DocumentsResponse(documents=docs, count=len(docs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/documents/upload")
# async def upload_document(file: UploadFile = File(...),current_user: TokenData = Depends(get_current_user)):
#     logger.info(f"User {current_user.user_id} uploading: {file.filename}")

#     if not file.filename.endswith(('.pdf', '.txt')):
#         raise HTTPException(status_code=400, detail="⚠️ Unsupported file type. Please upload a PDF or TXT file.")

    
#     user_upload_dir = f"{settings.UPLOAD_DIR}/{current_user.user_id}"

#     os.makedirs(user_upload_dir, exist_ok=True)
#     temp_path = f"{user_upload_dir}/{file.filename}"

#     try:
#         file.file.seek(0, 2)
#         file_size = file.file.tell()
#         file.file.seek(0)
#         if file_size == 0:
#             raise HTTPException(status_code=400, detail="⚠️ The uploaded file is empty.")
#         if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
#             raise HTTPException(status_code=413, detail=f"⚠️ File too large (Max {settings.MAX_FILE_SIZE_MB}MB).")

#         with open(temp_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         return ingest_document(current_user.user_id, temp_path)
#     except DocumentProcessingError as e:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         msg = str(e).lower()
#         if "empty" in msg or "chunk" in msg or "extract" in msg:
#             raise HTTPException(status_code=422, detail="⚠️ The PDF contains no readable text.")
#         raise HTTPException(status_code=422, detail="⚠️ Document processing failed.")
#     except HTTPException:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         raise
#     except Exception:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         logger.exception("Unexpected upload failure")
#         raise HTTPException(status_code=500, detail="Internal server error during upload.")

def validate_pdf_content(file_path: str):
    """
    Validate that the PDF:
    - Is readable
    - Contains extractable text
    """

    try:
        reader = PdfReader(file_path)

        if len(reader.pages) == 0:
            raise HTTPException(
                status_code=422,
                detail="Invalid PDF: No pages found"
            )

        extracted_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Invalid PDF: No readable text found"
            )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid or corrupted PDF file"
        )


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user)
):
    logger.info(f"User {current_user.user_id} uploading: {file.filename}")

    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400,
            detail="⚠️ Unsupported file type. Please upload a PDF or TXT file."
        )

    user_upload_dir = f"{settings.UPLOAD_DIR}/{current_user.user_id}"
    os.makedirs(user_upload_dir, exist_ok=True)

    temp_path = f"{user_upload_dir}/{file.filename}"

    try:
        # Validate size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="⚠️ The uploaded file is empty.")

        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"⚠️ File too large (Max {settings.MAX_FILE_SIZE_MB}MB)."
            )

        # Save file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        if file.filename.endswith(".pdf"):
            validate_pdf_content(temp_path)

        # ✅ Create upload job in users.db
        job_id = user_db.create_upload_job(
            current_user.user_id,
            file.filename
        )

        # ✅ Start background ingestion
        background_tasks.add_task(
            process_document_background,
            current_user.user_id,
            temp_path,
            job_id
        )

        # ✅ Return immediately (NO BLOCKING)
        return {
            "job_id": job_id,
            "status": "processing"
          #  "message": "File uploaded successfully. Processing started."
        }

    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.exception("Unexpected upload failure")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during upload."
        )
        
def process_document_background(user_id: str, file_path: str, job_id: str):
    try:
        result = ingest_document(user_id, file_path)

        user_db.update_upload_job(
            job_id,
            "done",
            result["message"]
        )

    except Exception as e:
        user_db.update_upload_job(
            job_id,
            "failed",
            str(e)
        )
    finally:
        # Optional cleanup on failure
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/documents/upload-status/{job_id}")
def get_upload_status(
    job_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    job = user_db.get_upload_job(
        current_user.user_id,
        job_id
    )

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job




# @app.delete("/documents/{filename}")
# def delete_doc(filename: str,current_user: TokenData = Depends(get_current_user)
# ):
#     logger.info(f"User {current_user.user_id} deleting: {filename}")
#     result = delete_document(current_user.user_id,filename)
#     file_path = f"{settings.UPLOAD_DIR}/{current_user.user_id}/{filename}"
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         logger.info(f"Deleted physical file: {file_path}")
#     return result

# @app.delete("/documents")
# def clear_all_documents(current_user: TokenData =Depends( get_current_user)):
#     logger.info(f"User {current_user.user_id} clearing all documents")
#     result = clear_documents(current_user.user_id)
#     user_upload_dir = f"{settings.UPLOAD_DIR}/{current_user.user_id}"
#     try:
#         if os.path.exists(user_upload_dir):
#             shutil.rmtree(user_upload_dir)
#         os.makedirs(user_upload_dir, exist_ok=True)
#         logger.info("Cleared uploads directory")
#     except Exception as e:
#         logger.warning(f"Non-critical cleanup error: {e}")
#     return result

@app.delete("/documents/{filename}")
def delete_doc(
    filename: str,
    current_user: TokenData = Depends(get_current_user)
):
    logger.info(f"User {current_user.user_id} deleting: {filename}")

    # 1Delete from vector store
    result = delete_document(current_user.user_id, filename)

    # Delete physical file
    file_path = f"{settings.UPLOAD_DIR}/{current_user.user_id}/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"Deleted physical file: {file_path}")

    # 3️ UPDATE JOB STATUS
    user_db.mark_job_deleted(current_user.user_id, filename)

    return result

@app.delete("/documents")
def clear_all_documents(
    current_user: TokenData = Depends(get_current_user)
):
    logger.info(f"User {current_user.user_id} clearing all documents")

    # 1️⃣ Delete vector store
    result = clear_documents(current_user.user_id)

    # 2️⃣ Delete physical files
    user_upload_dir = f"{settings.UPLOAD_DIR}/{current_user.user_id}"
    try:
        if os.path.exists(user_upload_dir):
            shutil.rmtree(user_upload_dir)
        os.makedirs(user_upload_dir, exist_ok=True)
        logger.info("Cleared uploads directory")
    except Exception as e:
        logger.warning(f"Non-critical cleanup error: {e}")

    # 3️⃣ 🔥 MARK ALL JOBS AS DELETED
    user_db.mark_all_jobs_deleted(current_user.user_id)

    return result
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # 8000 for local, Render overrides
    uvicorn.run(
        "backend.user_main2:app",
        host="0.0.0.0",
        port=port,
    )


