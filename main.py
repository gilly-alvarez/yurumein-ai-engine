import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.translater.translater_router import router as translater_router
#from app.services.text_to_speech.text_to_speech_router import router as text_to_speech_router
from app.services.learning_hub.learning_hub_router import router as learning_hub_router
from app.services.chatbot.cahtbot_router import router as chatbot_router
from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager
from app.services.utils.RAG_pipeline.DB.session_manager import SessionManager

store = VectorStoreManager()
DBManager = SessionManager()
app = FastAPI(title="Yurumein Al",version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(translater_router,tags=["Translation"])
app.include_router(learning_hub_router,tags=['learning hub'])
app.include_router(chatbot_router ,tags=['chatbot'])
#app.include_router(text_to_speech_router ,tags=["text_to_speech"])

@app.get("/",tags = ["health"])
async def root():
    return """
"message": "Welcome to the gaupo AI!",
        "status": "healthy",
        "version": "1.0.0"
"""

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker/monitoring"""
    return {
        "status": "healthy",
        "service": "Study Buddy AI"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup"""
    try:
        #await store.upload_docs_pipeline()
        await DBManager.create_tables()
        print("Vector store initialized on startup")
    except Exception as e:
        print(f"Warning: Vector store initialization failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8035, 
        reload=True
    )

