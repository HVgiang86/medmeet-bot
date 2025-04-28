from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from agent.rag import rag

from backend.db.mongodb import MongoDB
from backend.api.chat import router as chat_router
from backend.models.responses import BaseResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI backend
app = FastAPI(
    title="AI Chat API",
    description="API for managing AI chat conversations",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=BaseResponse(
            statusCode=exc.status_code,
            message=exc.detail,
            data=None
        ).model_dump(),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=BaseResponse(
            statusCode=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            data=None
        ).model_dump(),
    )

@app.on_event("startup")
async def startup_db_client():
    await MongoDB.connect_to_mongodb()

@app.on_event("shutdown")
async def shutdown_db_client():
    await MongoDB.close_mongodb_connection()

response = rag.invoke({
    "input": "Xin chao",
    "chat_history": [],  # Exclude the current message
})

answer = response["answer"]
print(answer)

@app.get("/", response_model=BaseResponse)
async def root():
    return BaseResponse(
        statusCode=status.HTTP_200_OK,
        message="AI Chat API is running",
        data=None
    ) 