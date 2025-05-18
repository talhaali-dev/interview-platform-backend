from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.config import get_settings
from api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the ML model
    settings = get_settings()
    # You could initialize any resources here
    yield
    # Shutdown: Clean up resources
    pass

app = FastAPI(
    title=get_settings().APP_NAME,
    description="AI-powered interview preparation API",
    version="1.0.0",
    lifespan=lifespan
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
app.include_router(router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    return {"message": "Interview Prep API is running"} 