import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import our compliance API router
from app.api.routes import router as compliance_router

# Configure global logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("compliance_agent")

# Initialize the FastAPI application with metadata for the Swagger UI
app = FastAPI(
    title="AI Compliance Monitoring Agent",
    description="An AI agent responsible for validating collected operational data before regulatory reporting.",
    version="1.0.0",
    contact={
        "name": "Compliance Engineering Team",
    }
)

# --- Middleware Configuration ---
# In production, restrict origins to your specific frontend domains
origins = [
    "http://localhost",
    "http://localhost:3000", # Common React/Vue port
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Router Mounting ---
# Mount the compliance routes under the /api/v1 prefix
app.include_router(compliance_router)

# --- Global Endpoints ---

@app.get("/", tags=["System"])
async def root():
    """Root endpoint to verify the API is reachable."""
    return {"message": "AI Compliance Monitoring Agent API is running. Visit /docs for documentation."}

@app.get("/health", tags=["System"])
async def health_check():
    """
    DevOps Health Check Endpoint. 
    Used by Kubernetes or Load Balancers to verify the service is healthy.
    """
    # In a full production system, you would also ping the Database here
    # to ensure the connection pool is alive.
    return JSONResponse(status_code=200, content={"status": "healthy"})

# --- Server Execution ---
if __name__ == "__main__":
    logger.info("Starting AI Compliance Monitoring Agent...")
    # Run the server using Uvicorn
    # reload=True automatically restarts the server when you make code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    