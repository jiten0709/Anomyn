import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# import our compliance api router
from app.api.routes import router as compliance_router

# logging setup
from app.utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="main.log")

# initialize the FastAPI application with metadata for the swagger ui
app = FastAPI(
    title="Anomyn",
    description="An AI agent responsible for validating collected operational data before regulatory reporting.",
    version="1.0",
    contact={
        "name": "Jiten Parmar",
    }
)

# --- middleware configuration ---
# in production, restrict origins to your specific frontend domains
origins = [
    "http://localhost",
    "http://localhost:3000", # common react/vue port
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # allow all http methods (get, post, etc.)
    allow_headers=["*"], # allow all headers
)

# --- router mounting ---
# mount the compliance routes under the /api/v1 prefix
app.include_router(compliance_router)

# --- global endpoints ---

@app.get("/", tags=["System"])
async def root():
    """Root endpoint to verify the API is reachable."""
    return {"message": "Anomym API is running. Visit /docs for documentation."}

@app.get("/health", tags=["System"])
async def health_check():
    """
    DevOps Health Check Endpoint. 
    Used by Kubernetes or Load Balancers to verify the service is healthy.
    """
    # in a full production system, we would also ping the database here to ensure the connection pool is alive.
    return JSONResponse(status_code=200, content={"status": "healthy"})

# --- server execution ---
if __name__ == "__main__":
    # run the server using uvicorn
    # reload=True automatically restarts the server when you make code changes
    logger.info("🚀 Anomym API is running at http://localhost:8000. Press CTRL+C to stop.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    