from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import sys

from app.config import settings
from app.database import init_db
from app.api import api_router
from app.middleware.logging import LoggingMiddleware
from app.ml.model_loader import get_model_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events

    Startup:
    - Initialize database
    - Load ML models

    Shutdown:
    - Cleanup resources
    """
    # Startup
    logger.info("Starting Insurance Fraud Detection System...")

    try:
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        logger.info("✓ Database initialized")

        # Load ML models
        logger.info("Loading ML models...")
        model_loader = get_model_loader()
        logger.info("✓ ML models loaded successfully")

        logger.info("=" * 70)
        logger.info("APPLICATION STARTED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
        logger.info(f"API Version: {settings.VERSION}")
        logger.info(f"Documentation: http://localhost:8000/docs")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Insurance Fraud Detection System...")
    logger.info("✓ Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Risk-Based Authentication and Fraud Prevention System for Online Insurance Transactions",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS Configuration
origins = settings.ALLOWED_ORIGINS.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)

# Include API routers
app.include_router(api_router, prefix="/api/v1")


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        JSON response with error details
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle all unhandled exceptions

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Health Check Endpoint
@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint

    Returns service status and version information
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": "development" if settings.DEBUG else "production"
    }


# Root Endpoint
@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint

    Returns API information and links
    """
    return {
        "message": "Insurance Fraud Detection System API",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "authentication": "/api/v1/auth",
            "claims": "/api/v1/claims",
            "risk_assessment": "/api/v1/risk",
            "fraud_detection": "/api/v1/fraud",
            "mfa": "/api/v1/mfa",
            "admin": "/api/v1/admin"
        }
    }


# API Info Endpoint
@app.get("/api/v1", tags=["API Info"])
def api_info():
    """
    API version information

    Returns API version and available endpoints
    """
    return {
        "version": "v1",
        "endpoints": [
            {
                "path": "/api/v1/auth",
                "description": "Authentication and user management",
                "methods": ["POST"]
            },
            {
                "path": "/api/v1/claims",
                "description": "Insurance claim submission and management",
                "methods": ["GET", "POST"]
            },
            {
                "path": "/api/v1/risk",
                "description": "Risk assessment and scoring",
                "methods": ["GET"]
            },
            {
                "path": "/api/v1/fraud",
                "description": "Fraud detection and analysis",
                "methods": ["GET"]
            },
            {
                "path": "/api/v1/mfa",
                "description": "Multi-factor authentication",
                "methods": ["POST"]
            },
            {
                "path": "/api/v1/admin",
                "description": "Administrative functions and analytics",
                "methods": ["GET"]
            }
        ]
    }


# Development/Debug endpoints (only in DEBUG mode)
if settings.DEBUG:
    @app.get("/debug/config", tags=["Debug"])
    def debug_config():
        """
        Debug endpoint to view configuration (DEBUG mode only)

        Returns non-sensitive configuration values
        """
        return {
            "app_name": settings.APP_NAME,
            "debug": settings.DEBUG,
            "version": settings.VERSION,
            "database_url": settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else "***",
            "models_path": settings.MODELS_PATH,
            "allowed_origins": settings.ALLOWED_ORIGINS,
            "risk_thresholds": {
                "low": settings.RISK_LOW_THRESHOLD,
                "high": settings.RISK_HIGH_THRESHOLD
            },
            "fraud_threshold": settings.FRAUD_PROBABILITY_THRESHOLD
        }


    @app.get("/debug/models", tags=["Debug"])
    def debug_models():
        """
        Debug endpoint to check loaded models (DEBUG mode only)

        Returns information about loaded ML models
        """
        try:
            model_loader = get_model_loader()

            return {
                "status": "Models loaded successfully",
                "models": {
                    "risk_model": model_loader.get_risk_model() is not None,
                    "fraud_classifier": model_loader.get_fraud_classifier() is not None,
                    "isolation_forest": model_loader.get_isolation_forest() is not None,
                    "risk_explainer": model_loader.get_risk_explainer() is not None,
                    "fraud_explainer": model_loader.get_fraud_explainer() is not None
                },
                "features": {
                    "risk_features_count": len(model_loader.get_risk_features() or []),
                    "fraud_features_count": len(model_loader.get_fraud_features() or [])
                },
                "metadata": {
                    "risk_model": model_loader.get_risk_metadata(),
                    "fraud_model": model_loader.get_fraud_metadata()
                }
            }
        except Exception as e:
            return {
                "status": "Error loading models",
                "error": str(e)
            }

# Run application (for development)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning"
    )
