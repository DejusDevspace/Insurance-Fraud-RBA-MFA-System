from fastapi import APIRouter
from app.api import auth, claims, risk, fraud, mfa, admin

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(claims.router, prefix="/claims", tags=["Claims"])
api_router.include_router(risk.router, prefix="/risk", tags=["Risk Assessment"])
api_router.include_router(fraud.router, prefix="/fraud", tags=["Fraud Detection"])
api_router.include_router(mfa.router, prefix="/mfa", tags=["Multi-Factor Authentication"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin Dashboard"])
