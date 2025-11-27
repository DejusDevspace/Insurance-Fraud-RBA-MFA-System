from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.auth import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse
)
from app.schemas.user import UserResponse
from app.services.auth_service import AuthService
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db)
):
    """
    Register a new user

    - **email**: User email (must be unique)
    - **password**: Password (min 8 characters, must include uppercase, lowercase, digit)
    - **first_name**: First name
    - **last_name**: Last name
    - **phone_number**: Phone number (optional)
    - **city**: City (optional)
    - **state**: State (optional)
    - **country**: Country (default: USA)
    - **policy_type**: Insurance policy type (optional)
    """
    try:
        # Convert request to dictionary
        user_data = request.dict()

        # Register user
        user, access_token, refresh_token = AuthService.register_user(
            user_data=user_data,
            db=db
        )

        # Convert user to response schema
        user_response = UserResponse.from_orm(user)

        return RegisterResponse(
            message="User registered successfully",
            user=user_response.dict(),
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and get access token

    - **email**: User email
    - **password**: User password

    Returns JWT access token and refresh token
    """
    # Get client IP
    ip_address = req.client.host if req.client else None

    # Authenticate user
    user, access_token, refresh_token = AuthService.authenticate_user(
        email=request.email,
        password=request.password,
        db=db,
        ip_address=ip_address
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Convert user to response schema
    user_response = UserResponse.from_orm(user)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=user_response.dict()
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information

    Requires valid JWT token in Authorization header
    """
    return UserResponse.from_orm(current_user)


@router.post("/logout")
def logout(
    current_user: User = Depends(get_current_user)
):
    """
    Logout current user

    Note: With JWT, logout is handled client-side by removing the token.
    This endpoint is provided for consistency and can be extended to
    implement token blacklisting if needed.
    """
    return {
        "message": "Logout successful",
        "user_id": str(current_user.user_id)
    }
