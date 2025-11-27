import datetime as dt
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
import uuid

from app.models.user import User
from app.models.authentication_event import AuthenticationEvent
from app.config import settings
from app.core.constants import AuthEventType

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Authentication service for user management and JWT tokens"""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token

        Args:
            data: Data to encode in token
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(dt.UTC) + expires_delta
        else:
            expire = datetime.now(dt.UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "type": "access"})

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

        return encoded_jwt

    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """
        Create JWT refresh token

        Args:
            data: Data to encode in token

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        expire = datetime.now(dt.UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({"exp": expire, "type": "refresh"})

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token to verify

        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            return payload
        except JWTError as e:
            logger.error(f"Token verification failed: {str(e)}")
            return None

    @staticmethod
    def register_user(
        user_data: dict,
        db: Session
    ) -> Tuple[User, str, str]:
        """
        Register a new user

        Args:
            user_data: User registration data
            db: Database session

        Returns:
            Tuple of (user, access_token, refresh_token)
        """
        # Check if user exists
        existing_user = db.query(User).filter(User.email == user_data['email']).first()
        if existing_user:
            raise ValueError("Email already registered")

        # Hash password
        hashed_password = AuthService.hash_password(user_data['password'])

        # Generate policy number
        policy_number = f"POL-{uuid.uuid4().hex[:10].upper()}"

        # Create user
        user = User(
            email=user_data['email'],
            password_hash=hashed_password,
            first_name=user_data['first_name'],
            last_name=user_data['last_name'],
            phone_number=user_data.get('phone_number'),
            city=user_data.get('city'),
            state=user_data.get('state'),
            country=user_data.get('country', 'USA'),
            policy_number=policy_number,
            policy_type=user_data.get('policy_type'),
            is_verified=True,  # Auto-verify for demo
            account_status='active'
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        # Create tokens
        access_token = AuthService.create_access_token(
            data={"sub": str(user.user_id), "email": user.email}
        )
        refresh_token = AuthService.create_refresh_token(
            data={"sub": str(user.user_id)}
        )

        # Log authentication event
        auth_event = AuthenticationEvent(
            user_id=user.user_id,
            event_type=AuthEventType.LOGIN.value,
            auth_method="password",
            auth_result="success"
        )
        db.add(auth_event)
        db.commit()

        logger.info(f"User registered: {user.email}")

        return user, access_token, refresh_token

    @staticmethod
    def authenticate_user(
            email: str,
            password: str,
            db: Session,
            ip_address: Optional[str] = None
    ) -> Tuple[Optional[User], Optional[str], Optional[str]]:
        """
        Authenticate user with email and password

        Args:
            email: User email
            password: User password
            db: Database session
            ip_address: Client IP address

        Returns:
            Tuple of (user, access_token, refresh_token) or (None, None, None) if failed
        """
        # Find user
        user = db.query(User).filter(User.email == email).first()

        if not user:
            logger.warning(f"Login attempt for non-existent user: {email}")
            return None, None, None

        # Verify password
        if not AuthService.verify_password(password, user.password_hash):
            # Log failed attempt
            auth_event = AuthenticationEvent(
                user_id=user.user_id,
                event_type=AuthEventType.LOGIN.value,
                auth_method="password",
                auth_result="failure",
                ip_address=ip_address
            )
            db.add(auth_event)
            db.commit()

            logger.warning(f"Failed login attempt for user: {email}")
            return None, None, None

        # Check account status
        if user.account_status != 'active':
            logger.warning(f"Login attempt for inactive account: {email}")
            return None, None, None

        # Update last login
        user.last_login_at = datetime.now(dt.UTC)

        # Create tokens
        access_token = AuthService.create_access_token(
            data={"sub": str(user.user_id), "email": user.email}
        )
        refresh_token = AuthService.create_refresh_token(
            data={"sub": str(user.user_id)}
        )

        # Log successful login
        auth_event = AuthenticationEvent(
            user_id=user.user_id,
            event_type=AuthEventType.LOGIN.value,
            auth_method="password",
            auth_result="success",
            ip_address=ip_address
        )
        db.add(auth_event)
        db.commit()

        logger.info(f"User authenticated: {email}")

        return user, access_token, refresh_token

    @staticmethod
    def get_user_from_token(token: str, db: Session) -> Optional[User]:
        """
        Get user from JWT token

        Args:
            token: JWT token
            db: Database session

        Returns:
            User object or None if token invalid
        """
        payload = AuthService.verify_token(token)

        if not payload:
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        user = db.query(User).filter(User.user_id == user_id).first()
        return user
