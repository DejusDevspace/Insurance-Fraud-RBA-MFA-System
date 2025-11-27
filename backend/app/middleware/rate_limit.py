from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Check rate limit and process request

        Args:
            request: FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response
        """
        # Get client identifier (IP address)
        client_host = request.client.host if request.client else "unknown"

        # Skip rate limiting for certain paths
        exempt_paths = ["/docs", "/redoc", "/openapi.json", "/health"]
        if any(request.url.path.startswith(path) for path in exempt_paths):
            return await call_next(request)

        # Get current time
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.request_counts[client_host] = [
            req_time for req_time in self.request_counts[client_host]
            if req_time > minute_ago
        ]

        # Check rate limit
        if len(self.request_counts[client_host]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_host}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )

        # Add current request
        self.request_counts[client_host].append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.request_counts[client_host])
        )

        return response
