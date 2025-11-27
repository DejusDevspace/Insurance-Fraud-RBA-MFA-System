from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses
    """

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Log request and response details

        Args:
            request: FastAPI request
            call_next: Next middleware/route handler

        Returns:
            Response
        """
        # Start timer
        start_time = time.time()

        # Get request details
        method = request.method
        url = str(request.url)
        client_host = request.client.host if request.client else "unknown"

        # Log request
        logger.info(f"→ {method} {url} from {client_host}")

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"← {method} {url} - Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )

            # Add custom headers
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"✗ {method} {url} - Error: {str(e)} - "
                f"Time: {process_time:.3f}s"
            )
            raise
