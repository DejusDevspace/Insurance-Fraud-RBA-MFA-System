from typing import Optional, Any
from datetime import datetime, date
from decimal import Decimal
from fastapi import Request


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format amount as currency

    Args:
        amount: Amount to format
        currency: Currency code (default: USD)

    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def calculate_percentage(part: float, whole: float, decimals: int = 2) -> float:
    """
    Calculate percentage

    Args:
        part: Part value
        whole: Whole value
        decimals: Number of decimal places

    Returns:
        Percentage value
    """
    if whole == 0:
        return 0.0

    percentage = (part / whole) * 100
    return round(percentage, decimals)


def get_client_ip(request: Request) -> Optional[str]:
    """
    Get client IP address from request
    Handles proxy headers (X-Forwarded-For, X-Real-IP)

    Args:
        request: FastAPI request object

    Returns:
        Client IP address or None
    """
    # Check for forwarded IP (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct client IP
    if request.client:
        return request.client.host

    return None


def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Convert datetime to ISO format string

    Args:
        dt: Datetime object

    Returns:
        ISO format string or None
    """
    if dt is None:
        return None

    return dt.isoformat()


def date_to_iso(d: Optional[date]) -> Optional[str]:
    """
    Convert date to ISO format string

    Args:
        d: Date object

    Returns:
        ISO format string or None
    """
    if d is None:
        return None

    return d.isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value
    """
    if value is None:
        return default

    if isinstance(value, (int, float, Decimal)):
        return float(value)

    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def generate_claim_number() -> str:
    """
    Generate unique claim number

    Returns:
        Claim number string
    """
    import uuid

    return f"CLM-{uuid.uuid4().hex[:12].upper()}"


def generate_policy_number() -> str:
    """
    Generate unique policy number

    Returns:
        Policy number string
    """
    import uuid

    return f"POL-{uuid.uuid4().hex[:10].upper()}"


def calculate_days_between(start_date: date, end_date: date) -> int:
    """
    Calculate number of days between two dates

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Number of days
    """
    delta = end_date - start_date
    return delta.days


def is_business_hours(hour: int) -> bool:
    """
    Check if hour is within business hours (9 AM - 5 PM)

    Args:
        hour: Hour (0-23)

    Returns:
        True if business hours, False otherwise
    """
    return 9 <= hour <= 17


def is_weekend(day_of_week: int) -> bool:
    """
    Check if day is weekend (Saturday=5, Sunday=6)

    Args:
        day_of_week: Day of week (0-6, where 0=Monday)

    Returns:
        True if weekend, False otherwise
    """
    return day_of_week in [5, 6]
