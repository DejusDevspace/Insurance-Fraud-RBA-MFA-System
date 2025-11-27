import re
from typing import Optional
from datetime import date, datetime


def validate_email(email: str) -> bool:
    """
    Validate email format

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format

    Args:
        phone: Phone number to validate

    Returns:
        True if valid, False otherwise
    """
    # Remove common separators
    cleaned = re.sub(r'[\s\-\(\)\.]', '', phone)

    # Check if it's a valid number (10-15 digits)
    pattern = r'^\+?[1-9]\d{9,14}$'
    return bool(re.match(pattern, cleaned))


def validate_claim_amount(amount: float, min_amount: float = 1, max_amount: float = 1_000_000) -> bool:
    """
    Validate claim amount is within acceptable range

    Args:
        amount: Claim amount
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount

    Returns:
        True if valid, False otherwise
    """
    return min_amount <= amount <= max_amount


def validate_date_not_future(date_value: date) -> bool:
    """
    Validate that a date is not in the future

    Args:
        date_value: Date to validate

    Returns:
        True if valid, False otherwise
    """
    return date_value <= date.today()


def validate_date_not_too_old(date_value: date, max_age_days: int = 730) -> bool:
    """
    Validate that a date is not too old

    Args:
        date_value: Date to validate
        max_age_days: Maximum age in days (default: 2 years)

    Returns:
        True if valid, False otherwise
    """
    from datetime import timedelta

    oldest_allowed = date.today() - timedelta(days=max_age_days)
    return date_value >= oldest_allowed


def validate_password_strength(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit"

    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(char.islower() for char in password):
        return False, "Password must contain at least one lowercase letter"

    # Check for special characters
    # if not any(char in "!@#$%^&*()_+-=[]{}|;:,.<>?" for char in password):
    #     return False, "Password must contain at least one special character"

    return True, None


def sanitize_input(input_string: str) -> str:
    """
    Sanitize user input to prevent injection attacks

    Args:
        input_string: String to sanitize

    Returns:
        Sanitized string
    """
    # Remove potentially dangerous characters
    # This is basic sanitization - normally, should use proper ORM to prevent SQL injection
    dangerous_chars = ['<', '>', '"', "'", ';', '--', '/*', '*/', 'script']

    sanitized = input_string
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized.strip()
