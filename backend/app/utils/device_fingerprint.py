import hashlib
from typing import Dict, Optional
from user_agents import parse


def generate_device_fingerprint(
    ip_address: str,
    user_agent: str,
    additional_data: Optional[Dict] = None
) -> str:
    """
    Generate a device fingerprint from request data

    Args:
        ip_address: Client IP address
        user_agent: User agent string
        additional_data: Additional data to include in fingerprint

    Returns:
        Device fingerprint hash
    """
    # Combine data for fingerprinting
    fingerprint_data = f"{ip_address}|{user_agent}"

    if additional_data:
        for key, value in sorted(additional_data.items()):
            fingerprint_data += f"|{key}:{value}"

    # Generate hash
    fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()

    return fingerprint


def parse_user_agent(user_agent_string: str) -> Dict[str, str]:
    """
    Parse user agent string to extract device information

    Args:
        user_agent_string: User agent string from request

    Returns:
        Dictionary with device information
    """
    user_agent = parse(user_agent_string)

    # Determine device type
    if user_agent.is_mobile:
        device_type = "mobile"
    elif user_agent.is_tablet:
        device_type = "tablet"
    elif user_agent.is_pc:
        device_type = "desktop"
    else:
        device_type = "other"

    return {
        'device_type': device_type,
        'browser': user_agent.browser.family,
        'browser_version': user_agent.browser.version_string,
        'os': user_agent.os.family,
        'os_version': user_agent.os.version_string,
        'is_mobile': user_agent.is_mobile,
        'is_tablet': user_agent.is_tablet,
        'is_pc': user_agent.is_pc,
        'is_bot': user_agent.is_bot
    }


def calculate_device_trust_score(
    device_age_days: int,
    usage_count: int,
    is_known_location: bool = True
) -> float:
    """
    Calculate trust score for a device

    Args:
        device_age_days: How long device has been used (days)
        usage_count: Number of times device has been used
        is_known_location: Whether device is in known location

    Returns:
        Trust score between 0 and 1
    """
    # Base score
    score = 0.3

    # Age factor (older devices are more trusted)
    if device_age_days > 180:
        score += 0.3
    elif device_age_days > 30:
        score += 0.2
    elif device_age_days > 7:
        score += 0.1

    # Usage factor (frequently used devices are more trusted)
    if usage_count > 50:
        score += 0.2
    elif usage_count > 20:
        score += 0.15
    elif usage_count > 5:
        score += 0.1

    # Location factor
    if is_known_location:
        score += 0.2

    # Cap at 1.0
    return min(score, 1.0)
