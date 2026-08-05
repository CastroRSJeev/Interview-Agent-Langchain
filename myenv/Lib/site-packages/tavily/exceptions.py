class TavilyError(Exception):
    """Base exception for Tavily client"""
    pass

class MissingAPIKeyError(TavilyError):
    """Raised when API key is missing"""
    pass

class InvalidAPIKeyError(TavilyError):
    """Raised when API key is invalid"""
    pass

class UsageLimitExceededError(TavilyError):
    """Raised when usage limit is exceeded"""
    pass

class BadRequestError(TavilyError):
    """Raised for bad requests"""
    pass

class ForbiddenError(TavilyError):
    """Raised for forbidden access"""
    pass

class TimeoutError(TavilyError):
    """Raised when request times out"""
    pass