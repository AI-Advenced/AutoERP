# tests/__init__.py
"""
AutoERP Test Suite
==================

Comprehensive test suite for the AutoERP Enterprise Resource Planning System.

This package contains all automated tests including:
- Unit tests for core business logic
- Integration tests for API endpoints  
- UI component tests for web interfaces
- Performance and load tests
- Security and validation tests

Test Structure:
- conftest.py: Shared pytest configuration and fixtures
- test_core.py: Core business logic and domain model tests
- test_api.py: REST API endpoint integration tests
- test_ui.py: User interface component tests
- test_security.py: Authentication and authorization tests
- test_performance.py: Performance and scalability tests

Usage:
    # Run all tests
    pytest tests/
    
    # Run specific test file
    pytest tests/test_core.py
    
    # Run with coverage
    pytest --cov=autoerp tests/
    
    # Run with verbose output
    pytest -v tests/

Requirements:
- pytest >= 7.4.0
- pytest-asyncio >= 0.21.1
- pytest-cov >= 4.1.0
- httpx >= 0.24.0 (for API testing)
- factory-boy >= 3.3.0 (for test data)
- faker >= 19.6.0 (for fake data generation)

Author: AutoERP Development Team
License: MIT
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test suite metadata
__version__ = "1.0.0"
__author__ = "AutoERP Development Team"
__email__ = "dev@autoerp.com"

# Test configuration constants
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/1"
TEST_API_BASE_URL = "http://testserver"

# Test data constants
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_EMAIL = "admin@autoerp.com" 
DEFAULT_ADMIN_PASSWORD = "admin123456"

DEFAULT_USER_USERNAME = "testuser"
DEFAULT_USER_EMAIL = "test@autoerp.com"
DEFAULT_USER_PASSWORD = "testpass123"

# Export commonly used test utilities
__all__ = [
    "TEST_DATABASE_URL",
    "TEST_REDIS_URL", 
    "TEST_API_BASE_URL",
    "DEFAULT_ADMIN_USERNAME",
    "DEFAULT_ADMIN_EMAIL",
    "DEFAULT_ADMIN_PASSWORD",
    "DEFAULT_USER_USERNAME",
    "DEFAULT_USER_EMAIL",
    "DEFAULT_USER_PASSWORD"
]