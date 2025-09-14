# tests/conftest.py
"""
Pytest Configuration and Shared Fixtures
========================================

This module provides shared pytest configuration, fixtures, and utilities
for the AutoERP test suite. It sets up test databases, clients, and
common test data that can be reused across test modules.

Key Features:
- Database fixtures with automatic cleanup
- FastAPI test client configuration
- Authentication and user management fixtures
- Logging configuration for test visibility
- Mock services and external dependencies
- Performance testing utilities

Fixtures provided:
- db_session: Database session for tests
- test_client: FastAPI test client
- admin_user: Admin user for testing
- regular_user: Regular user for testing
- sample_data: Common test data sets
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# AutoERP imports
from autoerp.core import (
    AutoERPApplication, AutoERPConfig, DatabaseConfig, SecurityConfig,
    User, UserRole, BaseModel, ConnectionManager, SessionManager,
    PasswordManager, UserService, NotificationService, CRUDService
)
from autoerp.api import app as fastapi_app

# Test constants from __init__.py
from . import (
    TEST_DATABASE_URL, DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_PASSWORD, DEFAULT_USER_USERNAME, DEFAULT_USER_EMAIL,
    DEFAULT_USER_PASSWORD
)

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test.log')
    ]
)

# Reduce noise from external libraries during testing
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "ui: marks tests as UI tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        elif "test_ui" in item.nodeid:
            item.add_marker(pytest.mark.ui)
        elif "test_security" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


# ==================== CONFIGURATION FIXTURES ====================

@pytest.fixture(scope="session")
def test_config() -> AutoERPConfig:
    """
    Create test configuration with in-memory database and test settings.
    
    Returns:
        AutoERPConfig: Test configuration instance
    """
    config = AutoERPConfig()
    
    # Override database settings for testing
    config.database.engine = 'sqlite'
    config.database.database = ':memory:'
    config.database.echo = False
    
    # Override security settings for testing
    config.security.secret_key = 'test-secret-key-not-for-production'
    config.security.jwt_expiration_minutes = 60
    config.security.password_min_length = 6
    
    # Override system settings for testing
    config.system.debug = True
    config.system.testing = True
    config.system.log_level = 'DEBUG'
    
    logger.info("Created test configuration")
    return config


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# ==================== DATABASE FIXTURES ====================

@pytest.fixture(scope="session")
def test_engine():
    """
    Create test database engine with in-memory SQLite.
    
    Returns:
        Engine: SQLAlchemy engine for testing
    """
    engine = create_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
        echo=False
    )
    
    logger.info("Created test database engine")
    return engine


@pytest.fixture(scope="session")
def test_sessionmaker(test_engine):
    """
    Create session maker for test database.
    
    Args:
        test_engine: Test database engine
        
    Returns:
        sessionmaker: Session factory for tests
    """
    return sessionmaker(bind=test_engine, autocommit=False, autoflush=False)


@pytest.fixture(scope="function")
def db_session(test_sessionmaker) -> Session:
    """
    Create database session for individual tests with automatic cleanup.
    
    Args:
        test_sessionmaker: Session factory
        
    Yields:
        Session: Database session for test
    """
    session = test_sessionmaker()
    
    try:
        # Create tables for each test
        from autoerp.core import BaseModel
        BaseModel.metadata.create_all(bind=session.bind)
        
        logger.debug("Created test database session")
        yield session
        
    except Exception as e:
        logger.error(f"Database session error: {e}")
        session.rollback()
        raise
    finally:
        session.close()
        
        # Drop tables after each test for clean state
        BaseModel.metadata.drop_all(bind=session.bind)
        logger.debug("Cleaned up test database session")


@pytest_asyncio.fixture
async def app_instance(test_config) -> AsyncGenerator[AutoERPApplication, None]:
    """
    Create AutoERP application instance for testing.
    
    Args:
        test_config: Test configuration
        
    Yields:
        AutoERPApplication: Configured application instance
    """
    async with AutoERPApplication(test_config) as app:
        await app.initialize()
        logger.info("Initialized test application instance")
        yield app
    
    logger.info("Cleaned up test application instance")


# ==================== API CLIENT FIXTURES ====================

@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """
    Create FastAPI test client for API testing.
    
    Returns:
        TestClient: Configured test client
    """
    client = TestClient(fastapi_app)
    logger.debug("Created FastAPI test client")
    return client


@pytest.fixture(scope="function")
def authenticated_client(test_client, admin_user_data) -> TestClient:
    """
    Create authenticated test client with admin user.
    
    Args:
        test_client: Base test client
        admin_user_data: Admin user data
        
    Returns:
        TestClient: Authenticated test client
    """
    # Login and get session token
    login_response = test_client.post(
        "/api/auth/login",
        json={
            "username_or_email": admin_user_data["username"],
            "password": admin_user_data["password"]
        }
    )
    
    if login_response.status_code == 200:
        token_data = login_response.json()
        session_token = token_data["data"]["session_token"]
        
        # Set authorization header
        test_client.headers.update({"Authorization": f"Bearer {session_token}"})
        logger.debug("Created authenticated test client")
    else:
        logger.error(f"Failed to authenticate test client: {login_response.json()}")
    
    return test_client


# ==================== USER FIXTURES ====================

@pytest.fixture(scope="session")
def admin_user_data() -> Dict[str, Any]:
    """
    Admin user data for testing.
    
    Returns:
        Dict[str, Any]: Admin user data
    """
    return {
        "username": DEFAULT_ADMIN_USERNAME,
        "email": DEFAULT_ADMIN_EMAIL,
        "password": DEFAULT_ADMIN_PASSWORD,
        "first_name": "Admin",
        "last_name": "User",
        "role": UserRole.SUPER_ADMIN
    }


@pytest.fixture(scope="session")
def regular_user_data() -> Dict[str, Any]:
    """
    Regular user data for testing.
    
    Returns:
        Dict[str, Any]: Regular user data
    """
    return {
        "username": DEFAULT_USER_USERNAME,
        "email": DEFAULT_USER_EMAIL,
        "password": DEFAULT_USER_PASSWORD,
        "first_name": "Test",
        "last_name": "User", 
        "role": UserRole.USER
    }


@pytest_asyncio.fixture
async def admin_user(app_instance, admin_user_data) -> User:
    """
    Create admin user for testing.
    
    Args:
        app_instance: Application instance
        admin_user_data: Admin user data
        
    Returns:
        User: Created admin user
    """
    user_service = app_instance.user_service
    
    result = await user_service.create_user(**admin_user_data)
    
    if result.is_success():
        user = result.get_data()
        logger.info(f"Created admin user: {user.username}")
        return user
    else:
        pytest.fail(f"Failed to create admin user: {result.error_message}")


@pytest_asyncio.fixture
async def regular_user(app_instance, regular_user_data) -> User:
    """
    Create regular user for testing.
    
    Args:
        app_instance: Application instance
        regular_user_data: Regular user data
        
    Returns:
        User: Created regular user
    """
    user_service = app_instance.user_service
    
    result = await user_service.create_user(**regular_user_data)
    
    if result.is_success():
        user = result.get_data()
        logger.info(f"Created regular user: {user.username}")
        return user
    else:
        pytest.fail(f"Failed to create regular user: {result.error_message}")


# ==================== SERVICE FIXTURES ====================

@pytest_asyncio.fixture
async def user_service(app_instance) -> UserService:
    """
    Get user service for testing.
    
    Args:
        app_instance: Application instance
        
    Returns:
        UserService: User service instance
    """
    return app_instance.user_service


@pytest_asyncio.fixture
async def notification_service(app_instance) -> NotificationService:
    """
    Get notification service for testing.
    
    Args:
        app_instance: Application instance
        
    Returns:
        NotificationService: Notification service instance
    """
    return app_instance.notification_service


@pytest_asyncio.fixture
async def crud_service(app_instance) -> CRUDService:
    """
    Get CRUD service for testing.
    
    Args:
        app_instance: Application instance
        
    Returns:
        CRUDService: CRUD service instance
    """
    return CRUDService(
        model_class=User,
        unit_of_work=app_instance.uow,
        audit_logger=getattr(app_instance, 'audit_logger', None),
        event_dispatcher=app_instance.event_dispatcher
    )


# ==================== TEST DATA FIXTURES ====================

@pytest.fixture(scope="function")
def sample_users_data() -> List[Dict[str, Any]]:
    """
    Sample users data for bulk testing.
    
    Returns:
        List[Dict[str, Any]]: List of user data
    """
    return [
        {
            "username": f"user{i}",
            "email": f"user{i}@autoerp.com",
            "password": "testpass123",
            "first_name": f"User",
            "last_name": f"{i}",
            "role": UserRole.USER
        }
        for i in range(1, 6)  # Create 5 test users
    ]


@pytest.fixture(scope="function")
def sample_company_data() -> Dict[str, Any]:
    """
    Sample company data for testing.
    
    Returns:
        Dict[str, Any]: Company data
    """
    from autoerp.core import Organization, Address, ContactInfo
    
    return {
        "name": "Test Company Ltd",
        "legal_name": "Test Company Limited",
        "organization_type": "Corporation",
        "tax_id": "TC123456789",
        "industry": "Technology",
        "founded_date": datetime(2020, 1, 1).date(),
        "contact_info": {
            "primary_email": "info@testcompany.com",
            "primary_phone": "+1-555-0123",
            "website": "https://testcompany.com"
        },
        "addresses": [
            {
                "street1": "123 Test Street",
                "city": "Test City",
                "state_province": "Test State", 
                "postal_code": "12345",
                "country": "US"
            }
        ]
    }


@pytest.fixture(scope="function")
def sample_product_data() -> Dict[str, Any]:
    """
    Sample product data for testing.
    
    Returns:
        Dict[str, Any]: Product data
    """
    return {
        "name": "Test Product",
        "description": "A test product for automated testing",
        "sku": "TEST-001",
        "price": 99.99,
        "currency": "USD",
        "category": "Test Category",
        "in_stock": True,
        "stock_quantity": 100,
        "weight": 1.5,
        "dimensions": {
            "length": 10.0,
            "width": 5.0,
            "height": 2.0
        }
    }


# ==================== MOCK FIXTURES ====================

@pytest.fixture(scope="function")
def mock_email_service():
    """
    Mock email service for testing notifications.
    
    Returns:
        Mock: Mocked email service
    """
    mock_service = Mock()
    mock_service.send_email.return_value = True
    mock_service.send_bulk_email.return_value = {"sent": 5, "failed": 0}
    
    with patch('autoerp.core.NotificationService._send_email', mock_service.send_email):
        yield mock_service


@pytest.fixture(scope="function")
def mock_redis_cache():
    """
    Mock Redis cache for testing caching functionality.
    
    Returns:
        Mock: Mocked Redis cache
    """
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.exists.return_value = False
    
    with patch('autoerp.core.CacheManager', return_value=mock_cache):
        yield mock_cache


@pytest.fixture(scope="function") 
def mock_external_api():
    """
    Mock external API calls for testing integrations.
    
    Returns:
        Mock: Mocked external API
    """
    mock_api = Mock()
    mock_api.get.return_value = {"status": "success", "data": {}}
    mock_api.post.return_value = {"status": "created", "id": "123"}
    mock_api.put.return_value = {"status": "updated"}
    mock_api.delete.return_value = {"status": "deleted"}
    
    yield mock_api


# ==================== PERFORMANCE FIXTURES ====================

@pytest.fixture(scope="function")
def performance_timer():
    """
    Performance timing fixture for benchmarking tests.
    
    Yields:
        callable: Timer function that returns elapsed time
    """
    import time
    
    start_times = {}
    
    def timer(name: str = "default") -> float:
        current_time = time.time()
        
        if name not in start_times:
            start_times[name] = current_time
            return 0.0
        else:
            elapsed = current_time - start_times[name]
            del start_times[name]
            return elapsed
    
    yield timer


@pytest.fixture(scope="function")
def memory_profiler():
    """
    Memory profiling fixture for monitoring memory usage.
    
    Yields:
        callable: Memory profiler function
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        def get_memory_usage() -> Dict[str, float]:
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024,  # MB
            }
        
        yield get_memory_usage
        
    except ImportError:
        # Fallback if psutil not available
        def dummy_profiler():
            return {"rss": 0.0, "vms": 0.0}
        
        yield dummy_profiler


# ==================== CLEANUP FIXTURES ====================

@pytest.fixture(scope="function", autouse=True)
def cleanup_test_files():
    """
    Automatically clean up temporary files created during tests.
    
    Yields:
        None: Runs cleanup after test completion
    """
    temp_files = []
    temp_dirs = []
    
    def register_temp_file(filepath: str):
        temp_files.append(filepath)
    
    def register_temp_dir(dirpath: str):
        temp_dirs.append(dirpath)
    
    # Make cleanup functions available to tests
    cleanup_test_files.register_file = register_temp_file
    cleanup_test_files.register_dir = register_temp_dir
    
    yield
    
    # Cleanup after test
    import shutil
    
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Cleaned up temp file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {filepath}: {e}")
    
    for dirpath in temp_dirs:
        try:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
                logger.debug(f"Cleaned up temp directory: {dirpath}")
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {dirpath}: {e}")


# ==================== TEST UTILITIES ====================

def assert_valid_uuid(uuid_string: str) -> None:
    """
    Assert that a string is a valid UUID.
    
    Args:
        uuid_string: String to validate as UUID
        
    Raises:
        AssertionError: If string is not a valid UUID
    """
    import uuid
    
    try:
        uuid.UUID(uuid_string)
    except ValueError:
        pytest.fail(f"'{uuid_string}' is not a valid UUID")


def assert_valid_datetime(datetime_string: str) -> None:
    """
    Assert that a string is a valid ISO datetime.
    
    Args:
        datetime_string: String to validate as datetime
        
    Raises:
        AssertionError: If string is not a valid datetime
    """
    try:
        datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
    except ValueError:
        pytest.fail(f"'{datetime_string}' is not a valid ISO datetime")


def assert_valid_email(email_string: str) -> None:
    """
    Assert that a string is a valid email address.
    
    Args:
        email_string: String to validate as email
        
    Raises:
        AssertionError: If string is not a valid email
    """
    import re
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email_string):
        pytest.fail(f"'{email_string}' is not a valid email address")


# Export utility functions for use in tests
__all__ = [
    "assert_valid_uuid",
    "assert_valid_datetime", 
    "assert_valid_email"
]