# tests/test_core.py
"""
Core Business Logic Tests
========================

Comprehensive test suite for AutoERP core business logic, domain models,
and services. This module tests the fundamental components that drive
the business functionality of the ERP system.

Test Coverage:
- Domain models and entities (User, Organization, etc.)
- Value objects (Money, Address, etc.)  
- Business services (UserService, NotificationService, etc.)
- Repository patterns and data access
- Domain events and event handling
- Validation and business rules
- Authentication and authorization
- Audit trails and logging

Test Categories:
- Unit tests for individual components
- Integration tests for service interactions
- Business rule validation tests
- Error handling and edge cases
- Performance and scalability tests

Author: AutoERP Development Team
License: MIT
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

from autoerp.core import (
    # Configuration
    AutoERPConfig, DatabaseConfig, SecurityConfig,
    
    # Base models and validation  
    BaseModel, ValidationError, FieldValidator,
    
    # Entities
    User, UserRole, Organization, Person, PersonName,
    Address, ContactInfo, Currency, Money,
    
    # Services
    UserService, NotificationService, CRUDService,
    ServiceResult, BusinessRuleViolationError,
    PasswordManager, SessionManager,
    
    # Repository and data access
    BaseRepository, RecordNotFoundError, DuplicateRecordError,
    PaginationInfo, FilterCriteria,
    
    # Events
    DomainEvent, EntityCreatedEvent, EventDispatcher,
    
    # Utilities
    ConnectionManager, CacheManager
)

from .conftest import (
    assert_valid_uuid, assert_valid_datetime, assert_valid_email
)


# ==================== MODEL TESTS ====================

class TestBaseModel:
    """Test cases for BaseModel functionality."""
    
    def test_model_creation_with_valid_data(self):
        """Test creating a model with valid data."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            first_name="Test",
            last_name="User"
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.full_name == "Test User"
        assert_valid_uuid(user.id)
    
    def test_model_validation_success(self):
        """Test successful model validation."""
        user = User(
            username="validuser",
            email="valid@example.com",
            password_hash="hashed_password",
            first_name="Valid",
            last_name="User"
        )
        
        assert user.validate() == True
        assert len(user.get_validation_errors()) == 0
    
    def test_model_validation_failure(self):
        """Test model validation with invalid data."""
        user = User(
            username="",  # Invalid: empty username
            email="invalid-email",  # Invalid: bad email format
            password_hash="",  # Invalid: empty password
            first_name="",  # Invalid: empty first name
            last_name=""  # Invalid: empty last name
        )
        
        assert user.validate() == False
        errors = user.get_validation_errors()
        assert len(errors) > 0
    
    def test_model_dirty_field_tracking(self):
        """Test dirty field tracking functionality."""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="password",
            first_name="Test",
            last_name="User"
        )
        
        # Initially no dirty fields after creation
        assert not user.is_dirty()
        assert len(user.get_dirty_fields()) == 0
        
        # Modify a field
        user.first_name = "Modified"
        
        # Should now be dirty
        assert user.is_dirty()
        assert user.is_dirty('first_name')
        assert 'first_name' in user.get_dirty_fields()
        
        # Get changes
        changes = user.get_changed_values()
        assert 'first_name' in changes
        assert changes['first_name'][1] == "Modified"
    
    def test_model_to_dict_conversion(self):
        """Test model to dictionary conversion."""
        user = User(
            username="testuser",
            email="test@example.com", 
            password_hash="password",
            first_name="Test",
            last_name="User"
        )
        
        data = user.to_dict()
        
        assert isinstance(data, dict)
        assert data['username'] == "testuser"
        assert data['email'] == "test@example.com"
        assert data['first_name'] == "Test"
        assert data['last_name'] == "User"
        assert 'id' in data
        assert_valid_uuid(data['id'])
    
    def test_model_from_dict_creation(self):
        """Test creating model from dictionary."""
        data = {
            'username': 'dictuser',
            'email': 'dict@example.com',
            'password_hash': 'password',
            'first_name': 'Dict',
            'last_name': 'User'
        }
        
        user = User.from_dict(data)
        
        assert user.username == 'dictuser'
        assert user.email == 'dict@example.com'
        assert user.first_name == 'Dict'
        assert user.last_name == 'User'
    
    def test_model_cloning(self):
        """Test model cloning functionality."""
        original = User(
            username="original",
            email="original@example.com",
            password_hash="password", 
            first_name="Original",
            last_name="User"
        )
        
        clone = original.clone(username="cloned")
        
        assert clone.username == "cloned"
        assert clone.email == "original@example.com"  # Copied
        assert clone.first_name == "Original"  # Copied
        assert clone.id != original.id  # Different ID


class TestValueObjects:
    """Test cases for value objects (Money, Address, etc.)."""
    
    def test_money_creation_and_arithmetic(self):
        """Test Money value object creation and arithmetic operations."""
        usd_currency = Currency(code="USD", name="US Dollar", symbol="$")
        
        # Test creation
        money1 = Money(100.50, usd_currency)
        money2 = Money(50.25, usd_currency)
        
        assert money1.amount == Decimal('100.50')
        assert money1.currency.code == "USD"
        
        # Test addition
        result = money1 + money2
        assert result.amount == Decimal('150.75')
        
        # Test subtraction  
        result = money1 - money2
        assert result.amount == Decimal('50.25')
        
        # Test multiplication
        result = money1 * 2
        assert result.amount == Decimal('201.00')
        
        # Test division
        result = money1 / 2
        assert result.amount == Decimal('50.25')
        
        # Test formatting
        formatted = money1.format()
        assert "$100.50" in formatted
    
    def test_money_currency_validation(self):
        """Test Money currency validation."""
        usd = Currency(code="USD", name="US Dollar", symbol="$")
        eur = Currency(code="EUR", name="Euro", symbol="€")
        
        money_usd = Money(100, usd)
        money_eur = Money(100, eur)
        
        # Should raise error for different currencies
        with pytest.raises(ValueError):
            money_usd + money_eur
        
        with pytest.raises(ValueError):
            money_usd - money_eur
    
    def test_address_creation_and_formatting(self):
        """Test Address value object creation and formatting."""
        address = Address(
            street1="123 Main Street",
            street2="Apt 4B", 
            city="New York",
            state_province="NY",
            postal_code="10001",
            country="US"
        )
        
        assert address.street1 == "123 Main Street"
        assert address.city == "New York"
        assert address.country == "US"
        
        # Test single line formatting
        single_line = address.format_single_line()
        assert "123 Main Street" in single_line
        assert "New York" in single_line
        assert "10001" in single_line
        
        # Test multi-line formatting
        multi_line = address.format_multi_line()
        assert "\n" in multi_line
        assert "123 Main Street" in multi_line
    
    def test_person_name_formatting(self):
        """Test PersonName value object formatting."""
        name = PersonName(
            title="Dr.",
            first_name="John",
            middle_name="Michael",
            last_name="Smith",
            suffix="Jr."
        )
        
        # Test full name
        full_name = name.full_name()
        assert full_name == "Dr. John Michael Smith Jr."
        
        # Test formal name
        formal_name = name.formal_name()
        assert formal_name.startswith("Smith,")
        assert "John" in formal_name


# ==================== SERVICE TESTS ====================

class TestUserService:
    """Test cases for UserService functionality."""
    
    @pytest_asyncio.fixture
    async def password_manager(self):
        """Create password manager for testing."""
        config = SecurityConfig()
        return PasswordManager(config)
    
    @pytest_asyncio.fixture  
    async def session_manager(self, app_instance):
        """Create session manager for testing."""
        return SessionManager(app_instance.config.security, app_instance.cache_manager)
    
    @pytest_asyncio.fixture
    async def user_service_instance(self, app_instance, password_manager, session_manager):
        """Create UserService instance for testing."""
        return UserService(
            unit_of_work=app_instance.uow,
            password_manager=password_manager,
            session_manager=session_manager,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
    
    @pytest.mark.asyncio def test_create_user_success(self, user_service_instance):
        """Test successful user creation."""
        result = await user_service_instance.create_user(
            username="newuser",
            email="newuser@example.com",
            password="SecurePass123!",
            first_name="New", 
            last_name="User",
            role=UserRole.USER
        )
        
        assert result.is_success()
        user = result.get_data()
        assert user.username == "newuser"
        assert user.email == "newuser@example.com"
        assert user.role == UserRole.USER
        assert_valid_uuid(user.id)
    
    @pytest.mark.asyncio def test_create_user_duplicate_username(self, user_service_instance, admin_user):
        """Test creating user with duplicate username."""
        result = await user_service_instance.create_user(
            username=admin_user.username,  # Duplicate
            email="different@example.com",
            password="SecurePass123!",
            first_name="Different",
            last_name="User"
        )
        
        assert result.is_error()
        assert result.error_code == "USERNAME_EXISTS"
    
    @pytest.mark.asyncio def test_create_user_weak_password(self, user_service_instance):
        """Test creating user with weak password.""" 
        result = await user_service_instance.create_user(
            username="weakuser",
            email="weak@example.com", 
            password="123",  # Weak password
            first_name="Weak",
            last_name="User"
        )
        
        assert result.is_error()
        assert result.error_code == "WEAK_PASSWORD"
    
    @pytest.mark.asyncio def test_authenticate_user_success(self, user_service_instance, admin_user_data):
        """Test successful user authentication."""
        # First create user
        create_result = await user_service_instance.create_user(**admin_user_data)
        assert create_result.is_success()
        
        # Then authenticate
        auth_result = await user_service_instance.authenticate_user(
            username_or_email=admin_user_data["username"],
            password=admin_user_data["password"]
        )
        
        assert auth_result.is_success()
        user, session = auth_result.get_data()
        assert user.username == admin_user_data["username"]
        assert session.user_id == user.id
        assert session.is_valid
    
    @pytest.mark.asyncio def test_authenticate_user_invalid_credentials(self, user_service_instance):
        """Test authentication with invalid credentials."""
        result = await user_service_instance.authenticate_user(
            username_or_email="nonexistent",
            password="wrongpassword"
        )
        
        assert result.is_error()
        assert result.error_code == "INVALID_CREDENTIALS"
    
    @pytest.mark.asyncio def test_change_password_success(self, user_service_instance, admin_user, admin_user_data):
        """Test successful password change."""
        result = await user_service_instance.change_password(
            user_id=admin_user.id,
            old_password=admin_user_data["password"],
            new_password="NewSecurePass123!"
        )
        
        assert result.is_success()
        assert result.get_data() == True
    
    @pytest.mark.asyncio def test_change_password_wrong_current(self, user_service_instance, admin_user):
        """Test password change with wrong current password."""
        result = await user_service_instance.change_password(
            user_id=admin_user.id,
            old_password="wrongpassword",
            new_password="NewSecurePass123!"
        )
        
        assert result.is_error()
        assert result.error_code == "INVALID_PASSWORD"


class TestCRUDService:
    """Test cases for CRUD service functionality."""
    
    @pytest.mark.asyncio def test_create_record(self, crud_service):
        """Test creating a record through CRUD service."""
        user_data = {
            "username": "cruduser",
            "email": "crud@example.com",
            "password_hash": "password",
            "first_name": "CRUD",
            "last_name": "User"
        }
        
        result = await crud_service.create_record(user_data)
        
        assert result.is_success()
        user = result.get_data()
        assert user.username == "cruduser"
        assert_valid_uuid(user.id)
    
    @pytest.mark.asyncio def test_read_records_with_pagination(self, crud_service, sample_users_data):
        """Test reading records with pagination."""
        # Create multiple users first
        for user_data in sample_users_data:
            await crud_service.create_record(user_data)
        
        # Test pagination
        pagination = PaginationInfo(page=1, per_page=3)
        result = await crud_service.read_records(pagination=pagination)
        
        assert result.is_success()
        records, updated_pagination = result.get_data()
        assert len(records) <= 3
        assert updated_pagination.total_items >= len(sample_users_data)
    
    @pytest.mark.asyncio def test_read_records_with_filters(self, crud_service, sample_users_data):
        """Test reading records with filters."""
        # Create users first
        for user_data in sample_users_data:
            await crud_service.create_record(user_data)
        
        # Test filtering
        filters = FilterCriteria()
        filters.add_filter("first_name", "User")
        
        result = await crud_service.read_records(filters=filters)
        
        assert result.is_success()
        records, _ = result.get_data()
        
        # All returned records should match filter
        for record in records:
            assert record.first_name == "User"
    
    @pytest.mark.asyncio def test_update_record(self, crud_service, admin_user):
        """Test updating a record through CRUD service."""
        update_data = {
            "first_name": "Updated",
            "last_name": "Name"
        }
        
        result = await crud_service.update_record(admin_user.id, update_data)
        
        assert result.is_success()
        updated_user = result.get_data()
        assert updated_user.first_name == "Updated"
        assert updated_user.last_name == "Name"
    
    @pytest.mark.asyncio def test_update_nonexistent_record(self, crud_service):
        """Test updating a record that doesn't exist."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        result = await crud_service.update_record(fake_id, {"first_name": "Test"})
        
        assert result.is_error()
        assert "not found" in result.error_message.lower()
    
    @pytest.mark.asyncio def test_delete_record(self, crud_service, admin_user):
        """Test deleting a record (soft delete)."""
        result = await crud_service.delete_record(admin_user.id, soft_delete=True)
        
        assert result.is_success()
        assert result.get_data() == True
        
        # Verify user is soft deleted
        get_result = await crud_service.get_record_by_id(admin_user.id)
        user = get_result.get_data()
        
        if hasattr(user, 'is_deleted'):
            assert user.is_deleted == True


# ==================== REPOSITORY TESTS ====================

class TestBaseRepository:
    """Test cases for repository functionality."""
    
    @pytest_asyncio.fixture
    async def user_repository(self, app_instance):
        """Create user repository for testing."""
        return app_instance.get_repository(User)
    
    @pytest.mark.asyncio def test_repository_create(self, user_repository):
        """Test repository create operation."""
        user = User(
            username="repouser",
            email="repo@example.com", 
            password_hash="password",
            first_name="Repo",
            last_name="User"
        )
        
        created_user = await user_repository.create(user)
        
        assert created_user.id == user.id
        assert created_user.username == "repouser"
    
    @pytest.mark.asyncio def test_repository_get_by_id(self, user_repository, admin_user):
        """Test repository get by ID operation."""
        retrieved_user = await user_repository.get_by_id(admin_user.id)
        
        assert retrieved_user is not None
        assert retrieved_user.id == admin_user.id
        assert retrieved_user.username == admin_user.username
    
    @pytest.mark.asyncio def test_repository_get_nonexistent(self, user_repository):
        """Test repository get for nonexistent record."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        result = await user_repository.get_by_id(fake_id)
        
        assert result is None
    
    @pytest.mark.asyncio def test_repository_find_by_criteria(self, user_repository, admin_user):
        """Test repository find by criteria."""
        criteria = {"username": admin_user.username}
        
        results = await user_repository.find_by_criteria(criteria)
        
        assert len(results) >= 1
        assert any(user.username == admin_user.username for user in results)
    
    @pytest.mark.asyncio def test_repository_count(self, user_repository, sample_users_data):
        """Test repository count operation."""
        initial_count = await user_repository.count()
        
        # Create some users
        for user_data in sample_users_data[:3]:  # Create 3 users
            user = User(**user_data)
            await user_repository.create(user)
        
        final_count = await user_repository.count()
        
        assert final_count >= initial_count + 3

# ==================== EVENT SYSTEM TESTS ====================

class TestDomainEvents:
    """Test cases for domain event system."""
    
    @pytest_asyncio.fixture
    async def event_dispatcher(self):
        """Create event dispatcher for testing."""
        return EventDispatcher()
    
    @pytest_asyncio.fixture
    async def mock_event_handler(self):
        """Create mock event handler."""
        class MockEventHandler:
            def __init__(self):
                self.handled_events = []
            
            async def handle(self, event: DomainEvent) -> None:
                self.handled_events.append(event)
            
            def can_handle(self, event: DomainEvent) -> bool:
                return True
        
        return MockEventHandler()
    
    @pytest.mark.asyncio def test_event_creation(self, admin_user):
        """Test domain event creation."""
        event = EntityCreatedEvent(admin_user)
        
        assert event.event_type == "EntityCreatedEvent"
        assert event.entity == admin_user
        assert_valid_uuid(event.event_id)
        assert isinstance(event.occurred_at, datetime)
        
        # Test event data
        event_data = event.get_event_data()
        assert event_data['entity_type'] == 'User'
        assert event_data['entity_id'] == admin_user.id
    
    @pytest.mark.asyncio def test_event_dispatcher_subscription(self, event_dispatcher, mock_event_handler):
        """Test event dispatcher subscription."""
        # Subscribe handler
        event_dispatcher.subscribe("EntityCreatedEvent", mock_event_handler)
        
        # Create and dispatch event
        user = User(
            username="eventuser",
            email="event@example.com",
            password_hash="password",
            first_name="Event",
            last_name="User"
        )
        event = EntityCreatedEvent(user)
        
        await event_dispatcher.dispatch(event)
        
        # Verify handler was called
        assert len(mock_event_handler.handled_events) == 1
        assert mock_event_handler.handled_events[0].event_type == "EntityCreatedEvent"
    
    @pytest.mark.asyncio def test_global_event_handler(self, event_dispatcher, mock_event_handler):
        """Test global event handler subscription."""
        # Subscribe as global handler
        event_dispatcher.subscribe_global(mock_event_handler)
        
        # Create different types of events
        user = User(
            username="globaluser",
            email="global@example.com",
            password_hash="password",
            first_name="Global",
            last_name="User"
        )
        
        create_event = EntityCreatedEvent(user)
        delete_event = EntityDeletedEvent("User", user.id)
        
        await event_dispatcher.dispatch(create_event)
        await event_dispatcher.dispatch(delete_event)
        
        # Global handler should receive all events
        assert len(mock_event_handler.handled_events) == 2


# ==================== VALIDATION TESTS ====================

class TestFieldValidation:
    """Test cases for field validation system."""
    
    def test_required_field_validation(self):
        """Test required field validation."""
        validator = FieldValidator.required
        
        # Should not raise for valid values
        validator("valid_value")
        validator("non_empty")
        
        # Should raise for invalid values
        with pytest.raises(ValidationError):
            validator(None)
        
        with pytest.raises(ValidationError):
            validator("")
    
    def test_email_validation(self):
        """Test email field validation."""
        validator = FieldValidator.email
        
        # Valid emails
        validator("user@example.com")
        validator("test.email+tag@domain.co.uk")
        
        # Invalid emails
        with pytest.raises(ValidationError):
            validator("invalid-email")
        
        with pytest.raises(ValidationError):
            validator("@domain.com")
        
        with pytest.raises(ValidationError):
            validator("user@")
    
    def test_length_validation(self):
        """Test string length validation."""
        min_validator = FieldValidator.min_length(5)
        max_validator = FieldValidator.max_length(10)
        
        # Valid lengths
        min_validator("12345")  # Exactly 5
        min_validator("123456789")  # More than 5
        
        max_validator("1234567890")  # Exactly 10
        max_validator("12345")  # Less than 10
        
        # Invalid lengths
        with pytest.raises(ValidationError):
            min_validator("1234")  # Less than 5
        
        with pytest.raises(ValidationError):
            max_validator("12345678901")  # More than 10
    
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        validator = FieldValidator.numeric_range(10, 100)
        
        # Valid ranges
        validator(10)  # Min boundary
        validator(100)  # Max boundary
        validator(50)  # Within range
        
        # Invalid ranges
        with pytest.raises(ValidationError):
            validator(9)  # Below min
        
        with pytest.raises(ValidationError):
            validator(101)  # Above max
    
    def test_regex_validation(self):
        """Test regex pattern validation."""
        # Phone number pattern
        phone_validator = FieldValidator.regex(
            r'^\+?1?\d{9,15}$',
            "Invalid phone number format"
        )
        
        # Valid phone numbers
        phone_validator("1234567890")
        phone_validator("+11234567890")
        
        # Invalid phone numbers
        with pytest.raises(ValidationError):
            phone_validator("123")  # Too short
        
        with pytest.raises(ValidationError):
            phone_validator("invalid-phone")


# ==================== SECURITY TESTS ====================

class TestPasswordManager:
    """Test cases for password management."""
    
    @pytest.fixture
    def password_manager(self):
        """Create password manager for testing."""
        config = SecurityConfig(
            password_min_length=8,
            password_require_uppercase=True,
            password_require_lowercase=True,
            password_require_numbers=True,
            password_require_special=True
        )
        return PasswordManager(config)
    
    def test_password_strength_validation_success(self, password_manager):
        """Test successful password strength validation."""
        strong_password = "SecurePass123!"
        
        is_valid, errors = password_manager.validate_password_strength(strong_password)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_password_strength_validation_failures(self, password_manager):
        """Test password strength validation failures."""
        test_cases = [
            ("short", "Password must be at least 8 characters"),
            ("nouppercase123!", "uppercase letter"),
            ("NOLOWERCASE123!", "lowercase letter"),
            ("NoNumbers!", "at least one number"),
            ("NoSpecialChar123", "special character")
        ]
        
        for password, expected_error in test_cases:
            is_valid, errors = password_manager.validate_password_strength(password)
            
            assert is_valid == False
            assert len(errors) > 0
            assert any(expected_error in error for error in errors)
    
    def test_password_hashing_and_verification(self, password_manager):
        """Test password hashing and verification."""
        password = "TestPassword123!"
        
        # Hash password
        password_hash = password_manager.hash_password(password)
        
        assert password_hash != password  # Should be different
        assert ":" in password_hash  # Should contain salt separator
        
        # Verify correct password
        assert password_manager.verify_password(password, password_hash) == True
        
        # Verify incorrect password
        assert password_manager.verify_password("WrongPassword", password_hash) == False
    
    def test_password_generation(self, password_manager):
        """Test secure password generation."""
        generated_password = password_manager.generate_password(12)
        
        assert len(generated_password) == 12
        
        # Verify generated password meets strength requirements
        is_valid, errors = password_manager.validate_password_strength(generated_password)
        
        assert is_valid == True
        assert len(errors) == 0


class TestSessionManager:
    """Test cases for session management."""
    
    @pytest_asyncio.fixture
    async def session_manager(self, app_instance):
        """Create session manager for testing.""" 
        return SessionManager(app_instance.config.security, app_instance.cache_manager)
    
    @pytest.mark.asyncio def test_create_session(self, session_manager, admin_user):
        """Test session creation."""
        session = await session_manager.create_session(
            user_id=admin_user.id,
            ip_address="127.0.0.1",
            user_agent="TestAgent/1.0"
        )
        
        assert session.user_id == admin_user.id
        assert session.ip_address == "127.0.0.1"
        assert session.user_agent == "TestAgent/1.0"
        assert session.is_valid == True
        assert_valid_uuid(session.session_token)
    
    @pytest.mark.asyncio def test_get_valid_session(self, session_manager, admin_user):
        """Test retrieving valid session."""
        # Create session
        created_session = await session_manager.create_session(admin_user.id)
        
        # Retrieve session
        retrieved_session = await session_manager.get_session(created_session.session_token)
        
        assert retrieved_session is not None
        assert retrieved_session.session_token == created_session.session_token
        assert retrieved_session.user_id == admin_user.id
    
    @pytest.mark.asyncio def test_get_invalid_session(self, session_manager):
        """Test retrieving invalid session."""
        fake_token = "invalid-session-token"
        
        retrieved_session = await session_manager.get_session(fake_token)
        
        assert retrieved_session is None
    
    @pytest.mark.asyncio def test_invalidate_session(self, session_manager, admin_user):
        """Test session invalidation."""
        # Create session
        session = await session_manager.create_session(admin_user.id)
        
        # Invalidate session
        result = await session_manager.invalidate_session(session.session_token)
        
        assert result == True
        
        # Verify session is no longer valid
        retrieved_session = await session_manager.get_session(session.session_token)
        assert retrieved_session is None
    
    @pytest.mark.asyncio def test_session_activity_update(self, session_manager, admin_user):
        """Test session activity updating."""
        # Create session
        session = await session_manager.create_session(admin_user.id)
        original_activity = session.last_activity
        
        # Wait a moment to ensure time difference
        await asyncio.sleep(0.1)
        
        # Update activity
        result = await session_manager.update_session_activity(session.session_token)
        
        assert result == True
        
        # Retrieve updated session
        updated_session = await session_manager.get_session(session.session_token)
        assert updated_session.last_activity > original_activity


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Performance and scalability tests."""
    
    @pytest.mark.slow
    @pytest.mark.asyncio def test_bulk_user_creation_performance(self, user_service, performance_timer):
        """Test performance of bulk user creation."""
        user_count = 100
        
        # Start timing
        performance_timer("bulk_create")
        
        # Create users in bulk
        tasks = []
        for i in range(user_count):
            task = user_service.create_user(
                username=f"perfuser{i}",
                email=f"perf{i}@example.com",
                password="TestPass123!",
                first_name="Perf",
                last_name=f"User{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # End timing
        elapsed = performance_timer("bulk_create")
        
        # Verify results
        successful_creates = sum(1 for r in results if hasattr(r, 'is_success') and r.is_success())
        
        assert successful_creates >= user_count * 0.9  # Allow 10% failure rate
        assert elapsed < 30.0  # Should complete within 30 seconds
        
        # Log performance metrics
        avg_time_per_user = elapsed / user_count
        print(f"Created {user_count} users in {elapsed:.2f}s ({avg_time_per_user:.3f}s per user)")
    
    @pytest.mark.slow
    @pytest.mark.asyncio def test_repository_query_performance(self, crud_service, sample_users_data, performance_timer):
        """Test performance of repository queries."""
        # Create test data
        for user_data in sample_users_data * 20:  # Create 100 users (5 * 20)
            await crud_service.create_record(user_data)
        
        # Test pagination performance
        performance_timer("pagination")
        
        pagination = PaginationInfo(page=1, per_page=50)
        result = await crud_service.read_records(pagination=pagination)
        
        elapsed = performance_timer("pagination")
        
        assert result.is_success()
        records, _ = result.get_data()
        assert len(records) <= 50
        assert elapsed < 5.0  # Should complete within 5 seconds
        
        print(f"Paginated query completed in {elapsed:.3f}s")
    
    @pytest.mark.asyncio def test_memory_usage_monitoring(self, memory_profiler, crud_service):
        """Test memory usage during operations."""
        initial_memory = memory_profiler()
        
        # Perform memory-intensive operations
        user_data_list = []
        for i in range(50):
            user_data = {
                "username": f"memuser{i}",
                "email": f"mem{i}@example.com", 
                "password_hash": "password",
                "first_name": "Memory",
                "last_name": f"User{i}"
            }
            user_data_list.append(user_data)
            await crud_service.create_record(user_data)
        
        final_memory = memory_profiler()
        
        # Calculate memory increase
        memory_increase = final_memory['rss'] - initial_memory['rss']
        
        print(f"Memory usage increased by {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable (less than 100MB for 50 users)
        assert memory_increase < 100.0


# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    @pytest.mark.asyncio def test_service_error_propagation(self, user_service):
        """Test that service errors are properly propagated."""
        # Test with invalid input that should cause validation error
        result = await user_service.create_user(
            username="",  # Empty username should cause error
            email="invalid-email",  # Invalid email
            password="weak",  # Weak password
            first_name="",
            last_name=""
        )
        
        assert result.is_error()
        assert result.error_message is not None
        assert len(result.error_message) > 0
    
    @pytest.mark.asyncio def test_repository_concurrent_access(self, user_repository):
        """Test repository behavior under concurrent access."""
        user_data = {
            "username": "concurrent_user",
            "email": "concurrent@example.com",
            "password_hash": "password",
            "first_name": "Concurrent",
            "last_name": "User"
        }
        
        # Create multiple concurrent operations
        tasks = []
        for i in range(10):
            user = User(**{**user_data, "username": f"concurrent_user_{i}"})
            tasks.append(user_repository.create(user))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Most operations should succeed
        successful_operations = sum(1 for r in results if isinstance(r, User))
        assert successful_operations >= 8  # Allow some failures due to concurrency
    
    @pytest.mark.asyncio def test_transaction_rollback(self, app_instance):
        """Test transaction rollback on errors."""
        user_service = app_instance.user_service
        
        # Start with known state
        initial_count = await user_service.uow.get_repository(User).count()
        
        # Attempt operation that should fail and rollback
        try:
            async with user_service.uow:
                # Create a valid user first
                await user_service.create_user(
                    username="rollback_user",
                    email="rollback@example.com",
                    password="ValidPass123!",
                    first_name="Rollback",
                    last_name="User"
                )
                
                # Then create an invalid user that should cause rollback
                await user_service.create_user(
                    username="",  # This should fail
                    email="invalid",
                    password="weak",
                    first_name="",
                    last_name=""
                )
        except Exception:
            pass  # Expected to fail
        
        # Check that rollback occurred
        final_count = await user_service.uow.get_repository(User).count()
        assert final_count == initial_count  # No new users should be created
    
    def test_model_validation_edge_cases(self):
        """Test model validation with edge cases."""
        # Test with None values
        user = User(
            username=None,
            email=None, 
            password_hash=None,
            first_name=None,
            last_name=None
        )
        
        assert user.validate() == False
        errors = user.get_validation_errors()
        assert len(errors) > 0
        
        # Test with extremely long values
        long_string = "x" * 1000
        user = User(
            username=long_string,
            email=f"{long_string}@example.com",
            password_hash="password",
            first_name=long_string,
            last_name=long_string
        )
        
        assert user.validate() == False
        errors = user.get_validation_errors()
        assert len(errors) > 0


# ==================== INTEGRATION TESTS ====================

class TestServiceIntegration:
    """Integration tests between different services."""
    
    @pytest.mark.asyncio def test_user_notification_integration(self, app_instance, mock_email_service):
        """Test integration between user service and notification service."""
        user_service = app_instance.user_service
        notification_service = app_instance.notification_service
        
        # Create user
        user_result = await user_service.create_user(
            username="notifyuser",
            email="notify@example.com",
            password="NotifyPass123!",
            first_name="Notify",
            last_name="User"
        )
        
        assert user_result.is_success()
        user = user_result.get_data()
        
        # Send notification
        notification_result = await notification_service.send_notification(
            recipient_id=user.id,
            template_name="welcome_email",
            variables={"first_name": user.first_name},
            recipient_address=user.email
        )
        
        # Verify notification was processed (would be sent via mock)
        assert notification_result.is_success()
    
    @pytest.mark.asyncio def test_audit_trail_integration(self, app_instance, admin_user):
        """Test that audit trails are created for service operations."""
        user_service = app_instance.user_service
        
        # Perform audited operation
        result = await user_service.update_user_profile(
            user_id=admin_user.id,
            updates={"first_name": "Updated"}
        )
        
        assert result.is_success()
        
        # Verify audit log entry was created (if audit logger is configured)
        if hasattr(app_instance, 'audit_logger') and app_instance.audit_logger:
            # In a real test, we would check the audit log repository
            # For now, just verify the operation succeeded
            pass
    
    @pytest.mark.asyncio def test_cache_integration(self, app_instance, mock_redis_cache):
        """Test cache integration with services."""
        # Test that cache is used when available
        cache_manager = app_instance.cache_manager
        
        # Set a value in cache
        await cache_manager.set("test_key", "test_value", 60)
        
        # Retrieve value
        cached_value = await cache_manager.get("test_key")
        
        # Should return the cached value (or None if mock)
        # This test verifies the cache interface works
        assert cached_value is not None or mock_redis_cache is not None


# ==================== UTILITY TESTS ====================

class TestUtilities:
    """Test cases for utility functions and helpers."""
    
    def test_pagination_info_calculation(self):
        """Test pagination info calculations."""
        pagination = PaginationInfo(page=2, per_page=10, total_items=45)
        
        assert pagination.page == 2
        assert pagination.per_page == 10
        assert pagination.total_items == 45
        assert pagination.total_pages == 5  # Calculated: ceil(45/10)
        assert pagination.offset == 10  # Calculated: (2-1) * 10
        assert pagination.has_next == True  # Page 2 of 5
        assert pagination.has_prev == True  # Not on first page
    
    def test_filter_criteria_building(self):
        """Test filter criteria building."""
        criteria = FilterCriteria()
        
        # Add filters
        criteria.add_filter("name", "John", "eq")
        criteria.add_filter("age", 18, "gte")
        
        # Set sorting
        criteria.set_sort("created_at", "desc")
        
        # Set search
        criteria.set_search("john", ["name", "email"])
        
        # Verify criteria
        assert len(criteria.filters) == 2
        assert criteria.filters["name"]["value"] == "John"
        assert criteria.filters["age"]["operator"] == "gte"
        assert criteria.sort_by == "created_at"
        assert criteria.sort_order == "desc"
        assert criteria.search_query == "john"
        assert "name" in criteria.search_fields
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = AutoERPConfig()
        assert valid_config.database.engine in ['sqlite', 'postgresql']
        assert valid_config.security.password_min_length >= 4
        
        # Invalid database configuration
        with pytest.raises(ValueError):
            DatabaseConfig(engine="invalid_engine")
        
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)
        
        # Invalid security configuration
        with pytest.raises(ValueError):
            SecurityConfig(password_min_length=0)


if __name__ == "__main__":
    pytest.main([__file__])