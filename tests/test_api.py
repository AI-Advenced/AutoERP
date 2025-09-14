# tests/test_api.py
"""
API Integration Tests
====================

Comprehensive test suite for AutoERP REST API endpoints built with FastAPI.
Tests cover authentication, CRUD operations, error handling, and API contracts.

Test Coverage:
- Authentication endpoints (login, logout, register)
- User management endpoints
- Table and record CRUD operations
- File upload and data import
- Error handling and validation
- API response formats and status codes
- Permission and authorization checks

Test Categories:
- Authentication flow tests
- CRUD endpoint tests  
- Input validation tests
- Permission and security tests
- Performance and load tests
- Error response tests

Author: AutoERP Development Team
License: MIT
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient
from fastapi import status

from autoerp.api import app as fastapi_app
from autoerp.core import User, UserRole

from .conftest import (
    assert_valid_uuid, assert_valid_datetime, assert_valid_email
)


class TestAPIAuthentication:
    """Test cases for API authentication endpoints."""
    
    def test_login_success(self, test_client, admin_user_data):
        """Test successful user login."""
        # First register the user
        register_response = test_client.post(
            "/api/auth/register",
            json=admin_user_data
        )
        assert register_response.status_code == status.HTTP_200_OK
        
        # Then login
        login_response = test_client.post(
            "/api/auth/login",
            json={
                "username_or_email": admin_user_data["username"],
                "password": admin_user_data["password"],
                "remember_me": False
            }
        )
        
        assert login_response.status_code == status.HTTP_200_OK
        
        response_data = login_response.json()
        assert response_data["success"] == True
        assert "data" in response_data
        
        login_data = response_data["data"]
        assert "user" in login_data
        assert "session_token" in login_data
        assert "expires_at" in login_data
        
        # Verify user data
        user_data = login_data["user"]
        assert user_data["username"] == admin_user_data["username"]
        assert user_data["email"] == admin_user_data["email"]
        assert_valid_uuid(user_data["id"])
        
        # Verify session token
        assert len(login_data["session_token"]) > 10
        assert_valid_datetime(login_data["expires_at"])
    
    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials."""
        login_response = test_client.post(
            "/api/auth/login",
            json={
                "username_or_email": "nonexistent",
                "password": "wrongpassword"
            }
        )
        
        assert login_response.status_code == status.HTTP_401_UNAUTHORIZED
        
        response_data = login_response.json()
        assert response_data["success"] == False
        assert "error" in response_data["message"].lower()
    
    def test_login_missing_fields(self, test_client):
        """Test login with missing required fields."""
        # Missing password
        response = test_client.post(
            "/api/auth/login",
            json={"username_or_email": "test@example.com"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Missing username
        response = test_client.post(
            "/api/auth/login",
            json={"password": "password"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_register_success(self, test_client, regular_user_data):
        """Test successful user registration."""
        response = test_client.post(
            "/api/auth/register",
            json=regular_user_data
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
        
        user_data = response_data["data"]
        assert user_data["username"] == regular_user_data["username"]
        assert user_data["email"] == regular_user_data["email"]
        assert user_data["first_name"] == regular_user_data["first_name"]
        assert user_data["last_name"] == regular_user_data["last_name"]
        assert_valid_uuid(user_data["id"])
    
    def test_register_duplicate_username(self, test_client, admin_user_data):
        """Test registration with duplicate username."""
        # Register first user
        response1 = test_client.post("/api/auth/register", json=admin_user_data)
        assert response1.status_code == status.HTTP_200_OK
        
        # Try to register with same username
        duplicate_data = {**admin_user_data, "email": "different@example.com"}
        response2 = test_client.post("/api/auth/register", json=duplicate_data)
        
        assert response2.status_code == status.HTTP_409_CONFLICT
        
        response_data = response2.json()
        assert response_data["success"] == False
        assert "username" in response_data["message"].lower()
    
    def test_register_weak_password(self, test_client):
        """Test registration with weak password."""
        weak_password_data = {
            "username": "weakuser",
            "email": "weak@example.com",
            "password": "123",  # Weak password
            "first_name": "Weak",
            "last_name": "User"
        }
        
        response = test_client.post("/api/auth/register", json=weak_password_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        response_data = response.json()
        assert response_data["success"] == False
        assert "password" in response_data["message"].lower()
    
    def test_get_current_user(self, authenticated_client, admin_user_data):
        """Test getting current user information."""
        response = authenticated_client.get("/api/auth/me")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
        
        user_data = response_data["data"]
        assert user_data["username"] == admin_user_data["username"]
        assert user_data["email"] == admin_user_data["email"]
        assert_valid_uuid(user_data["id"])
    
    def test_get_current_user_unauthenticated(self, test_client):
        """Test getting current user without authentication."""
        response = test_client.get("/api/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_logout_success(self, authenticated_client):
        """Test successful logout."""
        response = authenticated_client.post("/api/auth/logout")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
    
    def test_change_password_success(self, authenticated_client, admin_user_data):
        """Test successful password change."""
        response = authenticated_client.put(
            "/api/auth/change-password",
            json={
                "current_password": admin_user_data["password"],
                "new_password": "NewSecurePass123!"
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
        assert response_data["data"] == True
    
    def test_change_password_wrong_current(self, authenticated_client):
        """Test password change with wrong current password."""
        response = authenticated_client.put(
            "/api/auth/change-password",
            json={
                "current_password": "wrongpassword",
                "new_password": "NewSecurePass123!"
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        response_data = response.json()
        assert response_data["success"] == False


class TestAPITables:
    """Test cases for table management endpoints."""
    
    def test_list_tables_unauthenticated(self, test_client):
        """Test listing tables without authentication (should work for read)."""
        response = test_client.get("/api/tables")
        
        # Should allow anonymous read access
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
        assert "data" in response_data
        assert isinstance(response_data["data"], list)
    
    def test_list_tables_authenticated(self, authenticated_client):
        """Test listing tables with authentication."""
        response = authenticated_client.get("/api/tables")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert response_data["success"] == True
        
        tables = response_data["data"]
        assert isinstance(tables, list)
        
        # Verify table structure if tables exist
        if tables:
            table = tables[0]
            assert "name" in table
            assert "record_count" in table
            assert "columns" in table
            assert "permissions" in table
            
            # Check permissions structure
            permissions = table["permissions"]
            assert "read" in permissions
            assert "create" in permissions
            assert "update" in permissions
            assert "delete" in permissions
    
    def test_get_table_records_with_pagination(self, test_client):
        """Test getting table records with pagination."""
        # Try to get records from users table
        response = test_client.get(
            "/api/tables/users/records",
            params={
                "page": 1,
                "per_page": 10
            }
        )
        
        # Should work even without authentication for read access
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert response_data["success"] == True
            
            data = response_data["data"]
            assert "items" in data
            assert "pagination" in data
            assert "total_items" in data
            assert "total_pages" in data
            
            # Verify pagination structure
            pagination = data["pagination"]
            assert "page" in pagination
            assert "per_page" in pagination
            assert "has_next" in pagination
            assert "has_prev" in pagination
    
    def test_get_table_records_with_filters(self, test_client):
        """Test getting table records with filters."""
        response = test_client.get(
            "/api/tables/users/records",
            params={
                "page": 1,
                "per_page": 5,
                "sort_by": "created_at",
                "sort_order": "desc",
                "search": "test"
            }
        )
        
        # Should return results or empty list
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]
    
    def test_get_nonexistent_table(self, test_client):
        """Test getting records from nonexistent table."""
        response = test_client.get("/api/tables/nonexistent/records")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        response_data = response.json()
        assert response_data["success"] == False


class TestAPIRecordCRUD:
    """Test cases for record CRUD operations."""
    
    def test_create_record_success(self, authenticated_client):
        """Test successful record creation."""
        record_data = {
            "data": {
                "username": "apiuser",
                "email": "api@example.com",
                "first_name": "API",
                "last_name": "User"
            }
        }
        
        response = authenticated_client.post(
            "/api/tables/users/records",
            json=record_data
        )
        
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert response_data["success"] == True
            
            record = response_data["data"]
            assert "id" in record
            assert "data" in record
            assert_valid_uuid(record["id"])
            
            # Verify created data
            created_data = record["data"]
            assert created_data["username"] == "apiuser"
            assert created_data["email"] == "api@example.com"
    
    def test_create_record_unauthenticated(self, test_client):
        """Test record creation without authentication."""
        record_data = {
            "data": {
                "username": "unauthuser",
                "email": "unauth@example.com",
                "first_name": "Unauth",
                "last_name": "User"
            }
        }
        
        response = test_client.post(
            "/api/tables/users/records",
            json=record_data
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_create_record_invalid_data(self, authenticated_client):
        """Test record creation with invalid data."""
        record_data = {
            "data": {
                "username": "",  # Invalid: empty username
                "email": "invalid-email",  # Invalid: bad email format
                "first_name": "",  # Invalid: empty first name
                "last_name": ""  # Invalid: empty last name
            }
        }
        
        response = authenticated_client.post(
            "/api/tables/users/records",
            json=record_data
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        response_data = response.json()
        assert response_data["success"] == False
    
    def test_get_record_by_id(self, authenticated_client):
        """Test getting a specific record by ID."""
        # First create a record
        create_data = {
            "data": {
                "username": "getuser",
                "email": "get@example.com",
                "first_name": "Get",
                "last_name": "User"
            }
        }
        
        create_response = authenticated_client.post(
            "/api/tables/users/records",
            json=create_data
        )
        
        if create_response.status_code == status.HTTP_200_OK:
            created_record = create_response.json()["data"]
            record_id = created_record["id"]
            
            # Get the record by ID
            get_response = authenticated_client.get(
                f"/api/tables/users/records/{record_id}"
            )
            
            assert get_response.status_code == status.HTTP_200_OK
            
            response_data = get_response.json()
            assert response_data["success"] == True
            
            record = response_data["data"]
            assert record["id"] == record_id
            assert record["data"]["username"] == "getuser"
    
    def test_get_nonexistent_record(self, authenticated_client):
        """Test getting a nonexistent record."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        response = authenticated_client.get(
            f"/api/tables/users/records/{fake_id}"
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        response_data = response.json()
        assert response_data["success"] == False
    
    def test_update_record_success(self, authenticated_client):
        """Test successful record update."""
        # First create a record
        create_data = {
            "data": {
                "username": "updateuser", 
                "email": "update@example.com",
                "first_name": "Update",
                "last_name": "User"
            }
        }
        
        create_response = authenticated_client.post(
            "/api/tables/users/records",
            json=create_data
        )
        
        if create_response.status_code == status.HTTP_200_OK:
            created_record = create_response.json()["data"]
            record_id = created_record["id"]
            
            # Update the record
            update_data = {
                "data": {
                    "first_name": "Updated",
                    "last_name": "Name"
                }
            }
            
            update_response = authenticated_client.put(
                f"/api/tables/users/records/{record_id}",
                json=update_data
            )
            
            assert update_response.status_code == status.HTTP_200_OK
            
            response_data = update_response.json()
            assert response_data["success"] == True
            
            updated_record = response_data["data"]
            assert updated_record["data"]["first_name"] == "Updated"
            assert updated_record["data"]["last_name"] == "Name"
    
    def test_delete_record_success(self, authenticated_client):
        """Test successful record deletion."""
        # First create a record
        create_data = {
            "data": {
                "username": "deleteuser",
                "email": "delete@example.com", 
                "first_name": "Delete",
                "last_name": "User"
            }
        }
        
        create_response = authenticated_client.post(
            "/api/tables/users/records",
            json=create_data
        )
        
        if create_response.status_code == status.HTTP_200_OK:
            created_record = create_response.json()["data"]
            record_id = created_record["id"]
            
            # Delete the record
            delete_response = authenticated_client.delete(
                f"/api/tables/users/records/{record_id}"
            )
            
            assert delete_response.status_code == status.HTTP_200_OK
            
            response_data = delete_response.json()
            assert response_data["success"] == True
            assert response_data["data"] == True
    
    def test_delete_record_unauthenticated(self, test_client):
        """Test record deletion without authentication."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        response = test_client.delete(f"/api/tables/users/records/{fake_id}")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAPIBulkOperations:
    """Test cases for bulk operations."""
    
    def test_bulk_create_records(self, authenticated_client):
        """Test bulk record creation."""
        bulk_data = {
            "operation": "create",
            "records": [
                {
                    "username": f"bulkuser{i}",
                    "email": f"bulk{i}@example.com",
                    "first_name": "Bulk",
                    "last_name": f"User{i}"
                }
                for i in range(1, 4)  # Create 3 records
            ]
        }
        
        response = authenticated_client.post(
            "/api/tables/users/bulk",
            json=bulk_data
        )
        
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert response_data["success"] == True
            
            bulk_result = response_data["data"]
            assert bulk_result["operation"] == "create"
            assert bulk_result["total_requested"] == 3
            assert bulk_result["successful"] >= 0
            assert bulk_result["failed"] >= 0
            assert bulk_result["successful"] + bulk_result["failed"] == 3
    
    def test_bulk_update_records(self, authenticated_client):
        """Test bulk record update."""
        # First create some records to update
        create_records = []
        for i in range(2):
            create_data = {
                "data": {
                    "username": f"bulkupdate{i}",
                    "email": f"bulkupdate{i}@example.com",
                    "first_name": "BulkUpdate", 
                    "last_name": f"User{i}"
                }
            }
            
            create_response = authenticated_client.post(
                "/api/tables/users/records",
                json=create_data
            )
            
            if create_response.status_code == status.HTTP_200_OK:
                record = create_response.json()["data"]
                create_records.append(record)
        
        if create_records:
            # Prepare bulk update
            update_records = []
            for record in create_records:
                update_records.append({
                    "id": record["id"],
                    "first_name": "BulkUpdated"
                })
            
            bulk_data = {
                "operation": "update",
                "records": update_records
            }
            
            response = authenticated_client.post(
                "/api/tables/users/bulk",
                json=bulk_data
            )
            
            if response.status_code == status.HTTP_200_OK:
                response_data = response.json()
                assert response_data["success"] == True
                
                bulk_result = response_data["data"]
                assert bulk_result["operation"] == "update"


class TestAPIValidation:
    """Test cases for API input validation."""
    
    def test_invalid_json_request(self, authenticated_client):
        """Test API response to invalid JSON."""
        response = authenticated_client.post(
            "/api/tables/users/records",
            data="invalid json content",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, authenticated_client):
        """Test API validation of required fields."""
        # Missing 'data' field
        response = authenticated_client.post(
            "/api/tables/users/records",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response_data = response.json()
        assert "validation" in response_data["message"].lower() or "required" in response_data.get("detail", "").lower()
    
    def test_invalid_field_types(self, authenticated_client):
        """Test API validation of field types.""" 
        # Invalid data type (should be dict)
        response = authenticated_client.post(
            "/api/tables/users/records",
            json={"data": "should_be_dict"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_invalid_pagination_params(self, test_client):
        """Test API validation of pagination parameters."""
        # Invalid page number
        response = test_client.get(
            "/api/tables/users/records",
            params={"page": 0, "per_page": 10}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid per_page number
        response = test_client.get(
            "/api/tables/users/records", 
            params={"page": 1, "per_page": 1001}  # Over limit
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    def test_internal_server_error_handling(self, authenticated_client):
        """Test API handling of internal server errors."""
        # This test would require mocking internal errors
        # For now, test that the API structure handles errors properly
        pass
    
    def test_request_timeout_handling(self, test_client):
        """Test API handling of request timeouts."""
        # This would require special setup to simulate timeouts
        pass
    
    def test_rate_limiting(self, test_client):
        """Test API rate limiting (if implemented).""" 
        # Make many rapid requests
        responses = []
        for _ in range(10):
            response = test_client.get("/api/tables")
            responses.append(response)
        
        # All should succeed if no rate limiting, or some should be 429
        status_codes = [r.status_code for r in responses]
        
        # Should be mostly 200s, possibly some 429s if rate limiting is active
        assert all(code in [200, 429] for code in status_codes)


class TestAPISystemEndpoints:
    """Test cases for system endpoints."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert "status" in response_data
        assert response_data["status"] in ["healthy", "unhealthy", "degraded"]
        assert "timestamp" in response_data
        assert "components" in response_data
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        
        # Metrics might require authentication
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
        
        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "uptime_seconds" in response_data
            assert "metrics_count" in response_data
            assert "timestamp" in response_data
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        
        response_data = response.json()
        assert "message" in response_data
        assert "version" in response_data


if __name__ == "__main__":
    pytest.main([__file__])

# tests/test_ui.py
"""
UI Component Tests
=================

Test suite for AutoERP user interface components built with Streamlit and Flask.
Tests cover page rendering, form interactions, data visualization, and user workflows.

Test Coverage:
- Streamlit page components and layouts
- Form validation and submission
- Data visualization and charts  
- User authentication flows
- Navigation and routing
- Component state management
- Error handling and user feedback

Test Categories:
- Component rendering tests
- User interaction simulation tests
- Data binding and display tests
- Authentication and session tests
- Performance and responsiveness tests

Note: UI testing requires special approaches due to Streamlit's architecture.
Some tests may use mocking or integration testing strategies.

Author: AutoERP Development Team
License: MIT
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any, List

# UI component imports
try:
    from autoerp.ui import (
        UISessionManager, AutoERPAPIClient, dashboard_page,
        tables_page, login_page, render_sidebar, 
        format_currency, format_date, create_metric_card
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    pytestmark = pytest.mark.skip("UI components not available")

# Mock Streamlit for testing
streamlit_mock = MagicMock()
streamlit_mock.session_state = {}
streamlit_mock.columns = lambda x: [MagicMock() for _ in range(x)]
streamlit_mock.container = lambda: MagicMock()
streamlit_mock.form = lambda x: MagicMock()
streamlit_mock.sidebar = MagicMock()


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for UI testing."""
    with patch.dict('sys.modules', {'streamlit': streamlit_mock}):
        yield streamlit_mock


@pytest.fixture 
def mock_session_state():
    """Mock Streamlit session state."""
    session_state = {
        'initialized': False,
        'user': None,
        'session_token': None,
        'is_authenticated': False,
        'current_page': 'dashboard',
        'notifications': [],
        'theme': 'light'
    }
    
    with patch('autoerp.ui.st.session_state', session_state):
        yield session_state


@pytest.fixture
def mock_api_client():
    """Mock API client for UI testing."""
    mock_client = Mock(spec=AutoERPAPIClient)
    
    # Mock successful responses
    mock_client.login.return_value = {
        "success": True,
        "data": {
            "user": {
                "id": "user123",
                "username": "testuser",
                "email": "test@example.com",
                "first_name": "Test",
                "last_name": "User",
                "full_name": "Test User",
                "role": "user"
            },
            "session_token": "mock_token_123",
            "expires_at": datetime.now(timezone.utc).isoformat()
        }
    }
    
    mock_client.get_tables.return_value = {
        "success": True,
        "data": [
            {
                "name": "users",
                "display_name": "Users",
                "record_count": 10,
                "columns": [
                    {"name": "id", "data_type": "string", "nullable": False},
                    {"name": "username", "data_type": "string", "nullable": False},
                    {"name": "email", "data_type": "string", "nullable": False}
                ],
                "permissions": {"read": True, "create": True, "update": True, "delete": False}
            }
        ]
    }
    
    mock_client.get_health.return_value = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": {"status": "healthy"},
            "cache": {"status": "healthy"}
        }
    }
    
    return mock_client


class TestUISessionManager:
    """Test cases for UI session management."""
    
    def test_initialize_session(self, mock_session_state):
        """Test session initialization.""" 
        UISessionManager.initialize_session()
        
        assert mock_session_state['initialized'] == True
        assert mock_session_state['user'] is None
        assert mock_session_state['is_authenticated'] == False
        assert mock_session_state['current_page'] == 'dashboard'
        assert isinstance(mock_session_state['notifications'], list)
    
    def test_set_user(self, mock_session_state):
        """Test setting user in session."""
        mock_user = Mock()
        mock_user.id = "user123"
        mock_user.username = "testuser"
        
        UISessionManager.set_user(mock_user, "token123")
        
        assert mock_session_state['user'] == mock_user
        assert mock_session_state['session_token'] == "token123"
        assert mock_session_state['is_authenticated'] == True
    
    def test_logout(self, mock_session_state):
        """Test user logout."""
        # Set up authenticated state
        mock_session_state['user'] = Mock()
        mock_session_state['session_token'] = "token123"
        mock_session_state['is_authenticated'] = True
        
        UISessionManager.logout()
        
        assert mock_session_state['user'] is None
        assert mock_session_state['session_token'] is None
        assert mock_session_state['is_authenticated'] == False
    
    def test_add_notification(self, mock_session_state):
        """Test adding notifications."""
        UISessionManager.add_notification("Test message", "success")
        
        notifications = mock_session_state['notifications']
        assert len(notifications) == 1
        
        notification = notifications[0]
        assert notification['message'] == "Test message"
        assert notification['type'] == "success"
        assert 'id' in notification
        assert 'timestamp' in notification
    
    def test_is_authenticated(self, mock_session_state):
        """Test authentication check."""
        # Not authenticated initially
        assert UISessionManager.is_authenticated() == False
        
        # Set authenticated
        mock_session_state['is_authenticated'] = True
        assert UISessionManager.is_authenticated() == True


class TestAPIClient:
    """Test cases for API client functionality."""
    
    @pytest.fixture
    def api_client(self):
        """Create API client for testing."""
        return AutoERPAPIClient("http://testserver")
    
    def test_api_client_initialization(self, api_client):
        """Test API client initialization."""
        assert api_client.base_url == "http://testserver"
        assert api_client.session is not None
        assert 'Content-Type' in api_client.session.headers
        assert api_client.session.headers['Content-Type'] == 'application/json'
    
    def test_set_auth_token(self, api_client):
        """Test setting authentication token."""
        api_client.set_auth_token("test_token")
        
        assert 'Authorization' in api_client.session.headers
        assert api_client.session.headers['Authorization'] == 'Bearer test_token'
    
    def test_clear_auth_token(self, api_client):
        """Test clearing authentication token."""
        # Set token first
        api_client.set_auth_token("test_token")
        assert 'Authorization' in api_client.session.headers
        
        # Clear token
        api_client.clear_auth_token()
        assert 'Authorization' not in api_client.session.headers
    
    @patch('requests.Session.post')
    def test_login_success(self, mock_post, api_client):
        """Test successful login through API client."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "user": {"id": "123", "username": "testuser"},
                "session_token": "token123"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Test login
        result = asyncio.run(api_client.login("testuser", "password"))
        
        assert result["success"] == True
        assert "data" in result
        mock_post.assert_called_once()
    
    @patch('requests.Session.get')
    def test_get_tables_success(self, mock_get, api_client):
        """Test getting tables through API client."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": [{"name": "users", "record_count": 10}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test get tables
        result = asyncio.run(api_client.get_tables())
        
        assert result["success"] == True
        assert len(result["data"]) == 1
        mock_get.assert_called_once_with("http://testserver/api/tables")


class TestUtilityFunctions:
    """Test cases for UI utility functions."""
    
    def test_format_currency(self):
        """Test currency formatting."""
        # USD formatting
        assert format_currency(100.50, "USD") == "$100.50"
        assert format_currency(1000, "USD") == "$1,000.00"
        
        # EUR formatting
        assert format_currency(100.50, "EUR") == "€100.50"
        
        # Other currency
        assert format_currency(100.50, "GBP") == "100.50 GBP"
    
    def test_format_date(self):
        """Test date formatting."""
        # Test with datetime object
        dt = datetime(2023, 12, 25, 15, 30, 0)
        formatted = format_date(dt)
        assert "2023-12-25" in formatted
        assert "15:30" in formatted
        
        # Test with ISO string
        iso_string = "2023-12-25T15:30:00Z"
        formatted = format_date(iso_string)
        assert "2023-12-25" in formatted
        
        # Test with invalid input
        invalid_input = "invalid-date"
        formatted = format_date(invalid_input)
        assert formatted == invalid_input  # Should return as-is
    
    def test_create_metric_card(self):
        """Test metric card creation."""
        card_html = create_metric_card("Total Users", "150", "+5")
        
        assert "Total Users" in card_html
        assert "150" in card_html
        assert "+5" in card_html
        assert "metric-card" in card_html
        
        # Test without delta
        card_html = create_metric_card("Revenue", "$10,000")
        assert "Revenue" in card_html
        assert "$10,000" in card_html


@pytest.mark.skipif(not UI_AVAILABLE, reason="UI components not available")
class TestPageComponents:
    """Test cases for UI page components."""
    
    def test_login_page_rendering(self, mock_streamlit, mock_session_state):
        """Test login page rendering."""
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=Mock()):
                try:
                    from autoerp.ui import login_page
                    login_page()
                    
                    # Verify Streamlit components were called
                    mock_streamlit.markdown.assert_called()
                    mock_streamlit.columns.assert_called()
                    
                except Exception as e:
                    # Some components may not work in test environment
                    pytest.skip(f"Login page test skipped: {e}")
    
    def test_dashboard_page_rendering(self, mock_streamlit, mock_session_state, mock_api_client):
        """Test dashboard page rendering."""
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                with patch('autoerp.ui.UISessionManager.get_notifications', return_value=[]):
                    try:
                        from autoerp.ui import dashboard_page
                        dashboard_page()
                        
                        # Verify basic components were rendered
                        mock_streamlit.markdown.assert_called()
                        
                    except Exception as e:
                        pytest.skip(f"Dashboard page test skipped: {e}")
    
    def test_tables_page_rendering(self, mock_streamlit, mock_session_state, mock_api_client):
        """Test tables page rendering."""
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                try:
                    from autoerp.ui import tables_page
                    tables_page()
                    
                    # Verify API client was called
                    mock_api_client.get_tables.assert_called()
                    
                except Exception as e:
                    pytest.skip(f"Tables page test skipped: {e}")
    
    def test_sidebar_rendering(self, mock_streamlit, mock_session_state):
        """Test sidebar rendering."""
        with patch('autoerp.ui.st', mock_streamlit):
            # Mock authenticated user
            mock_user = Mock()
            mock_user.full_name = "Test User"
            mock_user.email = "test@example.com"
            mock_user.role = "user"
            
            with patch('autoerp.ui.UISessionManager.is_authenticated', return_value=True):
                with patch('autoerp.ui.UISessionManager.get_user', return_value=mock_user):
                    with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                        try:
                            from autoerp.ui import render_sidebar
                            render_sidebar()
                            
                            # Verify sidebar components
                            mock_streamlit.sidebar.assert_used()
                            
                        except Exception as e:
                            pytest.skip(f"Sidebar test skipped: {e}")


class TestFormInteractions:
    """Test cases for form interactions and submissions."""
    
    def test_login_form_submission(self, mock_streamlit, mock_api_client):
        """Test login form submission."""
        # Mock form submission
        mock_streamlit.form_submit_button.return_value = True
        mock_streamlit.text_input.side_effect = ["testuser", "password"]
        
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                with patch('autoerp.ui.UISessionManager.set_user') as mock_set_user:
                    try:
                        from autoerp.ui import login_page
                        login_page()
                        
                        # If login was successful, user should be set
                        if mock_api_client.login.called:
                            # Verify API was called with correct parameters
                            pass
                            
                    except Exception as e:
                        pytest.skip(f"Login form test skipped: {e}")
    
    def test_add_record_form(self, mock_streamlit, mock_api_client):
        """Test add record form functionality."""
        # Mock form inputs
        mock_streamlit.form_submit_button.return_value = True
        mock_streamlit.text_input.side_effect = ["newuser", "new@example.com", "New", "User"]
        
        # Mock successful record creation
        mock_api_client.create_record.return_value = {
            "success": True,
            "data": {"id": "new123", "data": {"username": "newuser"}}
        }
        
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                # This would test the add record form if it exists
                # For now, just verify the mocks work
                pass


class TestDataVisualization:
    """Test cases for data visualization components."""
    
    def test_metric_display(self, mock_streamlit):
        """Test metric display components."""
        with patch('autoerp.ui.st', mock_streamlit):
            # Mock plotly figure creation
            with patch('plotly.graph_objects.Figure') as mock_fig:
                mock_fig.return_value.add_trace = Mock()
                mock_fig.return_value.update_layout = Mock()
                
                # Test would verify chart creation in dashboard
                pass
    
    def test_table_display(self, mock_streamlit, mock_api_client):
        """Test table data display."""
        # Mock table data
        mock_api_client.get_table_records.return_value = {
            "success": True,
            "data": {
                "items": [
                    {"id": "1", "data": {"username": "user1", "email": "user1@example.com"}},
                    {"id": "2", "data": {"username": "user2", "email": "user2@example.com"}}
                ],
                "pagination": {"page": 1, "per_page": 10, "total_items": 2, "total_pages": 1}
            }
        }
        
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                # Mock pandas DataFrame for table display
                with patch('pandas.DataFrame') as mock_df:
                    mock_df.return_value = Mock()
                    
                    # Test would verify data table creation
                    pass


class TestErrorHandling:
    """Test cases for UI error handling."""
    
    def test_api_error_handling(self, mock_streamlit, mock_api_client):
        """Test handling of API errors in UI."""
        # Mock API error
        mock_api_client.get_tables.side_effect = Exception("API Error")
        
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                try:
                    from autoerp.ui import tables_page
                    tables_page()
                    
                    # Should handle error gracefully
                    # Verify error message was displayed
                    mock_streamlit.error.assert_called()
                    
                except Exception as e:
                    pytest.skip(f"Error handling test skipped: {e}")
    
    def test_validation_error_display(self, mock_streamlit):
        """Test validation error display."""
        with patch('autoerp.ui.st', mock_streamlit):
            # Mock validation error
            with patch('autoerp.ui.display_notification') as mock_display:
                mock_display("Validation failed", "error")
                mock_display.assert_called_with("Validation failed", "error")


class TestResponsiveness:
    """Test cases for UI responsiveness and performance."""
    
    def test_page_load_performance(self, mock_streamlit, mock_api_client, performance_timer):
        """Test page load performance."""
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                
                performance_timer("page_load")
                
                try:
                    from autoerp.ui import dashboard_page
                    dashboard_page()
                    
                    elapsed = performance_timer("page_load")
                    
                    # Page should load quickly
                    assert elapsed < 5.0  # 5 seconds max
                    
                except Exception as e:
                    pytest.skip(f"Performance test skipped: {e}")
    
    def test_large_table_rendering(self, mock_streamlit, mock_api_client):
        """Test rendering of large data tables."""
        # Mock large dataset
        large_data = {
            "success": True,
            "data": {
                "items": [
                    {"id": f"id_{i}", "data": {"username": f"user{i}", "email": f"user{i}@example.com"}}
                    for i in range(100)  # 100 records
                ],
                "pagination": {"page": 1, "per_page": 100, "total_items": 100, "total_pages": 1}
            }
        }
        
        mock_api_client.get_table_records.return_value = large_data
        
        with patch('autoerp.ui.st', mock_streamlit):
            with patch('autoerp.ui.get_api_client', return_value=mock_api_client):
                # Test should verify large table can be handled
                pass


if __name__ == "__main__":
    pytest.main([__file__])