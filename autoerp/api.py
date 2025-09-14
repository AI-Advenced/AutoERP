"""
AutoERP API Module - FastAPI REST API Implementation
Version: 1.0.0
Author: AutoERP Development Team
License: MIT

This module implements the REST API layer for the AutoERP system using FastAPI.
It provides HTTP endpoints for all system functionality with proper authentication,
validation, and error handling.
"""

import asyncio
import json
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import (
    T, Dict, List, Optional, Union, Callable, Type, 
    Annotated, Literal, get_type_hints
)
from pathlib import Path

# FastAPI and related imports
from fastapi import (
    FastAPI, Request, Response, Depends, HTTPException, 
    status, Header, Query, Path as PathParam, Body,
    BackgroundTasks, File, UploadFile, Form
)
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.encoders import jsonable_encoder
from typing import Generic, T
from pydantic.generics import GenericModel

# Pydantic imports
from pydantic import (
    BaseModel, Field, validator, root_validator, 
    ValidationError as PydanticValidationError,
    EmailStr, HttpUrl, constr, conint, confloat
)
from pydantic.generics import GenericModel

# Starlette imports
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR

# Standard library imports
import jwt
import secrets
import hashlib
import base64
from urllib.parse import unquote
import mimetypes
import io


from typing import Any, Dict, List, Optional, TypeVar, Generic, Tuple
from datetime import datetime, timezone
from pydantic import Field
from pydantic.generics import GenericModel

# Core module imports
from .core import (
    AutoERPApplication, AutoERPConfig, User, UserRole, Session,
    ServiceResult, UserService, NotificationService, CRUDService,
    ValidationError as CoreValidationError, RepositoryError,
    RecordNotFoundError, DuplicateRecordError, ConcurrencyError,
    BusinessRuleViolationError, AuthorizationError,
    PaginationInfo, FilterCriteria, BaseModel as CoreBaseModel,
    system_metrics, timed_operation
)

# Configure logging
logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS AND SCHEMAS ====================



T = TypeVar("T")  # تعريف TypeVar لاستخدامه في Generic

class APIResponse(GenericModel, Generic[T]):
    """Generic API response wrapper."""

    success: bool = Field(..., description="Whether the operation was successful")
    data: Optional[T] = Field(None, description="Response data")  # استخدم T بدلاً من T
    message: Optional[str] = Field(None, description="Response message")
    error_code: Optional[str] = Field(None, description="Error code if operation failed")
    metadata: Optional[Dict[str, T]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(None, description="Request tracking ID")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def success_response(
        cls,
        data: Optional[T] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, T]] = None,
        request_id: Optional[str] = None
    ) -> 'APIResponse[T]':
        """Create a success response."""
        return cls(
            success=True,
            data=data,
            message=message,
            metadata=metadata,
            request_id=request_id
        )

    @classmethod
    def error_response(
        cls,
        message: str,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, T]] = None,
        request_id: Optional[str] = None
    ) -> 'APIResponse[T]':
        """Create an error response."""
        return cls(
            success=False,
            message=message,
            error_code=error_code,
            metadata=metadata,
            request_id=request_id
        )


class PaginationRequest(BaseModel):
    """Pagination request parameters."""
    
    page: conint(ge=1) = Field(1, description="Page number (1-based)")
    per_page: conint(ge=1, le=1000) = Field(50, description="Items per page")
    
    def to_pagination_info(self) -> PaginationInfo:
        """Convert to core PaginationInfo."""
        return PaginationInfo(page=self.page, per_page=self.per_page)


class SortRequest(BaseModel):
    """Sort request parameters."""
    
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Literal["asc", "desc"] = Field("asc", description="Sort order")


class FilterRequest(BaseModel):
    """Filter request parameters."""
    
    filters: Optional[Dict[str, T]] = Field(None, description="Field filters")
    search: Optional[str] = Field(None, description="Search query")
    search_fields: Optional[List[str]] = Field(None, description="Fields to search in")
    
    def to_filter_criteria(self, sort_request: Optional[SortRequest] = None) -> FilterCriteria:
        """Convert to core FilterCriteria."""
        criteria = FilterCriteria()
        
        if self.filters:
            for field, value in self.filters.items():
                if isinstance(value, dict) and 'operator' in value:
                    criteria.add_filter(field, value.get('value'), value.get('operator', 'eq'))
                else:
                    criteria.add_filter(field, value)
        
        if self.search and self.search_fields:
            criteria.set_search(self.search, self.search_fields)
        
        if sort_request and sort_request.sort_by:
            criteria.set_sort(sort_request.sort_by, sort_request.sort_order)
        
        return criteria


class PaginatedResponse(GenericModel, Generic[T]):
    """Paginated response wrapper."""
    
    items: List[T] = Field(..., description="List of items")
    pagination: Dict[str, T] = Field(..., description="Pagination information")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")

    @classmethod
    def create(cls, items: List[T], pagination: PaginationInfo) -> 'PaginatedResponse':
        """Create paginated response from items and pagination info."""
        return cls(
            items=items,
            pagination=pagination.to_dict(),
            total_items=pagination.total_items or 0,
            total_pages=pagination.total_pages or 0
        )


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: Literal["healthy", "unhealthy", "degraded"] = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: Dict[str, Dict[str, T]] = Field(..., description="Component health details")
    uptime_seconds: Optional[float] = Field(None, description="Application uptime in seconds")
    version: Optional[str] = Field(None, description="Application version")


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    uptime_seconds: float = Field(..., description="Application uptime")
    metrics_count: int = Field(..., description="Number of metrics collected")
    metrics: Dict[str, Dict[str, T]] = Field(..., description="Metrics data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== USER AND AUTHENTICATION SCHEMAS ====================

class UserCreateRequest(BaseModel):
    """User creation request."""
    
    username: constr(min_length=3, max_length=50) = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: constr(min_length=8) = Field(..., description="Password")
    first_name: constr(max_length=50) = Field(..., description="First name")
    last_name: constr(max_length=50) = Field(..., description="Last name")
    role: Optional[UserRole] = Field(UserRole.USER, description="User role")
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Username can only contain letters, numbers, underscore, and dash")
        return v


class UserUpdateRequest(BaseModel):
    """User update request."""
    
    first_name: Optional[constr(max_length=50)] = Field(None, description="First name")
    last_name: Optional[constr(max_length=50)] = Field(None, description="Last name")
    email: Optional[EmailStr] = Field(None, description="Email address")
    preferences: Optional[Dict[str, T]] = Field(None, description="User preferences")


class UserResponse(BaseModel):
    """User response."""
    
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    full_name: str = Field(..., description="Full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Whether user is active")
    is_verified: bool = Field(..., description="Whether email is verified")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        orm_mode = True
        
    @classmethod
    def from_user(cls, user: User) -> 'UserResponse':
        """Create response from User entity."""
        return cls(
            id=user.id,
            username=user.username,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at
        )


class LoginRequest(BaseModel):
    """Login request."""
    
    username_or_email: str = Field(..., description="Username or email address")
    password: str = Field(..., description="Password")
    remember_me: bool = Field(False, description="Remember login session")


class LoginResponse(BaseModel):
    """Login response."""
    
    user: UserResponse = Field(..., description="User information")
    session_token: str = Field(..., description="Session token")
    expires_at: datetime = Field(..., description="Session expiration time")
    
    @classmethod
    def from_user_session(cls, user: User, session: Session) -> 'LoginResponse':
        """Create response from user and session."""
        return cls(
            user=UserResponse.from_user(user),
            session_token=session.session_token,
            expires_at=session.expires_at
        )


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    
    current_password: str = Field(..., description="Current password")
    new_password: constr(min_length=8) = Field(..., description="New password")


# ==================== TABLE AND RECORD SCHEMAS ====================

class TableInfo(BaseModel):
    """Table information."""
    
    name: str = Field(..., description="Table name")
    display_name: Optional[str] = Field(None, description="Human-readable table name")
    description: Optional[str] = Field(None, description="Table description")
    record_count: int = Field(..., description="Number of records in table")
    columns: List[Dict[str, T]] = Field(..., description="Column definitions")
    created_at: Optional[datetime] = Field(None, description="Table creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    permissions: Dict[str, bool] = Field(..., description="User permissions for this table")


class ColumnInfo(BaseModel):
    """Column information."""
    
    name: str = Field(..., description="Column name")
    data_type: str = Field(..., description="Data type")
    nullable: bool = Field(..., description="Whether column allows null values")
    primary_key: bool = Field(False, description="Whether column is primary key")
    unique: bool = Field(False, description="Whether column has unique constraint")
    max_length: Optional[int] = Field(None, description="Maximum length for string columns")
    default_value: Optional[T] = Field(None, description="Default value")
    description: Optional[str] = Field(None, description="Column description")


class RecordCreateRequest(BaseModel):
    """Record creation request."""
    
    data: Dict[str, T] = Field(..., description="Record data")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate record data is not empty."""
        if not v:
            raise ValueError("Record data cannot be empty")
        return v


class RecordUpdateRequest(BaseModel):
    """Record update request."""
    
    data: Dict[str, T] = Field(..., description="Fields to update")
    
    @validator('data')
    def validate_data(cls, v):
        """Validate update data is not empty."""
        if not v:
            raise ValueError("Update data cannot be empty")
        return v


class RecordResponse(BaseModel):
    """Record response."""
    
    id: str = Field(..., description="Record ID")
    data: Dict[str, T] = Field(..., description="Record data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Created by user")
    updated_by: Optional[str] = Field(None, description="Last updated by user")
    version: Optional[int] = Field(None, description="Record version")


class BulkOperationRequest(BaseModel):
    """Bulk operation request."""
    
    operation: Literal["create", "update", "delete"] = Field(..., description="Operation type")
    records: List[Dict[str, T]] = Field(..., description="Records to process")
    options: Optional[Dict[str, T]] = Field(None, description="Operation options")


class BulkOperationResponse(BaseModel):
    """Bulk operation response."""
    
    operation: str = Field(..., description="Operation type")
    total_requested: int = Field(..., description="Total records requested")
    successful: int = Field(..., description="Successfully processed records")
    failed: int = Field(..., description="Failed records")
    errors: List[Dict[str, T]] = Field(..., description="Error details")
    results: List[Dict[str, T]] = Field(..., description="Operation results")


# ==================== ERROR SCHEMAS ====================

class ErrorDetail(BaseModel):
    """Error detail."""
    
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    
    success: bool = Field(False)
    message: str = Field(..., description="Main error message")
    error_code: str = Field("VALIDATION_ERROR")
    errors: List[ErrorDetail] = Field(..., description="Detailed validation errors")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==================== REQUEST CONTEXT AND MIDDLEWARE ====================

class RequestContext:
    """Request context information."""
    
    def __init__(self, request: Request):
        self.request_id = str(uuid.uuid4())
        self.start_time = datetime.now(timezone.utc)
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.user: Optional[User] = None
        self.session: Optional[Session] = None
        self.ip_address = self._get_client_ip(request)
        self.user_agent = request.headers.get("user-agent", "")
        self.path = request.url.path
        self.method = request.method
        self.is_authenticated = False
        self.is_anonymous = True
        self.permissions: List[str] = []
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers (for reverse proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        return getattr(request.client, "host", "unknown")
    
    def set_user(self, user: User, session: Optional[Session] = None) -> None:
        """Set authenticated user."""
        self.user = user
        self.user_id = user.id
        self.is_authenticated = True
        self.is_anonymous = False
        self.permissions = user.permissions
        
        if session:
            self.session = session
            self.session_id = session.id
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission."""
        return permission in self.permissions or self.user.role == UserRole.SUPER_ADMIN
    
    def require_permission(self, permission: str) -> None:
        """Require user to have permission."""
        if not self.is_authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not self.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )


# Global request context storage
_request_context: Optional[RequestContext] = None


def get_request_context() -> Optional[RequestContext]:
    """Get current request context."""
    return _request_context


def set_request_context(context: RequestContext) -> None:
    """Set current request context."""
    global _request_context
    _request_context = context


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to manage request context."""
    
    async def dispatch(
        self, 
        request: StarletteRequest, 
        call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        """Process request with context."""
        context = RequestContext(request)
        set_request_context(context)
        
        try:
            # Add request ID to response headers
            response = await call_next(request)
            response.headers["X-Request-ID"] = context.request_id
            
            # Record metrics
            duration_ms = (datetime.now(timezone.utc) - context.start_time).total_seconds() * 1000
            system_metrics.record_timing(
                "http_request_duration",
                duration_ms,
                {
                    "method": context.method,
                    "path": context.path,
                    "status_code": str(response.status_code)
                }
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            system_metrics.increment_counter(
                "http_request_errors",
                {
                    "method": context.method,
                    "path": context.path,
                    "error_type": type(e).__name__
                }
            )
            raise
        finally:
            # Clear context
            set_request_context(None)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware with JWT support."""
    
    def __init__(self, app, app_instance: AutoERPApplication):
        super().__init__(app)
        self.app_instance = app_instance
        self.security_config = app_instance.config.security
        
        # Paths that don't require authentication
        self.public_paths = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/forgot-password",
            "/api/auth/reset-password"
        }
        
        # Paths that allow anonymous access
        self.anonymous_allowed_paths = {
            "/api/tables",  # Read-only table listing
            "/api/public/"  # Public API endpoints
        }
    
    async def dispatch(
        self,
        request: StarletteRequest,
        call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        """Process authentication for request."""
        context = get_request_context()
        if not context:
            return await call_next(request)
        
        path = request.url.path
        
        # Skip authentication for public paths
        if path in self.public_paths or path.startswith("/static/"):
            return await call_next(request)
        
        # Try to authenticate user
        try:
            await self._authenticate_request(request, context)
        except HTTPException:
            # Check if anonymous access is allowed for this path
            if T(path.startswith(anonymous_path) for anonymous_path in self.anonymous_allowed_paths):
                # Allow anonymous access
                pass
            else:
                # Re-raise authentication error
                raise
        
        return await call_next(request)
    
    async def _authenticate_request(self, request: StarletteRequest, context: RequestContext) -> None:
        """Authenticate the request."""
        # Try session token first (from headers or cookies)
        session_token = None
        
        # Check Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header[7:]  # Remove "Bearer " prefix
        
        # Check session cookie
        if not session_token:
            session_token = request.cookies.get("session_token")
        
        # Check X-Session-Token header
        if not session_token:
            session_token = request.headers.get("x-session-token")
        
        if not session_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required"
            )
        
        # Validate session token
        session = await self.app_instance.session_manager.get_session(session_token)
        if not session or not session.is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session token"
            )
        
        # Get user
        user_repository = self.app_instance.get_repository(User)
        user = await user_repository.get_by_id(session.user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive"
            )
        
        # Update session activity
        await self.app_instance.session_manager.update_session_activity(session_token)
        
        # Set user context
        context.set_user(user, session)
        
        # Set service context for all services
        if self.app_instance.user_service:
            self.app_instance.user_service.set_context(user.id, session.session_token)
        if self.app_instance.notification_service:
            self.app_instance.notification_service.set_context(user.id, session.session_token)


# ==================== EXCEPTION HANDLERS ====================

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors."""
    context = get_request_context()
    
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"][1:]) if len(error["loc"]) > 1 else None
        errors.append(ErrorDetail(
            field=field,
            message=error["msg"],
            code=error["type"]
        ))
    
    response_data = ValidationErrorResponse(
        message="Validation failed",
        errors=errors
    )
    
    # Add request ID if available
    response = JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(response_data)
    )
    
    if context:
        response.headers["X-Request-ID"] = context.request_id
    
    return response


async def core_validation_exception_handler(request: Request, exc: CoreValidationError) -> JSONResponse:
    """Handle core validation errors."""
    context = get_request_context()
    
    error_detail = ErrorDetail(
        field=getattr(exc, 'field', None),
        message=str(exc),
        code="VALIDATION_ERROR"
    )
    
    response_data = ValidationErrorResponse(
        message="Validation failed",
        errors=[error_detail]
    )
    
    response = JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(response_data)
    )
    
    if context:
        response.headers["X-Request-ID"] = context.request_id
    
    return response


async def business_rule_exception_handler(request: Request, exc: BusinessRuleViolationError) -> JSONResponse:
    """Handle business rule violations."""
    context = get_request_context()
    
    response_data = APIResponse.error_response(
        message=str(exc),
        error_code="BUSINESS_RULE_VIOLATION",
        request_id=context.request_id if context else None
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(response_data)
    )


async def authorization_exception_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
    """Handle authorization errors."""
    context = get_request_context()
    
    response_data = APIResponse.error_response(
        message=str(exc),
        error_code="AUTHORIZATION_ERROR",
        request_id=context.request_id if context else None
    )
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content=jsonable_encoder(response_data)
    )


async def repository_exception_handler(request: Request, exc: RepositoryError) -> JSONResponse:
    """Handle repository errors."""
    context = get_request_context()
    
    if isinstance(exc, RecordNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
        error_code = "RECORD_NOT_FOUND"
    elif isinstance(exc, DuplicateRecordError):
        status_code = status.HTTP_409_CONFLICT
        error_code = "DUPLICATE_RECORD"
    elif isinstance(exc, ConcurrencyError):
        status_code = status.HTTP_409_CONFLICT
        error_code = "CONCURRENCY_ERROR"
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_code = "REPOSITORY_ERROR"
    
    response_data = APIResponse.error_response(
        message=str(exc),
        error_code=error_code,
        request_id=context.request_id if context else None
    )
    
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(response_data)
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    context = get_request_context()
    
    response_data = APIResponse.error_response(
        message=exc.detail,
        error_code="HTTP_ERROR",
        request_id=context.request_id if context else None
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(response_data)
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    context = get_request_context()
    
    # Log the error
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # In production, don't expose internal error details
    if context and hasattr(context, 'user') and context.user and context.user.role == UserRole.SUPER_ADMIN:
        error_message = f"Internal server error: {str(exc)}"
        metadata = {"traceback": traceback.format_exc()}
    else:
        error_message = "Internal server error"
        metadata = None
    
    response_data = APIResponse.error_response(
        message=error_message,
        error_code="INTERNAL_ERROR",
        metadata=metadata,
        request_id=context.request_id if context else None
    )
    
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(response_data)
    )


# ==================== DEPENDENCY INJECTION ====================

def get_app_instance() -> AutoERPApplication:
    """Get the application instance."""
    return app_instance


def get_user_service() -> UserService:
    """Get the user service."""
    return app_instance.user_service


def get_notification_service() -> NotificationService:
    """Get the notification service."""
    return app_instance.notification_service


def get_current_user() -> User:
    """Get the current authenticated user."""
    context = get_request_context()
    if not context or not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return context.user


def get_current_user_optional() -> Optional[User]:
    """Get the current user if authenticated."""
    context = get_request_context()
    return context.user if context and context.is_authenticated else None


def require_permission(permission: str):
    """Dependency to require specific permission."""
    def _require_permission():
        context = get_request_context()
        if not context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        context.require_permission(permission)
    
    return _require_permission


def require_role(role: UserRole):
    """Dependency to require specific role."""
    def _require_role():
        user = get_current_user()
        if not user.role.can_access(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role.value}' required"
            )
    
    return _require_role


# ==================== APPLICATION LIFECYCLE ====================

app_instance: Optional[AutoERPApplication] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global app_instance
    
    # Startup
    logger.info("Starting AutoERP API application...")
    
    try:
        # Load configuration
        config_path = Path("config/autoerp.json")
        if config_path.exists():
            config = AutoERPConfig.from_file(config_path)
        else:
            config = AutoERPConfig()
            logger.warning("Using default configuration - config file not found")
        
        # Initialize application
        app_instance = AutoERPApplication(config)
        await app_instance.initialize()
        
        logger.info("AutoERP API application started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down AutoERP API application...")
        
        if app_instance:
            await app_instance.cleanup()
        
        logger.info("AutoERP API application shut down successfully")


# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title="AutoERP API",
    description="Enterprise Resource Planning System REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production
app.add_middleware(RequestContextMiddleware)

# Add authentication middleware after app_instance is available
@app.middleware("http")
async def auth_middleware_wrapper(request: Request, call_next):
    """Wrapper for auth middleware that gets app_instance dynamically."""
    if app_instance:
        auth_middleware = AuthMiddleware(app, app_instance)
        return await auth_middleware.dispatch(request, call_next)
    else:
        return await call_next(request)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(CoreValidationError, core_validation_exception_handler)
app.add_exception_handler(BusinessRuleViolationError, business_rule_exception_handler)
app.add_exception_handler(AuthorizationError, authorization_exception_handler)
app.add_exception_handler(RepositoryError, repository_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# ==================== SYSTEM ENDPOINTS ====================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {"message": "AutoERP API", "version": "1.0.0"}


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health Check",
    description="Get system health status"
)
async def health_check():
    """Get system health status."""
    try:
        health_data = await app_instance.get_health_status()
        
        # Determine overall status
        status_value = health_data.get('status', 'unhealthy')
        
        return HealthCheckResponse(
            status=status_value,
            components=health_data.get('components', {}),
            uptime_seconds=None,  # Will be calculated by the health checker
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            components={"error": {"status": "error", "message": str(e)}},
            version="1.0.0"
        )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["System"],
    summary="System Metrics",
    description="Get system metrics and statistics"
)
async def get_metrics():
    """Get system metrics."""
    try:
        metrics_data = app_instance.get_metrics()
        
        return MetricsResponse(
            uptime_seconds=metrics_data.get('uptime_seconds', 0),
            metrics_count=metrics_data.get('metrics_count', 0),
            metrics=metrics_data.get('metrics', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


# ==================== TABLE ENDPOINTS ====================

@app.get(
    "/api/tables",
    response_model=APIResponse[List[TableInfo]],
    tags=["Tables"],
    summary="List Tables",
    description="Get list of all available tables with metadata"
)
@timed_operation("api.tables.list")
async def list_tables(
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get list of all tables."""
    try:
        context = get_request_context()
        
        # Get all registered model classes from the model registry
        from .core import ModelRegistry
        models = ModelRegistry.get_all_models()
        
        tables = []
        for model_name, model_class in models.items():
            # Get table name
            table_name = getattr(model_class, '__table_name__', model_name.lower() + 's')
            
            # Get record count (simplified - in production, use proper repository)
            try:
                repository = app_instance.get_repository(model_class)
                record_count = await repository.count()
            except Exception:
                record_count = 0
            
            # Get column definitions
            columns = []
            if hasattr(model_class, '_fields'):
                for field_name, field_descriptor in model_class._fields.items():
                    column_info = {
                        "name": field_name,
                        "data_type": field_descriptor.field_type.__name__,
                        "nullable": field_descriptor.nullable,
                        "primary_key": field_name == "id",
                        "unique": field_name == "id",
                        "max_length": getattr(field_descriptor, 'max_length', None),
                        "default_value": field_descriptor.default,
                        "description": field_descriptor.description
                    }
                    columns.append(column_info)
            
            # Determine permissions
            permissions = {
                "read": True,  # Anonymous read allowed for table listing
                "create": current_user is not None,
                "update": current_user is not None,
                "delete": current_user is not None and (
                    current_user.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
                )
            }
            
            table_info = TableInfo(
                name=table_name,
                display_name=model_name,
                description=f"Table for {model_name} entities",
                record_count=record_count,
                columns=columns,
                created_at=None,
                updated_at=None,
                permissions=permissions
            )
            
            tables.append(table_info)
        
        return APIResponse.success_response(
            data=tables,
            message=f"Found {len(tables)} tables",
            request_id=context.request_id if context else None
        )
        
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve table list"
        )


@app.get(
    "/api/tables/{table_name}/records",
    response_model=APIResponse[PaginatedResponse[RecordResponse]],
    tags=["Tables"],
    summary="Get Table Records",
    description="Get records from a specific table with pagination and filtering"
)
@timed_operation("api.tables.records")
async def get_table_records(
    table_name: str = PathParam(..., description="Name of the table"),
    pagination: PaginationRequest = Depends(),
    sort: SortRequest = Depends(),
    filters: FilterRequest = Depends(),
    include_deleted: bool = Query(False, description="Include soft-deleted records"),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get records from a table."""
    try:
        context = get_request_context()
        
        # Find model class for table
        from .core import ModelRegistry
        models = ModelRegistry.get_all_models()
        
        model_class = None
        for model_name, cls in models.items():
            cls_table_name = getattr(cls, '__table_name__', model_name.lower() + 's')
            if cls_table_name == table_name:
                model_class = cls
                break
        
        if not model_class:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table '{table_name}' not found"
            )
        
        # Check permissions for deleted records
        if include_deleted and current_user and current_user.role not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
            include_deleted = False
        
        # Create CRUD service for this model
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=app_instance.audit_logger if hasattr(app_instance, 'audit_logger') else None,
            event_dispatcher=app_instance.event_dispatcher
        )
        
        # Set user context if authenticated
        if current_user:
            crud_service.set_context(current_user.id, context.session_id if context else None)
        
        # Convert request parameters to core types
        pagination_info = pagination.to_pagination_info()
        filter_criteria = filters.to_filter_criteria(sort)
        
        # Get records
        result = await crud_service.read_records(
            pagination=pagination_info,
            filters=filter_criteria,
            include_deleted=include_deleted
        )
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message
            )
        
        records, updated_pagination = result.get_data()
        
        # Convert records to response format
        record_responses = []
        for record in records:
            record_data = record.to_dict()
            record_response = RecordResponse(
                id=record.id,
                data={k: v for k, v in record_data.items() if k != 'id'},
                created_at=getattr(record, 'created_at', None),
                updated_at=getattr(record, 'updated_at', None),
                created_by=getattr(record, 'created_by', None),
                updated_by=getattr(record, 'updated_by', None),
                version=getattr(record, 'version', None)
            )
            record_responses.append(record_response)
        
        # Create paginated response
        paginated_data = PaginatedResponse.create(record_responses, updated_pagination)
        
        return APIResponse.success_response(
            data=paginated_data,
            message=f"Retrieved {len(records)} records from {table_name}",
            request_id=context.request_id if context else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get records from table {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve records from table '{table_name}'"
        )


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post(
    "/api/auth/login",
    response_model=APIResponse[LoginResponse],
    tags=["Authentication"],
    summary="User Login",
    description="Authenticate user and create session"
)
@timed_operation("api.auth.login")
async def login(
    login_request: LoginRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Authenticate user and return session token."""
    try:
        context = get_request_context()
        
        # Authenticate user
        result = await app_instance.user_service.authenticate_user(
            username_or_email=login_request.username_or_email,
            password=login_request.password,
            ip_address=context.ip_address if context else None,
            user_agent=context.user_agent if context else None
        )
        
        if not result.is_success():
            # Add delay for failed login attempts to prevent brute force
            background_tasks.add_task(asyncio.sleep, 1)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.error_message
            )
        
        user, session = result.get_data()
        
        # Create response
        login_response = LoginResponse.from_user_session(user, session)
        
        response_data = APIResponse.success_response(
            data=login_response,
            message="Login successful",
            request_id=context.request_id if context else None
        )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.post(
    "/api/auth/logout",
    response_model=APIResponse[bool],
    tags=["Authentication"],
    summary="User Logout",
    description="Logout user and invalidate session"
)
@timed_operation("api.auth.logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """Logout current user."""
    try:
        context = get_request_context()
        
        if context and context.session:
            result = await app_instance.user_service.logout_user(context.session.session_token)
            
            return APIResponse.success_response(
                data=result.is_success(),
                message="Logout successful" if result.is_success() else "Logout failed",
                request_id=context.request_id
            )
        
        return APIResponse.success_response(
            data=True,
            message="No active session to logout",
            request_id=context.request_id if context else None
        )
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@app.post(
    "/api/auth/register",
    response_model=APIResponse[UserResponse],
    tags=["Authentication"],
    summary="User Registration",
    description="Register new user account"
)
@timed_operation("api.auth.register")
async def register(
    user_request: UserCreateRequest,
    background_tasks: BackgroundTasks
):
    """Register a new user."""
    try:
        context = get_request_context()
        
        # Create user
        result = await app_instance.user_service.create_user(
            username=user_request.username,
            email=user_request.email,
            password=user_request.password,
            first_name=user_request.first_name,
            last_name=user_request.last_name,
            role=user_request.role or UserRole.USER
        )
        
        if not result.is_success():
            if result.error_code == "USERNAME_EXISTS":
                status_code = status.HTTP_409_CONFLICT
            elif result.error_code == "EMAIL_EXISTS":
                status_code = status.HTTP_409_CONFLICT
            elif result.error_code == "WEAK_PASSWORD":
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            
            raise HTTPException(
                status_code=status_code,
                detail=result.error_message
            )
        
        user = result.get_data()
        
        # Send welcome email in background
        if app_instance.notification_service:
            background_tasks.add_task(
                send_welcome_email,
                user.id,
                user.email,
                user.first_name
            )
        
        user_response = UserResponse.from_user(user)
        
        return APIResponse.success_response(
            data=user_response,
            message="User registered successfully",
            request_id=context.request_id if context else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@app.get(
    "/api/auth/me",
    response_model=APIResponse[UserResponse],
    tags=["Authentication"],
    summary="Current User Info",
    description="Get current authenticated user information"
)
@timed_operation("api.auth.me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    try:
        context = get_request_context()
        
        user_response = UserResponse.from_user(current_user)
        
        return APIResponse.success_response(
            data=user_response,
            message="User information retrieved",
            request_id=context.request_id if context else None
        )
        
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )


@app.put(
    "/api/auth/me",
    response_model=APIResponse[UserResponse],
    tags=["Authentication"],
    summary="Update User Profile",
    description="Update current user profile information"
)
@timed_operation("api.auth.update_profile")
async def update_user_profile(
    update_request: UserUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """Update current user profile."""
    try:
        context = get_request_context()
        
        # Prepare updates
        updates = {}
        if update_request.first_name is not None:
            updates['first_name'] = update_request.first_name
        if update_request.last_name is not None:
            updates['last_name'] = update_request.last_name
        if update_request.email is not None:
            updates['email'] = update_request.email
        if update_request.preferences is not None:
            updates['preferences'] = update_request.preferences
        
        if not updates:
            return APIResponse.success_response(
                data=UserResponse.from_user(current_user),
                message="No updates provided",
                request_id=context.request_id
            )
        
        # Update user profile
        result = await app_instance.user_service.update_user_profile(
            user_id=current_user.id,
            updates=updates
        )
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        updated_user = result.get_data()
        user_response = UserResponse.from_user(updated_user)
        
        return APIResponse.success_response(
            data=user_response,
            message="Profile updated successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@app.put(
    "/api/auth/change-password",
    response_model=APIResponse[bool],
    tags=["Authentication"],
    summary="Change Password",
    description="Change user password"
)
@timed_operation("api.auth.change_password")
async def change_password(
    password_request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user)
):
    """Change user password."""
    try:
        context = get_request_context()
        
        result = await app_instance.user_service.change_password(
            user_id=current_user.id,
            old_password=password_request.current_password,
            new_password=password_request.new_password
        )
        
        if not result.is_success():
            if result.error_code == "INVALID_PASSWORD":
                status_code = status.HTTP_400_BAD_REQUEST
            elif result.error_code == "WEAK_PASSWORD":
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            
            raise HTTPException(
                status_code=status_code,
                detail=result.error_message
            )
        
        return APIResponse.success_response(
            data=True,
            message="Password changed successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )


# ==================== USER MANAGEMENT ENDPOINTS ====================

@app.get(
    "/api/users",
    response_model=APIResponse[PaginatedResponse[UserResponse]],
    tags=["Users"],
    summary="List Users",
    description="Get paginated list of users (Admin only)"
)
@timed_operation("api.users.list")
async def list_users(
    pagination: PaginationRequest = Depends(),
    sort: SortRequest = Depends(),
    filters: FilterRequest = Depends(),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_role(UserRole.ADMIN))
):
    """Get paginated list of users."""
    try:
        context = get_request_context()
        
        # Create CRUD service for User
        crud_service = CRUDService(
            model_class=User,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        crud_service.set_context(current_user.id, context.session_id if context else None)
        
        # Get users
        pagination_info = pagination.to_pagination_info()
        filter_criteria = filters.to_filter_criteria(sort)
        
        result = await crud_service.read_records(
            pagination=pagination_info,
            filters=filter_criteria
        )
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message
            )
        
        users, updated_pagination = result.get_data()
        
        # Convert to response format
        user_responses = [UserResponse.from_user(user) for user in users]
        paginated_data = PaginatedResponse.create(user_responses, updated_pagination)
        
        return APIResponse.success_response(
            data=paginated_data,
            message=f"Retrieved {len(users)} users",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@app.get(
    "/api/users/{user_id}",
    response_model=APIResponse[UserResponse],
    tags=["Users"],
    summary="Get User",
    description="Get user by ID (Admin only)"
)
@timed_operation("api.users.get")
async def get_user(
    user_id: str = PathParam(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_role(UserRole.ADMIN))
):
    """Get user by ID."""
    try:
        context = get_request_context()
        
        user_repository = app_instance.get_repository(User)
        user = await user_repository.get_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_response = UserResponse.from_user(user)
        
        return APIResponse.success_response(
            data=user_response,
            message="User retrieved successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


@app.put(
    "/api/users/{user_id}",
    response_model=APIResponse[UserResponse],
    tags=["Users"],
    summary="Update User",
    description="Update user by ID (Admin only)"
)
@timed_operation("api.users.update")
async def update_user(
    user_id: str = PathParam(..., description="User ID"),
    update_request: UserUpdateRequest = Body(...),
    current_user: User = Depends(get_current_user),
    _: None = Depends(require_role(UserRole.ADMIN))
):
    """Update user by ID."""
    try:
        context = get_request_context()
        
        # Prepare updates
        updates = {}
        if update_request.first_name is not None:
            updates['first_name'] = update_request.first_name
        if update_request.last_name is not None:
            updates['last_name'] = update_request.last_name
        if update_request.email is not None:
            updates['email'] = update_request.email
        if update_request.preferences is not None:
            updates['preferences'] = update_request.preferences
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )
        
        # Update user
        result = await app_instance.user_service.update_user_profile(
            user_id=user_id,
            updates=updates
        )
        
        if not result.is_success():
            if result.error_code == "USER_NOT_FOUND":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        updated_user = result.get_data()
        user_response = UserResponse.from_user(updated_user)
        
        return APIResponse.success_response(
            data=user_response,
            message="User updated successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


# ==================== TABLE RECORD CRUD ENDPOINTS ====================

@app.post(
    "/api/tables/{table_name}/records",
    response_model=APIResponse[RecordResponse],
    tags=["Tables"],
    summary="Create Record",
    description="Create a new record in the specified table"
)
@timed_operation("api.tables.create_record")
async def create_table_record(
    table_name: str = PathParam(..., description="Name of the table"),
    record_request: RecordCreateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Create a new record in table."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        crud_service.set_context(current_user.id, context.session_id)
        
        # Create record
        result = await crud_service.create_record(record_request.data)
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        record = result.get_data()
        
        # Convert to response format
        record_data = record.to_dict()
        record_response = RecordResponse(
            id=record.id,
            data={k: v for k, v in record_data.items() if k != 'id'},
            created_at=getattr(record, 'created_at', None),
            updated_at=getattr(record, 'updated_at', None),
            created_by=getattr(record, 'created_by', None),
            updated_by=getattr(record, 'updated_by', None),
            version=getattr(record, 'version', None)
        )
        
        return APIResponse.success_response(
            data=record_response,
            message="Record created successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create record in {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create record"
        )


@app.get(
    "/api/tables/{table_name}/records/{record_id}",
    response_model=APIResponse[RecordResponse],
    tags=["Tables"],
    summary="Get Record",
    description="Get a specific record by ID"
)
@timed_operation("api.tables.get_record")
async def get_table_record(
    table_name: str = PathParam(..., description="Name of the table"),
    record_id: str = PathParam(..., description="Record ID"),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Get record by ID."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        
        if current_user:
            crud_service.set_context(current_user.id, context.session_id if context else None)
        
        # Get record
        result = await crud_service.get_record_by_id(record_id)
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message
            )
        
        record = result.get_data()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Record not found"
            )
        
        # Convert to response format
        record_data = record.to_dict()
        record_response = RecordResponse(
            id=record.id,
            data={k: v for k, v in record_data.items() if k != 'id'},
            created_at=getattr(record, 'created_at', None),
            updated_at=getattr(record, 'updated_at', None),
            created_by=getattr(record, 'created_by', None),
            updated_by=getattr(record, 'updated_by', None),
            version=getattr(record, 'version', None)
        )
        
        return APIResponse.success_response(
            data=record_response,
            message="Record retrieved successfully",
            request_id=context.request_id if context else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get record {record_id} from {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve record"
        )


@app.put(
    "/api/tables/{table_name}/records/{record_id}",
    response_model=APIResponse[RecordResponse],
    tags=["Tables"],
    summary="Update Record",
    description="Update a specific record by ID"
)
@timed_operation("api.tables.update_record")
async def update_table_record(
    table_name: str = PathParam(..., description="Name of the table"),
    record_id: str = PathParam(..., description="Record ID"),
    update_request: RecordUpdateRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Update record by ID."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        crud_service.set_context(current_user.id, context.session_id)
        
        # Update record
        result = await crud_service.update_record(record_id, update_request.data)
        
        if not result.is_success():
            if "not found" in result.error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Record not found"
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.error_message
            )
        
        record = result.get_data()
        
        # Convert to response format
        record_data = record.to_dict()
        record_response = RecordResponse(
            id=record.id,
            data={k: v for k, v in record_data.items() if k != 'id'},
            created_at=getattr(record, 'created_at', None),
            updated_at=getattr(record, 'updated_at', None),
            created_by=getattr(record, 'created_by', None),
            updated_by=getattr(record, 'updated_by', None),
            version=getattr(record, 'version', None)
        )
        
        return APIResponse.success_response(
            data=record_response,
            message="Record updated successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update record {record_id} in {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update record"
        )


@app.delete(
    "/api/tables/{table_name}/records/{record_id}",
    response_model=APIResponse[bool],
    tags=["Tables"],
    summary="Delete Record",
    description="Delete a specific record by ID"
)
@timed_operation("api.tables.delete_record")
async def delete_table_record(
    table_name: str = PathParam(..., description="Name of the table"),
    record_id: str = PathParam(..., description="Record ID"),
    soft_delete: bool = Query(True, description="Use soft delete if available"),
    current_user: User = Depends(get_current_user)
):
    """Delete record by ID."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        crud_service.set_context(current_user.id, context.session_id)
        
        # Delete record
        result = await crud_service.delete_record(record_id, soft_delete)
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message
            )
        
        success = result.get_data()
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Record not found"
            )
        
        return APIResponse.success_response(
            data=True,
            message="Record deleted successfully",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete record {record_id} from {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete record"
        )


# ==================== BULK OPERATIONS ====================

@app.post(
    "/api/tables/{table_name}/bulk",
    response_model=APIResponse[BulkOperationResponse],
    tags=["Tables"],
    summary="Bulk Operations",
    description="Perform bulk operations on table records"
)
@timed_operation("api.tables.bulk_operation")
async def bulk_table_operations(
    table_name: str = PathParam(..., description="Name of the table"),
    bulk_request: BulkOperationRequest = Body(...),
    current_user: User = Depends(get_current_user)
):
    """Perform bulk operations on records."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        crud_service.set_context(current_user.id, context.session_id)
        
        total_requested = len(bulk_request.records)
        successful = 0
        failed = 0
        errors = []
        results = []
        
        if bulk_request.operation == "create":
            # Bulk create
            result = await crud_service.bulk_create_records(bulk_request.records)
            
            if result.is_success():
                created_records = result.get_data()
                successful = len(created_records)
                results = [{"id": record.id, "status": "created"} for record in created_records]
            else:
                failed = total_requested
                errors = [{"error": result.error_message}]
        
        elif bulk_request.operation == "update":
            # Bulk update - expects records to have 'id' field
            updates = []
            for record_data in bulk_request.records:
                if 'id' not in record_data:
                    errors.append({"error": "Missing 'id' field in record", "record": record_data})
                    failed += 1
                    continue
                
                record_id = record_data.pop('id')
                updates.append((record_id, record_data))
            
            if updates:
                result = await crud_service.bulk_update_records(updates)
                
                if result.is_success():
                    updated_records = result.get_data()
                    successful = len(updated_records)
                    results = [{"id": record.id, "status": "updated"} for record in updated_records]
                else:
                    failed += len(updates)
                    errors.append({"error": result.error_message})
        
        elif bulk_request.operation == "delete":
            # Bulk delete - expects records to have 'id' field
            soft_delete = bulk_request.options.get('soft_delete', True) if bulk_request.options else True
            
            for record_data in bulk_request.records:
                if 'id' not in record_data:
                    errors.append({"error": "Missing 'id' field in record", "record": record_data})
                    failed += 1
                    continue
                
                try:
                    result = await crud_service.delete_record(record_data['id'], soft_delete)
                    
                    if result.is_success() and result.get_data():
                        successful += 1
                        results.append({"id": record_data['id'], "status": "deleted"})
                    else:
                        failed += 1
                        errors.append({
                            "id": record_data['id'],
                            "error": result.error_message or "Delete failed"
                        })
                        
                except Exception as e:
                    failed += 1
                    errors.append({"id": record_data['id'], "error": str(e)})
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported bulk operation: {bulk_request.operation}"
            )
        
        # Create response
        bulk_response = BulkOperationResponse(
            operation=bulk_request.operation,
            total_requested=total_requested,
            successful=successful,
            failed=failed,
            errors=errors,
            results=results
        )
        
        return APIResponse.success_response(
            data=bulk_response,
            message=f"Bulk {bulk_request.operation} completed: {successful} successful, {failed} failed",
            request_id=context.request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk operation failed for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk operation failed"
        )


# ==================== UTILITY FUNCTIONS ====================

async def get_model_class_for_table(table_name: str) -> Type[CoreBaseModel]:
    """Get model class for table name."""
    from .core import ModelRegistry
    
    models = ModelRegistry.get_all_models()
    
    for model_name, model_class in models.items():
        cls_table_name = getattr(model_class, '__table_name__', model_name.lower() + 's')
        if cls_table_name == table_name:
            return model_class
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Table '{table_name}' not found"
    )


async def send_welcome_email(user_id: str, email: str, first_name: str):
    """Send welcome email to new user."""
    try:
        if app_instance.notification_service:
            await app_instance.notification_service.send_notification(
                recipient_id=user_id,
                template_name="welcome_email",
                variables={
                    "first_name": first_name,
                    "email": email
                },
                recipient_address=email
            )
    except Exception as e:
        logger.error(f"Failed to send welcome email to {email}: {e}")


# ==================== FILE UPLOAD ENDPOINTS ====================

@app.post(
    "/api/upload",
    response_model=APIResponse[Dict[str, T]],
    tags=["Files"],
    summary="Upload File",
    description="Upload a file for processing"
)
@timed_operation("api.upload.file")
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    table_name: Optional[str] = Form(None, description="Target table for data import"),
    current_user: User = Depends(get_current_user)
):
    """Upload and process a file."""
    try:
        context = get_request_context()
        
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        file_size = 0
        content = io.BytesIO()
        
        while chunk := await file.read(8192):
            file_size += len(chunk)
            if file_size > max_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File too large (max 10MB)"
                )
            content.write(chunk)
        
        content.seek(0)
        
        # Save file temporarily
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(content.read())
            temp_file_path = temp_file.name
        
        try:
            # Process file if table_name is provided
            if table_name:
                # Use DataLoaderService to process the file
                from .core import DataLoaderService
                
                data_loader = DataLoaderService(
                    unit_of_work=app_instance.uow,
                    audit_logger=getattr(app_instance, 'audit_logger', None),
                    event_dispatcher=app_instance.event_dispatcher
                )
                data_loader.set_context(current_user.id, context.session_id)
                
                # Detect format
                format_result = await data_loader.detect_format(temp_file_path)
                if not format_result.is_success():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to detect file format: {format_result.error_message}"
                    )
                
                file_format = format_result.get_data()
                
                # Detect schema
                schema_result = await data_loader.detect_schema(temp_file_path, file_format)
                if not schema_result.is_success():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to detect file schema: {schema_result.error_message}"
                    )
                
                schema = schema_result.get_data()
                
                response_data = {
                    "file_id": str(uuid.uuid4()),
                    "filename": file.filename,
                    "size": file_size,
                    "format": file_format.value,
                    "schema": schema.to_dict(),
                    "table_name": table_name,
                    "temp_path": temp_file_path,  # In production, use secure storage
                    "status": "analyzed"
                }
            else:
                # Just upload without processing
                response_data = {
                    "file_id": str(uuid.uuid4()),
                    "filename": file.filename,
                    "size": file_size,
                    "temp_path": temp_file_path,  # In production, use secure storage
                    "status": "uploaded"
                }
            
            return APIResponse.success_response(
                data=response_data,
                message="File uploaded successfully",
                request_id=context.request_id
            )
            
        except HTTPException:
            # Clean up temp file on error
            Path(temp_file_path).unlink(missing_ok=True)
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )


# ==================== EXPORT ENDPOINTS ====================

@app.get(
    "/api/tables/{table_name}/export",
    tags=["Tables"],
    summary="Export Table Data",
    description="Export table data in various formats"
)
@timed_operation("api.tables.export")
async def export_table_data(
    table_name: str = PathParam(..., description="Name of the table"),
    format: Literal["csv", "json", "excel"] = Query("csv", description="Export format"),
    filters: FilterRequest = Depends(),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """Export table data."""
    try:
        context = get_request_context()
        
        # Find model class
        model_class = await get_model_class_for_table(table_name)
        
        # Create CRUD service
        crud_service = CRUDService(
            model_class=model_class,
            unit_of_work=app_instance.uow,
            audit_logger=getattr(app_instance, 'audit_logger', None),
            event_dispatcher=app_instance.event_dispatcher
        )
        
        if current_user:
            crud_service.set_context(current_user.id, context.session_id if context else None)
        
        # Get all records (with large pagination)
        pagination = PaginationInfo(page=1, per_page=10000)  # Large batch
        filter_criteria = filters.to_filter_criteria()
        
        result = await crud_service.read_records(
            pagination=pagination,
            filters=filter_criteria
        )
        
        if not result.is_success():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error_message
            )
        
        records, _ = result.get_data()
        
        # Convert to export format
        if format == "csv":
            content, media_type, filename = await export_as_csv(records, table_name)
        elif format == "json":
            content, media_type, filename = await export_as_json(records, table_name)
        elif format == "excel":
            content, media_type, filename = await export_as_excel(records, table_name)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported export format: {format}"
            )
        
        # Return file response
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed for {table_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Export failed"
        )


async def export_as_csv(records: List[CoreBaseModel], table_name: str) -> Tuple[bytes, str, str]:
    """Export records as CSV."""
    import csv
    
    if not records:
        return b"", "text/csv", f"{table_name}.csv"
    
    output = io.StringIO()
    
    # Get field names from first record
    field_names = list(records[0].to_dict().keys())
    
    writer = csv.DictWriter(output, fieldnames=field_names)
    writer.writeheader()
    
    for record in records:
        row_data = record.to_dict()
        # Convert datetime objects to strings
        for key, value in row_data.items():
            if isinstance(value, datetime):
                row_data[key] = value.isoformat()
        writer.writerow(row_data)
    
    content = output.getvalue().encode('utf-8')
    return content, "text/csv", f"{table_name}.csv"


async def export_as_json(records: List[CoreBaseModel], table_name: str) -> Tuple[bytes, str, str]:
    """Export records as JSON."""
    records_data = [record.to_dict() for record in records]
    content = json.dumps(records_data, indent=2, default=str).encode('utf-8')
    return content, "application/json", f"{table_name}.json"


async def export_as_excel(records: List[CoreBaseModel], table_name: str) -> Tuple[bytes, str, str]:
    """Export records as Excel."""
    # Simplified Excel export - in production use openpyxl or xlsxwriter
    # For now, fall back to CSV format
    return await export_as_csv(records, table_name.replace('.xlsx', '.csv'))


# ==================== WEBSOCKET SUPPORT ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time communications."""
    await websocket.accept()
    
    try:
        while True:
            # Simple echo server - extend for real-time notifications
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ==================== MODULE EXPORTS ====================

__all__ = [
    # FastAPI app
    'app',
    
    # Schemas
    'APIResponse', 'PaginationRequest', 'SortRequest', 'FilterRequest',
    'PaginatedResponse', 'HealthCheckResponse', 'MetricsResponse',
    'UserCreateRequest', 'UserUpdateRequest', 'UserResponse',
    'LoginRequest', 'LoginResponse', 'ChangePasswordRequest',
    'TableInfo', 'ColumnInfo', 'RecordCreateRequest', 'RecordUpdateRequest',
    'RecordResponse', 'BulkOperationRequest', 'BulkOperationResponse',
    
    # Context and middleware
    'RequestContext', 'RequestContextMiddleware', 'AuthMiddleware',
    
    # Dependencies
    'get_app_instance', 'get_user_service', 'get_notification_service',
    'get_current_user', 'get_current_user_optional',
    'require_permission', 'require_role'
]


if __name__ == "__main__":
    # For development - use uvicorn in production
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )