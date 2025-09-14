"""
AutoERP Core Module - Hexagonal Architecture Implementation
Version: 1.0.0
Author: AutoERP Development Team
License: MIT

This module implements the core domain logic for the AutoERP system using hexagonal architecture.
It contains the central business logic, entities, value objects, and domain services.
"""

import asyncio
import sqlite3
import psycopg2
import psycopg2.pool
import threading
import logging
import uuid
import json
import hashlib
import secrets
import datetime
import decimal
import enum
import re
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import (
    Any, Dict, List, Optional, Union, Callable, Generic, TypeVar, 
    Protocol, runtime_checkable, Type, Set, Tuple, Iterator, 
    AsyncIterator, ClassVar, Final, Literal, overload
)
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps, lru_cache, partial
from pathlib import Path
from queue import Queue, Empty
from threading import Lock, RLock, Event, Condition
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlparse
import smtplib
import ssl

from pydantic import BaseModel
from enum import Enum
from typing import Optional, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoerp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION DATACLASSES ====================

class SchemaField(BaseModel):
    name: str
    type: str
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None
    
class FilterCriteria(BaseModel):
    field: str
    operator: str   # مثل: '=', '!=', '>', '<', 'LIKE'
    value: Any

class DataType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"

class PaginationInfo(BaseModel):
    page: int = 1
    size: int = 10
    total: Optional[int] = None

@dataclass(frozen=True)
class DatabaseConfig:
    """Database connection configuration."""
    
    engine: Literal['sqlite', 'postgresql'] = 'sqlite'
    host: str = 'localhost'
    port: int = 5432
    database: str = 'autoerp.db'
    username: str = 'autoerp'
    password: str = ''
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = 'prefer'
    charset: str = 'utf8'
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.engine not in ('sqlite', 'postgresql'):
            raise ValueError(f"Unsupported database engine: {self.engine}")
        if self.pool_size <= 0:
            raise ValueError("Pool size must be positive")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")


@dataclass(frozen=True)
class SecurityConfig:
    """Security configuration for the application."""
    
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = 'HS256'
    jwt_expiration_minutes: int = 1440  # 24 hours
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    csrf_protection: bool = True
    secure_cookies: bool = True
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.password_min_length < 4:
            raise ValueError("Password minimum length must be at least 4")
        if self.jwt_expiration_minutes <= 0:
            raise ValueError("JWT expiration must be positive")


@dataclass(frozen=True)
class CacheConfig:
    """Cache configuration."""
    
    enabled: bool = True
    backend: Literal['memory', 'redis'] = 'memory'
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_timeout: int = 300  # 5 minutes
    max_entries: int = 10000
    cleanup_interval: int = 600  # 10 minutes


@dataclass(frozen=True)
class EmailConfig:
    """Email configuration for notifications."""
    
    smtp_server: str = 'localhost'
    smtp_port: int = 587
    username: str = ''
    password: str = ''
    use_tls: bool = True
    use_ssl: bool = False
    from_address: str = 'noreply@autoerp.com'
    from_name: str = 'AutoERP System'
    
    def __post_init__(self):
        """Validate email configuration."""
        if self.use_tls and self.use_ssl:
            raise ValueError("Cannot use both TLS and SSL")


@dataclass(frozen=True)
class BusinessConfig:
    """Business logic configuration."""
    
    default_currency: str = 'USD'
    decimal_places: int = 2
    tax_calculation_precision: int = 4
    inventory_tracking: bool = True
    multi_location_support: bool = False
    multi_company_support: bool = False
    fiscal_year_start_month: int = 1  # January
    default_payment_terms_days: int = 30
    auto_generate_codes: bool = True
    code_prefix_separator: str = '-'
    
    def __post_init__(self):
        """Validate business configuration."""
        if self.fiscal_year_start_month < 1 or self.fiscal_year_start_month > 12:
            raise ValueError("Fiscal year start month must be between 1 and 12")
        if self.decimal_places < 0 or self.decimal_places > 10:
            raise ValueError("Decimal places must be between 0 and 10")


@dataclass(frozen=True)
class SystemConfig:
    """System-wide configuration."""
    
    debug: bool = False
    testing: bool = False
    log_level: str = 'INFO'
    log_file: str = 'autoerp.log'
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    timezone: str = 'UTC'
    date_format: str = '%Y-%m-%d'
    datetime_format: str = '%Y-%m-%d %H:%M:%S'
    language: str = 'en'
    
    def __post_init__(self):
        """Validate system configuration."""
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")


@dataclass
class AutoERPConfig:
    """Main configuration container for AutoERP system."""
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoERPConfig':
        """Create configuration from dictionary."""
        return cls(
            database=DatabaseConfig(**config_dict.get('database', {})),
            security=SecurityConfig(**config_dict.get('security', {})),
            cache=CacheConfig(**config_dict.get('cache', {})),
            email=EmailConfig(**config_dict.get('email', {})),
            business=BusinessConfig(**config_dict.get('business', {})),
            system=SystemConfig(**config_dict.get('system', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': asdict(self.database),
            'security': asdict(self.security),
            'cache': asdict(self.cache),
            'email': asdict(self.email),
            'business': asdict(self.business),
            'system': asdict(self.system)
        }
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'AutoERPConfig':
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# ==================== BASE MODEL AND MIXINS ====================

T = TypeVar('T', bound='BaseModel')


class ValidationError(Exception):
    """Raised when model validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.field:
            return f"Validation error in field '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class ModelRegistry:
    """Registry for all model classes."""
    
    _models: Dict[str, Type['BaseModel']] = {}
    _lock = Lock()
    
    @classmethod
    def register(cls, model_class: Type['BaseModel']) -> None:
        """Register a model class."""
        with cls._lock:
            cls._models[model_class.__name__] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Type['BaseModel']]:
        """Get a model class by name."""
        return cls._models.get(name)
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Type['BaseModel']]:
        """Get all registered models."""
        with cls._lock:
            return cls._models.copy()
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered models."""
        with cls._lock:
            cls._models.clear()


class FieldValidator:
    """Field validation utilities."""
    
    @staticmethod
    def required(value: Any) -> None:
        """Validate required field."""
        if value is None or value == '':
            raise ValidationError("Field is required")
    
    @staticmethod
    def min_length(min_len: int) -> Callable[[str], None]:
        """Validate minimum string length."""
        def validator(value: str) -> None:
            if len(value) < min_len:
                raise ValidationError(f"Minimum length is {min_len}")
        return validator
    
    @staticmethod
    def max_length(max_len: int) -> Callable[[str], None]:
        """Validate maximum string length."""
        def validator(value: str) -> None:
            if len(value) > max_len:
                raise ValidationError(f"Maximum length is {max_len}")
        return validator
    
    @staticmethod
    def email(value: str) -> None:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, value):
            raise ValidationError("Invalid email format")
    
    @staticmethod
    def numeric_range(min_val: Union[int, float], max_val: Union[int, float]) -> Callable[[Union[int, float]], None]:
        """Validate numeric range."""
        def validator(value: Union[int, float]) -> None:
            if value < min_val or value > max_val:
                raise ValidationError(f"Value must be between {min_val} and {max_val}")
        return validator
    
    @staticmethod
    def regex(pattern: str, message: str = "Invalid format") -> Callable[[str], None]:
        """Validate against regex pattern."""
        compiled_pattern = re.compile(pattern)
        def validator(value: str) -> None:
            if not compiled_pattern.match(value):
                raise ValidationError(message)
        return validator


class FieldDescriptor:
    """Descriptor for model fields with validation."""
    
    def __init__(
        self,
        name: str,
        field_type: Type,
        default: Any = None,
        nullable: bool = True,
        validators: Optional[List[Callable]] = None,
        description: str = ""
    ):
        self.name = name
        self.field_type = field_type
        self.default = default
        self.nullable = nullable
        self.validators = validators or []
        self.description = description
        self.private_name = f"_{name}"
    
    def __set_name__(self, owner: Type, name: str) -> None:
        self.name = name
        self.private_name = f"_{name}"
    
    def __get__(self, instance: Optional['BaseModel'], owner: Type) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.private_name, self.default)
    
    def __set__(self, instance: 'BaseModel', value: Any) -> None:
        if value is None and not self.nullable:
            raise ValidationError(f"Field '{self.name}' cannot be null")
        
        if value is not None:
            # Type validation
            if not isinstance(value, self.field_type):
                try:
                    value = self.field_type(value)
                except (ValueError, TypeError):
                    raise ValidationError(
                        f"Field '{self.name}' must be of type {self.field_type.__name__}"
                    )
            
            # Run custom validators
            for validator in self.validators:
                try:
                    validator(value)
                except ValidationError as e:
                    e.field = self.name
                    e.value = value
                    raise
        
        setattr(instance, self.private_name, value)
        instance._mark_dirty(self.name)


class BaseModelMeta(type):
    """Metaclass for BaseModel that handles field registration and validation setup."""
    
    def __new__(cls, name: str, bases: Tuple[Type, ...], namespace: Dict[str, Any]) -> Type:
        # Collect fields from this class and parent classes
        fields = {}
        
        # Get fields from parent classes
        for base in reversed(bases):
            if hasattr(base, '_fields'):
                fields.update(base._fields)
        
        # Get fields from this class
        for key, value in list(namespace.items()):
            if isinstance(value, FieldDescriptor):
                fields[key] = value
        
        namespace['_fields'] = fields
        
        # Create the class
        new_class = super().__new__(cls, name, bases, namespace)
        
        # Register with ModelRegistry
        if name != 'BaseModel':  # Don't register the base class itself
            ModelRegistry.register(new_class)
        
        return new_class


class AuditMixin:
    """Mixin for audit trail functionality."""
    
    created_at: Optional[datetime] = FieldDescriptor(
        'created_at', datetime, nullable=True,
        description="When the record was created"
    )
    
    updated_at: Optional[datetime] = FieldDescriptor(
        'updated_at', datetime, nullable=True,
        description="When the record was last updated"
    )
    
    created_by: Optional[str] = FieldDescriptor(
        'created_by', str, nullable=True,
        validators=[FieldValidator.max_length(100)],
        description="User who created the record"
    )
    
    updated_by: Optional[str] = FieldDescriptor(
        'updated_by', str, nullable=True,
        validators=[FieldValidator.max_length(100)],
        description="User who last updated the record"
    )
    
    version: int = FieldDescriptor(
        'version', int, default=1,
        validators=[FieldValidator.numeric_range(1, 999999999)],
        description="Record version for optimistic locking"
    )
    
    def _update_audit_fields(self, user_id: Optional[str] = None) -> None:
        """Update audit fields."""
        now = datetime.now(timezone.utc)
        
        if not hasattr(self, '_created_at') or self._created_at is None:
            self.created_at = now
            self.created_by = user_id
        
        self.updated_at = now
        self.updated_by = user_id
        self.version = getattr(self, '_version', 0) + 1
    
    def get_audit_info(self) -> Dict[str, Any]:
        """Get audit information."""
        return {
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'created_by': self.created_by,
            'updated_by': self.updated_by,
            'version': self.version
        }


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    deleted_at: Optional[datetime] = FieldDescriptor(
        'deleted_at', datetime, nullable=True,
        description="When the record was deleted (soft delete)"
    )
    
    deleted_by: Optional[str] = FieldDescriptor(
        'deleted_by', str, nullable=True,
        validators=[FieldValidator.max_length(100)],
        description="User who deleted the record"
    )
    
    is_deleted: bool = FieldDescriptor(
        'is_deleted', bool, default=False,
        description="Whether the record is soft deleted"
    )
    
    def soft_delete(self, user_id: Optional[str] = None) -> None:
        """Soft delete the record."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = user_id
    
    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = None
    
    @property
    def is_active(self) -> bool:
        """Check if the record is active (not soft deleted)."""
        return not self.is_deleted


class BaseModel(AuditMixin, SoftDeleteMixin, metaclass=BaseModelMeta):
    """Base model class with core functionality."""
    
    id: Optional[str] = FieldDescriptor(
        'id', str, nullable=True,
        description="Primary key identifier"
    )
    
    def __init__(self, **kwargs):
        """Initialize the model with provided data."""
        self._dirty_fields: Set[str] = set()
        self._original_values: Dict[str, Any] = {}
        self._validation_errors: List[ValidationError] = []
        
        # Generate ID if not provided
        if 'id' not in kwargs or kwargs['id'] is None:
            kwargs['id'] = self._generate_id()
        
        # Set field values
        for field_name, field_descriptor in self._fields.items():
            value = kwargs.get(field_name, field_descriptor.default)
            if callable(value):
                value = value()
            setattr(self, field_name, value)
            self._original_values[field_name] = getattr(self, field_name)
        
        # Clear dirty fields after initialization
        self._dirty_fields.clear()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the model."""
        return str(uuid.uuid4())
    
    def _mark_dirty(self, field_name: str) -> None:
        """Mark a field as dirty (changed)."""
        self._dirty_fields.add(field_name)
    
    def is_dirty(self, field_name: Optional[str] = None) -> bool:
        """Check if the model or specific field is dirty."""
        if field_name:
            return field_name in self._dirty_fields
        return bool(self._dirty_fields)
    
    def get_dirty_fields(self) -> Set[str]:
        """Get all dirty field names."""
        return self._dirty_fields.copy()
    
    def get_changed_values(self) -> Dict[str, Tuple[Any, Any]]:
        """Get changed values as (old_value, new_value) tuples."""
        changes = {}
        for field_name in self._dirty_fields:
            old_value = self._original_values.get(field_name)
            new_value = getattr(self, field_name)
            changes[field_name] = (old_value, new_value)
        return changes
    
    def reset_dirty_tracking(self) -> None:
        """Reset dirty field tracking."""
        for field_name in self._dirty_fields:
            self._original_values[field_name] = getattr(self, field_name)
        self._dirty_fields.clear()
    
    def validate(self) -> bool:
        """Validate the model and return True if valid."""
        self._validation_errors.clear()
        
        try:
            # Run field-level validation
            for field_name, field_descriptor in self._fields.items():
                value = getattr(self, field_name)
                
                # Check required fields
                if not field_descriptor.nullable and value is None:
                    self._validation_errors.append(
                        ValidationError(f"Field '{field_name}' is required", field_name, value)
                    )
                
                # Run field validators
                if value is not None:
                    for validator in field_descriptor.validators:
                        try:
                            validator(value)
                        except ValidationError as e:
                            e.field = field_name
                            e.value = value
                            self._validation_errors.append(e)
            
            # Run model-level validation
            self._validate_model()
            
        except ValidationError as e:
            self._validation_errors.append(e)
        
        return len(self._validation_errors) == 0
    
    def _validate_model(self) -> None:
        """Override in subclasses for model-specific validation."""
        pass
    
    def get_validation_errors(self) -> List[ValidationError]:
        """Get all validation errors."""
        return self._validation_errors.copy()
    
    def to_dict(self, include_none: bool = True, exclude_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        """Convert model to dictionary."""
        exclude_fields = exclude_fields or set()
        result = {}
        
        for field_name in self._fields:
            if field_name in exclude_fields:
                continue
            
            value = getattr(self, field_name)
            if value is None and not include_none:
                continue
            
            # Convert datetime objects to ISO format
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, date):
                value = value.isoformat()
            elif isinstance(value, Decimal):
                value = str(value)
            
            result[field_name] = value
        
        return result
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create model instance from dictionary."""
        # Convert string dates back to datetime objects
        processed_data = {}
        
        for field_name, field_descriptor in cls._fields.items():
            if field_name not in data:
                continue
            
            value = data[field_name]
            
            # Convert ISO format strings back to datetime
            if field_descriptor.field_type == datetime and isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    pass  # Keep original value if conversion fails
            elif field_descriptor.field_type == date and isinstance(value, str):
                try:
                    value = date.fromisoformat(value)
                except ValueError:
                    pass
            elif field_descriptor.field_type == Decimal and isinstance(value, str):
                try:
                    value = Decimal(value)
                except (ValueError, decimal.InvalidOperation):
                    pass
            
            processed_data[field_name] = value
        
        return cls(**processed_data)
    
    def clone(self: T, **overrides) -> T:
        """Create a copy of the model with optional field overrides."""
        data = self.to_dict()
        data.update(overrides)
        # Generate new ID for the clone
        data['id'] = self._generate_id()
        return self.__class__.from_dict(data)
    
    def __eq__(self, other: Any) -> bool:
        """Compare models by ID."""
        if not isinstance(other, BaseModel):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        return f"{class_name}(id='{self.id}')"
    
    def __str__(self) -> str:
        """String representation of the model."""
        return self.__repr__()


# ==================== CONNECTION MANAGER ====================

class ConnectionPool:
    """Thread-safe connection pool."""
    
    def __init__(
        self, 
        create_connection: Callable[[], Any],
        max_connections: int = 10,
        timeout: float = 30.0
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: Queue = Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = Lock()
        self._closed = False
    
    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        try:
            # Try to get existing connection
            return self._pool.get_nowait()
        except Empty:
            # Create new connection if under limit
            with self._lock:
                if self._created_connections < self.max_connections:
                    self._created_connections += 1
                    return self.create_connection()
            
            # Wait for available connection
            try:
                return self._pool.get(timeout=self.timeout)
            except Empty:
                raise RuntimeError("Connection pool timeout")
    
    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        if self._closed:
            self._close_connection(connection)
            return
        
        try:
            self._pool.put_nowait(connection)
        except:
            # Pool is full, close the connection
            self._close_connection(connection)
            with self._lock:
                self._created_connections -= 1
    
    def _close_connection(self, connection: Any) -> None:
        """Close a single connection."""
        try:
            if hasattr(connection, 'close'):
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        
        # Close all pooled connections
        while True:
            try:
                connection = self._pool.get_nowait()
                self._close_connection(connection)
            except Empty:
                break
        
        with self._lock:
            self._created_connections = 0


class SQLiteConnectionManager:
    """SQLite connection manager with pooling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.database_path = Path(config.database)
        self._ensure_database_exists()
        
        # Create connection pool
        self.pool = ConnectionPool(
            create_connection=self._create_connection,
            max_connections=config.pool_size,
            timeout=config.pool_timeout
        )
        
        logger.info(f"SQLite connection manager initialized: {self.database_path}")
    
    def _ensure_database_exists(self) -> None:
        """Ensure the database file exists."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database if it doesn't exist
        if not self.database_path.exists():
            conn = sqlite3.connect(str(self.database_path))
            conn.close()
            logger.info(f"Created SQLite database: {self.database_path}")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        conn = sqlite3.connect(
            str(self.database_path),
            timeout=self.config.pool_timeout,
            check_same_thread=False
        )
        
        # Configure connection
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
        conn.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode for better concurrency
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with context manager."""
        connection = None
        try:
            connection = self.pool.get_connection()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection:
                self.pool.return_connection(connection)
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params or ())
            return cursor.fetchall()
    
    def execute_command(self, command: str, params: Optional[Tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE command and return affected rows."""
        with self.get_connection() as conn:
            cursor = conn.execute(command, params or ())
            conn.commit()
            return cursor.rowcount
    
    def close(self) -> None:
        """Close all connections."""
        self.pool.close_all()
        logger.info("SQLite connection manager closed")


class PostgreSQLConnectionManager:
    """PostgreSQL connection manager with pooling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        
        # Create connection pool
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=config.pool_size,
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password,
                sslmode=config.ssl_mode,
                connect_timeout=config.pool_timeout
            )
            logger.info(f"PostgreSQL connection pool created: {config.host}:{config.port}/{config.database}")
        except psycopg2.Error as e:
            logger.error(f"Failed to create PostgreSQL connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool with context manager."""
        connection = None
        try:
            connection = self.pool.getconn()
            if connection:
                yield connection
            else:
                raise RuntimeError("Failed to get connection from pool")
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection:
                self.pool.putconn(connection)
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()
    
    def execute_command(self, command: str, params: Optional[Tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE command and return affected rows."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(command, params or ())
                conn.commit()
                return cursor.rowcount
    
    def close(self) -> None:
        """Close all connections."""
        if self.pool:
            self.pool.closeall()
        logger.info("PostgreSQL connection manager closed")


class ConnectionManager:
    """Main connection manager that delegates to specific implementations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._manager = None
        self._lock = Lock()
        
        self._create_manager()
    
    def _create_manager(self) -> None:
        """Create the appropriate connection manager."""
        if self.config.engine == 'sqlite':
            self._manager = SQLiteConnectionManager(self.config)
        elif self.config.engine == 'postgresql':
            self._manager = PostgreSQLConnectionManager(self.config)
        else:
            raise ValueError(f"Unsupported database engine: {self.config.engine}")
    
    @contextmanager
    def get_connection(self):
        """Get a connection with context manager."""
        with self._manager.get_connection() as conn:
            yield conn
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Any]:
        """Execute a SELECT query and return results."""
        return self._manager.execute_query(query, params)
    
    def execute_command(self, command: str, params: Optional[Tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE command and return affected rows."""
        return self._manager.execute_command(command, params)
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self.get_connection():
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information."""
        info = {
            'engine': self.config.engine,
            'database': self.config.database,
            'pool_size': self.config.pool_size
        }
        
        if self.config.engine == 'postgresql':
            info.update({
                'host': self.config.host,
                'port': self.config.port,
                'username': self.config.username
            })
        
        return info
    
    def close(self) -> None:
        """Close the connection manager."""
        if self._manager:
            self._manager.close()
        logger.info("Connection manager closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ==================== QUERY BUILDER ====================

class QueryBuilder:
    """SQL query builder with method chaining."""
    
    def __init__(self, table_name: str, connection_manager: ConnectionManager):
        self.table_name = table_name
        self.connection_manager = connection_manager
        self._select_fields = ['*']
        self._where_conditions = []
        self._join_clauses = []
        self._order_by = []
        self._group_by = []
        self._having_conditions = []
        self._limit_value = None
        self._offset_value = None
        self._params = []
    
    def select(self, *fields: str) -> 'QueryBuilder':
        """Set SELECT fields."""
        self._select_fields = list(fields) if fields else ['*']
        return self
    
    def where(self, condition: str, *params) -> 'QueryBuilder':
        """Add WHERE condition."""
        self._where_conditions.append(condition)
        self._params.extend(params)
        return self
    
    def join(self, table: str, on_condition: str, join_type: str = 'INNER') -> 'QueryBuilder':
        """Add JOIN clause."""
        self._join_clauses.append(f"{join_type} JOIN {table} ON {on_condition}")
        return self
    
    def order_by(self, field: str, direction: str = 'ASC') -> 'QueryBuilder':
        """Add ORDER BY clause."""
        self._order_by.append(f"{field} {direction}")
        return self
    
    def group_by(self, *fields: str) -> 'QueryBuilder':
        """Add GROUP BY clause."""
        self._group_by.extend(fields)
        return self
    
    def having(self, condition: str, *params) -> 'QueryBuilder':
        """Add HAVING condition."""
        self._having_conditions.append(condition)
        self._params.extend(params)
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """Add LIMIT clause."""
        self._limit_value = count
        return self
    
    def offset(self, count: int) -> 'QueryBuilder':
        """Add OFFSET clause."""
        self._offset_value = count
        return self
    
    def build_select(self) -> Tuple[str, List]:
        """Build SELECT query."""
        query_parts = [
            f"SELECT {', '.join(self._select_fields)}",
            f"FROM {self.table_name}"
        ]
        
        # Add JOINs
        if self._join_clauses:
            query_parts.extend(self._join_clauses)
        
        # Add WHERE
        if self._where_conditions:
            query_parts.append(f"WHERE {' AND '.join(self._where_conditions)}")
        
        # Add GROUP BY
        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")
        
        # Add HAVING
        if self._having_conditions:
            query_parts.append(f"HAVING {' AND '.join(self._having_conditions)}")
        
        # Add ORDER BY
        if self._order_by:
            query_parts.append(f"ORDER BY {', '.join(self._order_by)}")
        
        # Add LIMIT and OFFSET
        if self._limit_value is not None:
            query_parts.append(f"LIMIT {self._limit_value}")
        
        if self._offset_value is not None:
            query_parts.append(f"OFFSET {self._offset_value}")
        
        return ' '.join(query_parts), self._params
    
    def execute(self) -> List[Any]:
        """Execute the query and return results."""
        query, params = self.build_select()
        return self.connection_manager.execute_query(query, params)
    
    def first(self) -> Optional[Any]:
        """Execute query and return first result."""
        results = self.limit(1).execute()
        return results[0] if results else None
    
    def count(self) -> int:
        """Get count of matching records."""
        # Save original select fields
        original_fields = self._select_fields
        
        # Change to COUNT(*)
        self._select_fields = ['COUNT(*)']
        
        try:
            query, params = self.build_select()
            result = self.connection_manager.execute_query(query, params)
            return result[0][0] if result else 0
        finally:
            # Restore original select fields
            self._select_fields = original_fields

# ==================== REPOSITORY PATTERN ====================

class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class RecordNotFoundError(RepositoryError):
    """Raised when a requested record is not found."""
    pass


class DuplicateRecordError(RepositoryError):
    """Raised when trying to create a duplicate record."""
    pass


class ConcurrencyError(RepositoryError):
    """Raised when optimistic locking fails."""
    pass


@runtime_checkable
class IRepository(Protocol, Generic[T]):
    """Repository interface for data access operations."""
    
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        ...
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        ...
    
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        ...
    
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        ...
    
    async def find_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Find all entities with pagination."""
        ...
    
    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Find entities by criteria."""
        ...
    
    async def count(self) -> int:
        """Count total entities."""
        ...
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists."""
        ...


class BaseRepository(Generic[T]):
    """Base repository implementation with common functionality."""
    
    def __init__(self, model_class: Type[T], connection_manager: ConnectionManager):
        self.model_class = model_class
        self.connection_manager = connection_manager
        self.table_name = self._get_table_name()
    
    def _get_table_name(self) -> str:
        """Get table name from model class."""
        return getattr(
            self.model_class, 
            '__table_name__', 
            self.model_class.__name__.lower() + 's'
        )
    
    def _get_field_mappings(self) -> Dict[str, str]:
        """Get field to column mappings."""
        return getattr(self.model_class, '_field_mappings', {})
    
    def _map_field_to_column(self, field_name: str) -> str:
        """Map model field name to database column name."""
        mappings = self._get_field_mappings()
        return mappings.get(field_name, field_name)
    
    def _build_insert_query(self, entity: T) -> Tuple[str, List[Any]]:
        """Build INSERT query for entity."""
        data = entity.to_dict(include_none=False)
        columns = [self._map_field_to_column(k) for k in data.keys()]
        placeholders = ['?' if self.connection_manager.config.engine == 'sqlite' else '%s'] * len(columns)
        
        query = f"""
        INSERT INTO {self.table_name} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """
        
        return query.strip(), list(data.values())
    
    def _build_update_query(self, entity: T) -> Tuple[str, List[Any]]:
        """Build UPDATE query for entity."""
        data = entity.to_dict(include_none=False)
        entity_id = data.pop('id')
        
        # Build SET clause
        set_clauses = []
        values = []
        placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
        
        for field_name, value in data.items():
            column_name = self._map_field_to_column(field_name)
            set_clauses.append(f"{column_name} = {placeholder}")
            values.append(value)
        
        # Add WHERE clause for ID
        values.append(entity_id)
        
        query = f"""
        UPDATE {self.table_name}
        SET {', '.join(set_clauses)}
        WHERE id = {placeholder}
        """
        
        return query.strip(), values
    
    def _build_select_query(self, where_clause: str = "", params: List[Any] = None) -> Tuple[str, List[Any]]:
        """Build SELECT query."""
        query = f"SELECT * FROM {self.table_name}"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        return query, params or []
    
    def _row_to_entity(self, row: Any) -> T:
        """Convert database row to entity."""
        if hasattr(row, 'keys'):  # sqlite3.Row or dict-like
            data = dict(row)
        else:  # tuple
            # Get column names from model fields
            field_names = list(self.model_class._fields.keys())
            data = dict(zip(field_names, row))
        
        return self.model_class.from_dict(data)
    
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        # Update audit fields
        if hasattr(entity, '_update_audit_fields'):
            entity._update_audit_fields()
        
        # Validate entity
        if not entity.validate():
            errors = entity.get_validation_errors()
            raise ValidationError(f"Validation failed: {[str(e) for e in errors]}")
        
        try:
            query, params = self._build_insert_query(entity)
            rows_affected = self.connection_manager.execute_command(query, params)
            
            if rows_affected == 0:
                raise RepositoryError("Failed to create entity")
            
            # Reset dirty tracking
            entity.reset_dirty_tracking()
            
            return entity
            
        except Exception as e:
            if "UNIQUE constraint failed" in str(e) or "duplicate key" in str(e):
                raise DuplicateRecordError(f"Entity with ID {entity.id} already exists")
            raise RepositoryError(f"Failed to create entity: {e}")
    
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID."""
        try:
            placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
            query = f"SELECT * FROM {self.table_name} WHERE id = {placeholder}"
            
            results = self.connection_manager.execute_query(query, (entity_id,))
            
            if not results:
                return None
            
            return self._row_to_entity(results[0])
            
        except Exception as e:
            raise RepositoryError(f"Failed to get entity by ID {entity_id}: {e}")
    
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        # Check if entity exists
        existing = await self.get_by_id(entity.id)
        if not existing:
            raise RecordNotFoundError(f"Entity with ID {entity.id} not found")
        
        # Optimistic locking check
        if hasattr(entity, 'version') and hasattr(existing, 'version'):
            if entity.version != existing.version:
                raise ConcurrencyError(
                    f"Entity has been modified by another user. "
                    f"Expected version {entity.version}, found {existing.version}"
                )
        
        # Update audit fields
        if hasattr(entity, '_update_audit_fields'):
            entity._update_audit_fields()
        
        # Validate entity
        if not entity.validate():
            errors = entity.get_validation_errors()
            raise ValidationError(f"Validation failed: {[str(e) for e in errors]}")
        
        try:
            query, params = self._build_update_query(entity)
            rows_affected = self.connection_manager.execute_command(query, params)
            
            if rows_affected == 0:
                raise RecordNotFoundError(f"Entity with ID {entity.id} not found")
            
            # Reset dirty tracking
            entity.reset_dirty_tracking()
            
            return entity
            
        except Exception as e:
            if isinstance(e, (RecordNotFoundError, ConcurrencyError)):
                raise
            raise RepositoryError(f"Failed to update entity: {e}")
    
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity (hard delete)."""
        try:
            placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
            query = f"DELETE FROM {self.table_name} WHERE id = {placeholder}"
            
            rows_affected = self.connection_manager.execute_command(query, (entity_id,))
            
            return rows_affected > 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to delete entity {entity_id}: {e}")
    
    async def soft_delete(self, entity_id: str, user_id: Optional[str] = None) -> bool:
        """Soft delete an entity."""
        entity = await self.get_by_id(entity_id)
        if not entity:
            return False
        
        if hasattr(entity, 'soft_delete'):
            entity.soft_delete(user_id)
            await self.update(entity)
            return True
        
        # Fallback to hard delete if soft delete not supported
        return await self.delete(entity_id)
    
    async def find_all(self, skip: int = 0, limit: int = 100, include_deleted: bool = False) -> List[T]:
        """Find all entities with pagination."""
        try:
            query = f"SELECT * FROM {self.table_name}"
            params = []
            
            # Exclude soft deleted records by default
            if not include_deleted and hasattr(self.model_class, 'is_deleted'):
                placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
                query += f" WHERE is_deleted = {placeholder}"
                params.append(False)
            
            query += f" LIMIT {limit} OFFSET {skip}"
            
            results = self.connection_manager.execute_query(query, params)
            
            return [self._row_to_entity(row) for row in results]
            
        except Exception as e:
            raise RepositoryError(f"Failed to find entities: {e}")
    
    async def find_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Find entities by criteria."""
        try:
            where_clauses = []
            params = []
            placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
            
            for field_name, value in criteria.items():
                column_name = self._map_field_to_column(field_name)
                where_clauses.append(f"{column_name} = {placeholder}")
                params.append(value)
            
            where_clause = " AND ".join(where_clauses) if where_clauses else ""
            query, query_params = self._build_select_query(where_clause, params)
            
            results = self.connection_manager.execute_query(query, query_params)
            
            return [self._row_to_entity(row) for row in results]
            
        except Exception as e:
            raise RepositoryError(f"Failed to find entities by criteria: {e}")
    
    async def count(self, include_deleted: bool = False) -> int:
        """Count total entities."""
        try:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            params = []
            
            # Exclude soft deleted records by default
            if not include_deleted and hasattr(self.model_class, 'is_deleted'):
                placeholder = '?' if self.connection_manager.config.engine == 'sqlite' else '%s'
                query += f" WHERE is_deleted = {placeholder}"
                params.append(False)
            
            results = self.connection_manager.execute_query(query, params)
            
            return results[0][0] if results else 0
            
        except Exception as e:
            raise RepositoryError(f"Failed to count entities: {e}")
    
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists."""
        entity = await self.get_by_id(entity_id)
        return entity is not None
    
    def query(self) -> QueryBuilder:
        """Get a query builder for advanced queries."""
        return QueryBuilder(self.table_name, self.connection_manager)


# ==================== UNIT OF WORK PATTERN ====================

class UnitOfWork:
    """Unit of Work pattern implementation for transaction management."""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self._repositories: Dict[Type, BaseRepository] = {}
        self._new_entities: List[BaseModel] = []
        self._dirty_entities: List[BaseModel] = []
        self._removed_entities: List[str] = []
        self._connection = None
        self._in_transaction = False
    
    def register_new(self, entity: BaseModel) -> None:
        """Register a new entity for creation."""
        if entity not in self._new_entities:
            self._new_entities.append(entity)
    
    def register_dirty(self, entity: BaseModel) -> None:
        """Register a dirty entity for update."""
        if entity not in self._dirty_entities and entity.is_dirty():
            self._dirty_entities.append(entity)
    
    def register_removed(self, entity_id: str) -> None:
        """Register an entity for removal."""
        if entity_id not in self._removed_entities:
            self._removed_entities.append(entity_id)
    
    def get_repository(self, model_class: Type[T]) -> BaseRepository[T]:
        """Get or create repository for model class."""
        if model_class not in self._repositories:
            self._repositories[model_class] = BaseRepository(model_class, self.connection_manager)
        
        return self._repositories[model_class]
    
    async def commit(self) -> None:
        """Commit all changes in a transaction."""
        if not (self._new_entities or self._dirty_entities or self._removed_entities):
            return  # Nothing to commit
        
        try:
            await self._begin_transaction()
            
            # Process new entities
            for entity in self._new_entities:
                repository = self.get_repository(entity.__class__)
                await repository.create(entity)
            
            # Process dirty entities
            for entity in self._dirty_entities:
                repository = self.get_repository(entity.__class__)
                await repository.update(entity)
            
            # Process removed entities
            for entity_id in self._removed_entities:
                # Note: We need entity type to get the right repository
                # This is a limitation of the current design
                pass
            
            await self._commit_transaction()
            self._clear_tracking()
            
        except Exception as e:
            await self._rollback_transaction()
            raise RepositoryError(f"Failed to commit unit of work: {e}")
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        try:
            await self._rollback_transaction()
            self._clear_tracking()
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {e}")
    
    def _clear_tracking(self) -> None:
        """Clear all tracked changes."""
        self._new_entities.clear()
        self._dirty_entities.clear()
        self._removed_entities.clear()
    
    async def _begin_transaction(self) -> None:
        """Begin a database transaction."""
        if self._in_transaction:
            return
        
        # Note: This is a simplified implementation
        # In a real scenario, we'd need to handle transactions per connection type
        self._in_transaction = True
    
    async def _commit_transaction(self) -> None:
        """Commit the database transaction."""
        if not self._in_transaction:
            return
        
        # Note: Implementation depends on database type
        self._in_transaction = False
    
    async def _rollback_transaction(self) -> None:
        """Rollback the database transaction."""
        if not self._in_transaction:
            return
        
        # Note: Implementation depends on database type
        self._in_transaction = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()


# ==================== DOMAIN EVENTS ====================

class DomainEvent:
    """Base class for domain events."""
    
    def __init__(self, event_id: str = None, occurred_at: datetime = None):
        self.event_id = event_id or str(uuid.uuid4())
        self.occurred_at = occurred_at or datetime.now(timezone.utc)
        self.event_type = self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'occurred_at': self.occurred_at.isoformat(),
            'data': self.get_event_data()
        }
    
    def get_event_data(self) -> Dict[str, Any]:
        """Override in subclasses to provide event-specific data."""
        return {}


class EntityCreatedEvent(DomainEvent):
    """Event raised when an entity is created."""
    
    def __init__(self, entity: BaseModel, **kwargs):
        super().__init__(**kwargs)
        self.entity = entity
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity.__class__.__name__,
            'entity_id': self.entity.id,
            'entity_data': self.entity.to_dict()
        }


class EntityUpdatedEvent(DomainEvent):
    """Event raised when an entity is updated."""
    
    def __init__(self, entity: BaseModel, changes: Dict[str, Tuple[Any, Any]], **kwargs):
        super().__init__(**kwargs)
        self.entity = entity
        self.changes = changes
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity.__class__.__name__,
            'entity_id': self.entity.id,
            'changes': {
                field: {'old': old_val, 'new': new_val}
                for field, (old_val, new_val) in self.changes.items()
            }
        }


class EntityDeletedEvent(DomainEvent):
    """Event raised when an entity is deleted."""
    
    def __init__(self, entity_type: str, entity_id: str, **kwargs):
        super().__init__(**kwargs)
        self.entity_type = entity_type
        self.entity_id = entity_id
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            'entity_type': self.entity_type,
            'entity_id': self.entity_id
        }


class EventHandler(ABC):
    """Base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the domain event."""
        pass
    
    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can handle the event."""
        return True


class EventDispatcher:
    """Event dispatcher for domain events."""
    
    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type."""
        self._handlers[event_type].append(handler)
    
    def subscribe_global(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events."""
        self._global_handlers.append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def unsubscribe_global(self, handler: EventHandler) -> None:
        """Unsubscribe a global handler."""
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            pass
    
    async def dispatch(self, event: DomainEvent) -> None:
        """Dispatch an event to all registered handlers."""
        handlers_to_notify = []
        
        # Get specific handlers
        if event.event_type in self._handlers:
            handlers_to_notify.extend(self._handlers[event.event_type])
        
        # Add global handlers
        handlers_to_notify.extend(self._global_handlers)
        
        # Filter handlers that can handle this event
        applicable_handlers = [h for h in handlers_to_notify if h.can_handle(event)]
        
        # Handle events asynchronously
        tasks = [handler.handle(event) for handler in applicable_handlers]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ==================== CACHING ====================

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set key-value pair with optional timeout."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self, max_entries: int = 10000, cleanup_interval: int = 600):
        self.max_entries = max_entries
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, Tuple[Any, Optional[datetime]]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """Start the periodic cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
                await self._cleanup_lru()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if expiry and expiry <= now
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._access_times.pop(key, None)
    
    async def _cleanup_lru(self) -> None:
        """Remove least recently used entries if over limit."""
        async with self._lock:
            if len(self._cache) <= self.max_entries:
                return
            
            # Sort by access time and remove oldest
            sorted_keys = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            
            keys_to_remove = sorted_keys[:len(self._cache) - self.max_entries]
            
            for key, _ in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, expiry = self._cache[key]
            
            # Check if expired
            if expiry and expiry <= datetime.now(timezone.utc):
                del self._cache[key]
                self._access_times.pop(key, None)
                return None
            
            # Update access time
            self._access_times[key] = datetime.now(timezone.utc)
            
            return value
    
    async def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set key-value pair with optional timeout."""
        async with self._lock:
            expiry = None
            if timeout:
                expiry = datetime.now(timezone.utc) + timedelta(seconds=timeout)
            
            self._cache[key] = (value, expiry)
            self._access_times[key] = datetime.now(timezone.utc)
    
    async def delete(self, key: str) -> None:
        """Delete key."""
        async with self._lock:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.get(key) is not None
    
    def __del__(self):
        """Cancel cleanup task on deletion."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class CacheManager:
    """Cache manager that handles different backends."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._backend: Optional[CacheBackend] = None
        self._initialize_backend()
    
    def _initialize_backend(self) -> None:
        """Initialize the cache backend."""
        if not self.config.enabled:
            self._backend = None
            return
        
        if self.config.backend == 'memory':
            self._backend = MemoryCacheBackend(
                max_entries=self.config.max_entries,
                cleanup_interval=self.config.cleanup_interval
            )
        # Add Redis backend here if needed
        else:
            raise ValueError(f"Unsupported cache backend: {self.config.backend}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._backend:
            return None
        
        return await self._backend.get(key)
    
    async def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        """Set value in cache."""
        if not self._backend:
            return
        
        timeout = timeout or self.config.default_timeout
        await self._backend.set(key, value, timeout)
    
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self._backend:
            return
        
        await self._backend.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        if not self._backend:
            return
        
        await self._backend.clear()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._backend:
            return False
        
        return await self._backend.exists(key)
    
    def cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key."""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ':'.join(key_parts)


def cached(
    timeout: Optional[int] = None,
    key_prefix: str = 'cached',
    cache_manager: Optional[CacheManager] = None
):
    """Decorator for caching function results."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_manager:
                return await func(*args, **kwargs)
            
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            cache_key = ':'.join(key_parts)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, timeout)
            
            return result
        
        return wrapper
    
    return decorator


# ==================== BUSINESS DOMAIN MODELS ====================

class Currency(BaseModel):
    """Currency value object."""
    
    code: str = FieldDescriptor(
        'code', str,
        validators=[
            FieldValidator.required,
            FieldValidator.min_length(3),
            FieldValidator.max_length(3),
            FieldValidator.regex(r'^[A-Z]{3}$', "Currency code must be 3 uppercase letters")
        ],
        description="ISO 4217 currency code"
    )
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Currency name"
    )
    
    symbol: str = FieldDescriptor(
        'symbol', str,
        validators=[FieldValidator.max_length(5)],
        description="Currency symbol"
    )
    
    decimal_places: int = FieldDescriptor(
        'decimal_places', int, default=2,
        validators=[FieldValidator.numeric_range(0, 6)],
        description="Number of decimal places"
    )
    
    is_active: bool = FieldDescriptor(
        'is_active', bool, default=True,
        description="Whether the currency is active"
    )
    
    def format_amount(self, amount: Decimal) -> str:
        """Format amount with currency symbol."""
        rounded_amount = amount.quantize(
            Decimal('0.01') if self.decimal_places == 2 else 
            Decimal(f"0.{'0' * self.decimal_places}"),
            rounding=ROUND_HALF_UP
        )
        return f"{self.symbol}{rounded_amount}"
    
    def __str__(self) -> str:
        return f"{self.code} ({self.symbol})"


class Money:
    """Money value object that combines amount and currency."""
    
    def __init__(self, amount: Union[Decimal, float, int, str], currency: Union[Currency, str]):
        if isinstance(amount, (float, int, str)):
            self.amount = Decimal(str(amount))
        else:
            self.amount = amount
        
        if isinstance(currency, str):
            # In a real implementation, this would lookup the currency
            self.currency = Currency(code=currency, name=currency, symbol=currency)
        else:
            self.currency = currency
    
    def __add__(self, other: 'Money') -> 'Money':
        """Add two money amounts."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot add different currencies: {self.currency.code} and {other.currency.code}")
        
        return Money(self.amount + other.amount, self.currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """Subtract two money amounts."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot subtract different currencies: {self.currency.code} and {other.currency.code}")
        
        return Money(self.amount - other.amount, self.currency)
    
    def __mul__(self, factor: Union[Decimal, float, int]) -> 'Money':
        """Multiply money by a factor."""
        return Money(self.amount * Decimal(str(factor)), self.currency)
    
    def __truediv__(self, divisor: Union[Decimal, float, int]) -> 'Money':
        """Divide money by a divisor."""
        return Money(self.amount / Decimal(str(divisor)), self.currency)
    
    def __eq__(self, other: Any) -> bool:
        """Compare two money amounts."""
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount and self.currency.code == other.currency.code
    
    def __lt__(self, other: 'Money') -> bool:
        """Less than comparison."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot compare different currencies: {self.currency.code} and {other.currency.code}")
        return self.amount < other.amount
    
    def __le__(self, other: 'Money') -> bool:
        """Less than or equal comparison."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot compare different currencies: {self.currency.code} and {other.currency.code}")
        return self.amount <= other.amount
    
    def __gt__(self, other: 'Money') -> bool:
        """Greater than comparison."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot compare different currencies: {self.currency.code} and {other.currency.code}")
        return self.amount > other.amount
    
    def __ge__(self, other: 'Money') -> bool:
        """Greater than or equal comparison."""
        if self.currency.code != other.currency.code:
            raise ValueError(f"Cannot compare different currencies: {self.currency.code} and {other.currency.code}")
        return self.amount >= other.amount
    
    def is_zero(self) -> bool:
        """Check if amount is zero."""
        return self.amount == Decimal('0')
    
    def is_positive(self) -> bool:
        """Check if amount is positive."""
        return self.amount > Decimal('0')
    
    def is_negative(self) -> bool:
        """Check if amount is negative."""
        return self.amount < Decimal('0')
    
    def abs(self) -> 'Money':
        """Get absolute value."""
        return Money(abs(self.amount), self.currency)
    
    def round(self, decimal_places: Optional[int] = None) -> 'Money':
        """Round to specified decimal places."""
        places = decimal_places or self.currency.decimal_places
        quantizer = Decimal('0.1') ** places
        rounded_amount = self.amount.quantize(quantizer, rounding=ROUND_HALF_UP)
        return Money(rounded_amount, self.currency)
    
    def format(self) -> str:
        """Format money as string."""
        return self.currency.format_amount(self.amount)
    
    def __str__(self) -> str:
        return self.format()
    
    def __repr__(self) -> str:
        return f"Money({self.amount}, {self.currency.code})"


# ==================== ADDRESS VALUE OBJECT ====================

class Address(BaseModel):
    """Address value object for geographical locations."""
    
    street1: str = FieldDescriptor(
        'street1', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(200)
        ],
        description="Primary street address"
    )
    
    street2: Optional[str] = FieldDescriptor(
        'street2', str, nullable=True,
        validators=[FieldValidator.max_length(200)],
        description="Secondary street address (apartment, suite, etc.)"
    )
    
    city: str = FieldDescriptor(
        'city', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="City name"
    )
    
    state_province: str = FieldDescriptor(
        'state_province', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="State or province"
    )
    
    postal_code: str = FieldDescriptor(
        'postal_code', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(20)
        ],
        description="Postal or ZIP code"
    )
    
    country: str = FieldDescriptor(
        'country', str,
        validators=[
            FieldValidator.required,
            FieldValidator.min_length(2),
            FieldValidator.max_length(2),
            FieldValidator.regex(r'^[A-Z]{2}$', "Country must be 2-letter ISO code")
        ],
        description="ISO 3166-1 alpha-2 country code"
    )
    
    def format_single_line(self) -> str:
        """Format address as single line."""
        parts = [self.street1]
        
        if self.street2:
            parts.append(self.street2)
        
        parts.extend([
            f"{self.city}, {self.state_province} {self.postal_code}",
            self.country
        ])
        
        return ", ".join(parts)
    
    def format_multi_line(self) -> str:
        """Format address as multiple lines."""
        lines = [self.street1]
        
        if self.street2:
            lines.append(self.street2)
        
        lines.append(f"{self.city}, {self.state_province} {self.postal_code}")
        lines.append(self.country)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.format_single_line()


# ==================== CONTACT INFORMATION ====================

class EmailAddress:
    """Email address value object."""
    
    def __init__(self, email: str):
        self.email = email.lower().strip()
        self._validate()
    
    def _validate(self) -> None:
        """Validate email format."""
        if not self.email:
            raise ValueError("Email address cannot be empty")
        
        FieldValidator.email(self.email)
    
    @property
    def domain(self) -> str:
        """Get email domain."""
        return self.email.split('@')[1]
    
    @property
    def local_part(self) -> str:
        """Get local part of email."""
        return self.email.split('@')[0]
    
    def __str__(self) -> str:
        return self.email
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, EmailAddress):
            return self.email == other.email
        elif isinstance(other, str):
            return self.email == other.lower().strip()
        return False
    
    def __hash__(self) -> int:
        return hash(self.email)


class PhoneNumber:
    """Phone number value object."""
    
    def __init__(self, number: str, country_code: str = '+1'):
        self.raw_number = number.strip()
        self.country_code = country_code.strip()
        self.formatted_number = self._format_number()
    
    def _format_number(self) -> str:
        """Format phone number."""
        # Remove all non-digit characters
        digits_only = ''.join(c for c in self.raw_number if c.isdigit())
        
        # Basic validation
        if len(digits_only) < 7:
            raise ValueError("Phone number must have at least 7 digits")
        
        # For US numbers (simplified formatting)
        if self.country_code == '+1' and len(digits_only) == 10:
            return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
        
        return digits_only
    
    def __str__(self) -> str:
        return f"{self.country_code} {self.formatted_number}"
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PhoneNumber):
            return str(self) == str(other)
        return False
    
    def __hash__(self) -> int:
        return hash(str(self))


class ContactInfo(BaseModel):
    """Contact information aggregate."""
    
    primary_email: Optional[str] = FieldDescriptor(
        'primary_email', str, nullable=True,
        validators=[FieldValidator.email],
        description="Primary email address"
    )
    
    secondary_email: Optional[str] = FieldDescriptor(
        'secondary_email', str, nullable=True,
        validators=[FieldValidator.email],
        description="Secondary email address"
    )
    
    primary_phone: Optional[str] = FieldDescriptor(
        'primary_phone', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Primary phone number"
    )
    
    secondary_phone: Optional[str] = FieldDescriptor(
        'secondary_phone', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Secondary phone number"
    )
    
    mobile_phone: Optional[str] = FieldDescriptor(
        'mobile_phone', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Mobile phone number"
    )
    
    fax: Optional[str] = FieldDescriptor(
        'fax', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Fax number"
    )
    
    website: Optional[str] = FieldDescriptor(
        'website', str, nullable=True,
        validators=[FieldValidator.max_length(200)],
        description="Website URL"
    )
    
    def get_primary_email_obj(self) -> Optional[EmailAddress]:
        """Get primary email as EmailAddress object."""
        return EmailAddress(self.primary_email) if self.primary_email else None
    
    def get_secondary_email_obj(self) -> Optional[EmailAddress]:
        """Get secondary email as EmailAddress object."""
        return EmailAddress(self.secondary_email) if self.secondary_email else None
    
    def has_valid_email(self) -> bool:
        """Check if has at least one valid email."""
        return bool(self.primary_email or self.secondary_email)
    
    def has_valid_phone(self) -> bool:
        """Check if has at least one valid phone."""
        return bool(self.primary_phone or self.secondary_phone or self.mobile_phone)


# ==================== PERSON AND ORGANIZATION ENTITIES ====================

class PersonName(BaseModel):
    """Person name value object."""
    
    title: Optional[str] = FieldDescriptor(
        'title', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Title (Mr., Mrs., Dr., etc.)"
    )
    
    first_name: str = FieldDescriptor(
        'first_name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="First name"
    )
    
    middle_name: Optional[str] = FieldDescriptor(
        'middle_name', str, nullable=True,
        validators=[FieldValidator.max_length(50)],
        description="Middle name or initial"
    )
    
    last_name: str = FieldDescriptor(
        'last_name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="Last name"
    )
    
    suffix: Optional[str] = FieldDescriptor(
        'suffix', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Suffix (Jr., Sr., III, etc.)"
    )
    
    def full_name(self) -> str:
        """Get full name."""
        parts = []
        
        if self.title:
            parts.append(self.title)
        
        parts.append(self.first_name)
        
        if self.middle_name:
            parts.append(self.middle_name)
        
        parts.append(self.last_name)
        
        if self.suffix:
            parts.append(self.suffix)
        
        return " ".join(parts)
    
    def formal_name(self) -> str:
        """Get formal name (Last, First Middle)."""
        parts = [self.last_name + ","]
        
        if self.title:
            parts.append(self.title)
        
        parts.append(self.first_name)
        
        if self.middle_name:
            parts.append(self.middle_name)
        
        if self.suffix:
            parts.append(self.suffix)
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        return self.full_name()


class Person(BaseModel):
    """Person entity."""
    
    name: PersonName = FieldDescriptor(
        'name', PersonName,
        description="Person's name"
    )
    
    date_of_birth: Optional[date] = FieldDescriptor(
        'date_of_birth', date, nullable=True,
        description="Date of birth"
    )
    
    gender: Optional[str] = FieldDescriptor(
        'gender', str, nullable=True,
        validators=[FieldValidator.max_length(20)],
        description="Gender"
    )
    
    nationality: Optional[str] = FieldDescriptor(
        'nationality', str, nullable=True,
        validators=[FieldValidator.max_length(50)],
        description="Nationality"
    )
    
    contact_info: Optional[ContactInfo] = FieldDescriptor(
        'contact_info', ContactInfo, nullable=True,
        description="Contact information"
    )
    
    addresses: List[Address] = FieldDescriptor(
        'addresses', list, default=list,
        description="List of addresses"
    )
    
    notes: Optional[str] = FieldDescriptor(
        'notes', str, nullable=True,
        validators=[FieldValidator.max_length(2000)],
        description="Additional notes"
    )
    
    @property
    def age(self) -> Optional[int]:
        """Calculate age from date of birth."""
        if not self.date_of_birth:
            return None
        
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    def add_address(self, address: Address) -> None:
        """Add an address."""
        if not isinstance(self.addresses, list):
            self.addresses = []
        self.addresses.append(address)
        self._mark_dirty('addresses')
    
    def get_primary_address(self) -> Optional[Address]:
        """Get the first address as primary."""
        return self.addresses[0] if self.addresses else None
    
    def __str__(self) -> str:
        return str(self.name)


class Organization(BaseModel):
    """Organization entity."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(200)
        ],
        description="Organization name"
    )
    
    legal_name: Optional[str] = FieldDescriptor(
        'legal_name', str, nullable=True,
        validators=[FieldValidator.max_length(200)],
        description="Legal name of organization"
    )
    
    organization_type: Optional[str] = FieldDescriptor(
        'organization_type', str, nullable=True,
        validators=[FieldValidator.max_length(50)],
        description="Type of organization (Corporation, LLC, etc.)"
    )
    
    tax_id: Optional[str] = FieldDescriptor(
        'tax_id', str, nullable=True,
        validators=[FieldValidator.max_length(50)],
        description="Tax identification number"
    )
    
    industry: Optional[str] = FieldDescriptor(
        'industry', str, nullable=True,
        validators=[FieldValidator.max_length(100)],
        description="Industry classification"
    )
    
    founded_date: Optional[date] = FieldDescriptor(
        'founded_date', date, nullable=True,
        description="Date organization was founded"
    )
    
    contact_info: Optional[ContactInfo] = FieldDescriptor(
        'contact_info', ContactInfo, nullable=True,
        description="Contact information"
    )
    
    addresses: List[Address] = FieldDescriptor(
        'addresses', list, default=list,
        description="List of addresses"
    )
    
    notes: Optional[str] = FieldDescriptor(
        'notes', str, nullable=True,
        validators=[FieldValidator.max_length(2000)],
        description="Additional notes"
    )
    
    def add_address(self, address: Address) -> None:
        """Add an address."""
        if not isinstance(self.addresses, list):
            self.addresses = []
        self.addresses.append(address)
        self._mark_dirty('addresses')
    
    def get_headquarters_address(self) -> Optional[Address]:
        """Get the first address as headquarters."""
        return self.addresses[0] if self.addresses else None
    
    def __str__(self) -> str:
        return self.name


# ==================== USER MANAGEMENT ====================

class UserRole(enum.Enum):
    """User roles enumeration."""
    
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    GUEST = "guest"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def get_hierarchy(cls) -> Dict[str, int]:
        """Get role hierarchy levels."""
        return {
            cls.SUPER_ADMIN.value: 100,
            cls.ADMIN.value: 80,
            cls.MANAGER.value: 60,
            cls.USER.value: 40,
            cls.GUEST.value: 20
        }
    
    def level(self) -> int:
        """Get numerical level for role."""
        hierarchy = self.get_hierarchy()
        return hierarchy.get(self.value, 0)
    
    def can_access(self, required_role: 'UserRole') -> bool:
        """Check if this role can access functionality requiring another role."""
        return self.level() >= required_role.level()


class Permission(BaseModel):
    """Permission entity."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Permission name"
    )
    
    description: Optional[str] = FieldDescriptor(
        'description', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="Permission description"
    )
    
    resource: str = FieldDescriptor(
        'resource', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Resource this permission applies to"
    )
    
    action: str = FieldDescriptor(
        'action', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="Action allowed (create, read, update, delete, etc.)"
    )
    
    def __str__(self) -> str:
        return f"{self.resource}:{self.action}"


class User(BaseModel):
    """User entity."""
    
    username: str = FieldDescriptor(
        'username', str,
        validators=[
            FieldValidator.required,
            FieldValidator.min_length(3),
            FieldValidator.max_length(50),
            FieldValidator.regex(r'^[a-zA-Z0-9_-]+$', "Username can only contain letters, numbers, underscore, and dash")
        ],
        description="Unique username"
    )
    
    email: str = FieldDescriptor(
        'email', str,
        validators=[
            FieldValidator.required,
            FieldValidator.email
        ],
        description="Email address"
    )
    
    password_hash: str = FieldDescriptor(
        'password_hash', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(255)
        ],
        description="Hashed password"
    )
    
    first_name: str = FieldDescriptor(
        'first_name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="First name"
    )
    
    last_name: str = FieldDescriptor(
        'last_name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="Last name"
    )
    
    role: UserRole = FieldDescriptor(
        'role', UserRole, default=UserRole.USER,
        description="User role"
    )
    
    is_active: bool = FieldDescriptor(
        'is_active', bool, default=True,
        description="Whether user account is active"
    )
    
    is_verified: bool = FieldDescriptor(
        'is_verified', bool, default=False,
        description="Whether email is verified"
    )
    
    last_login: Optional[datetime] = FieldDescriptor(
        'last_login', datetime, nullable=True,
        description="Last login timestamp"
    )
    
    failed_login_attempts: int = FieldDescriptor(
        'failed_login_attempts', int, default=0,
        validators=[FieldValidator.numeric_range(0, 999)],
        description="Number of consecutive failed login attempts"
    )
    
    locked_until: Optional[datetime] = FieldDescriptor(
        'locked_until', datetime, nullable=True,
        description="Account locked until this timestamp"
    )
    
    permissions: List[str] = FieldDescriptor(
        'permissions', list, default=list,
        description="List of permission IDs"
    )
    
    preferences: Dict[str, Any] = FieldDescriptor(
        'preferences', dict, default=dict,
        description="User preferences as key-value pairs"
    )
    
    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock the user account for specified duration."""
        self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self._mark_dirty('locked_until')
    
    def unlock_account(self) -> None:
        """Unlock the user account."""
        self.locked_until = None
        self.failed_login_attempts = 0
        self._mark_dirty('locked_until')
        self._mark_dirty('failed_login_attempts')
    
    def record_login_attempt(self, success: bool, max_attempts: int = 5) -> None:
        """Record a login attempt."""
        if success:
            self.last_login = datetime.now(timezone.utc)
            self.failed_login_attempts = 0
            self._mark_dirty('last_login')
            self._mark_dirty('failed_login_attempts')
        else:
            self.failed_login_attempts += 1
            self._mark_dirty('failed_login_attempts')
            
            if self.failed_login_attempts >= max_attempts:
                self.lock_account()
    
    def has_permission(self, permission_id: str) -> bool:
        """Check if user has a specific permission."""
        return permission_id in self.permissions
    
    def add_permission(self, permission_id: str) -> None:
        """Add a permission to the user."""
        if permission_id not in self.permissions:
            self.permissions.append(permission_id)
            self._mark_dirty('permissions')
    
    def remove_permission(self, permission_id: str) -> None:
        """Remove a permission from the user."""
        if permission_id in self.permissions:
            self.permissions.remove(permission_id)
            self._mark_dirty('permissions')
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        if not isinstance(self.preferences, dict):
            self.preferences = {}
        self.preferences[key] = value
        self._mark_dirty('preferences')
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        if not isinstance(self.preferences, dict):
            return default
        return self.preferences.get(key, default)
    
    def __str__(self) -> str:
        return f"{self.username} ({self.email})"


# ==================== PASSWORD UTILITIES ====================

class PasswordManager:
    """Password management utilities."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength according to configuration."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt (simplified implementation)."""
        # In a real implementation, use bcrypt or similar
        salt = secrets.token_hex(16)
        password_bytes = password.encode('utf-8')
        salt_bytes = salt.encode('utf-8')
        
        # Simple hash for demonstration (use bcrypt in production)
        combined = salt_bytes + password_bytes
        hashed = hashlib.pbkdf2_hmac('sha256', combined, salt_bytes, 100000)
        
        return f"{salt}:{hashed.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, stored_hash = password_hash.split(':', 1)
            salt_bytes = salt.encode('utf-8')
            password_bytes = password.encode('utf-8')
            
            combined = salt_bytes + password_bytes
            computed_hash = hashlib.pbkdf2_hmac('sha256', combined, salt_bytes, 100000)
            
            return secrets.compare_digest(computed_hash.hex(), stored_hash)
        except Exception:
            return False
    
    def generate_password(self, length: int = 12) -> str:
        """Generate a secure random password."""
        import string
        
        chars = string.ascii_letters + string.digits
        if self.config.password_require_special:
            chars += "!@#$%^&*"
        
        password = ''.join(secrets.choice(chars) for _ in range(length))
        
        # Ensure password meets requirements
        is_valid, _ = self.validate_password_strength(password)
        if not is_valid:
            # Regenerate until we get a valid password (with max attempts)
            for _ in range(100):
                password = ''.join(secrets.choice(chars) for _ in range(length))
                is_valid, _ = self.validate_password_strength(password)
                if is_valid:
                    break
        
        return password


# ==================== SESSION MANAGEMENT ====================

class Session(BaseModel):
    """User session entity."""
    
    user_id: str = FieldDescriptor(
        'user_id', str,
        validators=[FieldValidator.required],
        description="ID of the user this session belongs to"
    )
    
    session_token: str = FieldDescriptor(
        'session_token', str,
        validators=[FieldValidator.required],
        description="Unique session token"
    )
    
    expires_at: datetime = FieldDescriptor(
        'expires_at', datetime,
        validators=[FieldValidator.required],
        description="Session expiration timestamp"
    )
    
    ip_address: Optional[str] = FieldDescriptor(
        'ip_address', str, nullable=True,
        validators=[FieldValidator.max_length(45)],  # IPv6 max length
        description="IP address of the client"
    )
    
    user_agent: Optional[str] = FieldDescriptor(
        'user_agent', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="User agent string from the client"
    )
    
    is_active: bool = FieldDescriptor(
        'is_active', bool, default=True,
        description="Whether the session is active"
    )
    
    last_activity: datetime = FieldDescriptor(
        'last_activity', datetime,
        default=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp"
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now(timezone.utc) >= self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired
    
    def extend_expiration(self, minutes: int) -> None:
        """Extend session expiration."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self._mark_dirty('expires_at')
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        self._mark_dirty('last_activity')
    
    def invalidate(self) -> None:
        """Invalidate the session."""
        self.is_active = False
        self._mark_dirty('is_active')
    
    @classmethod
    def create_new(
        cls,
        user_id: str,
        duration_minutes: int = 60,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> 'Session':
        """Create a new session."""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        
        return cls(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )


class SessionManager:
    """Session management service."""
    
    def __init__(self, config: SecurityConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self._active_sessions: Dict[str, Session] = {}
        self._lock = Lock()
    
    async def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Session:
        """Create a new user session."""
        session = Session.create_new(
            user_id=user_id,
            duration_minutes=self.config.session_timeout_minutes,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store in cache and memory
        await self._store_session(session)
        
        return session
    
    async def get_session(self, session_token: str) -> Optional[Session]:
        """Get session by token."""
        # Try memory first
        with self._lock:
            if session_token in self._active_sessions:
                session = self._active_sessions[session_token]
                if session.is_valid:
                    return session
                else:
                    # Remove invalid session
                    del self._active_sessions[session_token]
        
        # Try cache
        cache_key = self._get_cache_key(session_token)
        cached_session_data = await self.cache.get(cache_key)
        
        if cached_session_data:
            session = Session.from_dict(cached_session_data)
            if session.is_valid:
                # Store back in memory
                with self._lock:
                    self._active_sessions[session_token] = session
                return session
            else:
                # Remove invalid session from cache
                await self.cache.delete(cache_key)
        
        return None
    
    async def update_session_activity(self, session_token: str) -> bool:
        """Update session last activity."""
        session = await self.get_session(session_token)
        if not session:
            return False
        
        session.update_activity()
        
        # Extend expiration if more than half the timeout has passed
        time_since_created = datetime.now(timezone.utc) - (session.expires_at - timedelta(minutes=self.config.session_timeout_minutes))
        if time_since_created > timedelta(minutes=self.config.session_timeout_minutes // 2):
            session.extend_expiration(self.config.session_timeout_minutes)
        
        await self._store_session(session)
        return True
    
    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session."""
        session = await self.get_session(session_token)
        if not session:
            return False
        
        session.invalidate()
        
        # Remove from memory and cache
        with self._lock:
            self._active_sessions.pop(session_token, None)
        
        cache_key = self._get_cache_key(session_token)
        await self.cache.delete(cache_key)
        
        return True
    
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        invalidated_count = 0
        
        # Check memory sessions
        sessions_to_remove = []
        with self._lock:
            for token, session in self._active_sessions.items():
                if session.user_id == user_id:
                    sessions_to_remove.append(token)
        
        for token in sessions_to_remove:
            if await self.invalidate_session(token):
                invalidated_count += 1
        
        return invalidated_count
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        cleaned_count = 0
        
        # Clean memory sessions
        expired_tokens = []
        with self._lock:
            for token, session in self._active_sessions.items():
                if not session.is_valid:
                    expired_tokens.append(token)
        
        for token in expired_tokens:
            with self._lock:
                self._active_sessions.pop(token, None)
            
            cache_key = self._get_cache_key(token)
            await self.cache.delete(cache_key)
            cleaned_count += 1
        
        return cleaned_count
    
    async def _store_session(self, session: Session) -> None:
        """Store session in memory and cache."""
        with self._lock:
            self._active_sessions[session.session_token] = session
        
        cache_key = self._get_cache_key(session.session_token)
        cache_timeout = int((session.expires_at - datetime.now(timezone.utc)).total_seconds())
        
        if cache_timeout > 0:
            await self.cache.set(cache_key, session.to_dict(), cache_timeout)
    
    def _get_cache_key(self, session_token: str) -> str:
        """Get cache key for session token."""
        return f"session:{session_token}"
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions in memory."""
        with self._lock:
            return len(self._active_sessions)
    
    async def get_user_session_count(self, user_id: str) -> int:
        """Get count of active sessions for a user."""
        count = 0
        with self._lock:
            for session in self._active_sessions.values():
                if session.user_id == user_id and session.is_valid:
                    count += 1
        return count


# ==================== AUDIT LOG ====================

class AuditLogEntry(BaseModel):
    """Audit log entry entity."""
    
    user_id: Optional[str] = FieldDescriptor(
        'user_id', str, nullable=True,
        description="ID of the user who performed the action"
    )
    
    action: str = FieldDescriptor(
        'action', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Action that was performed"
    )
    
    resource_type: str = FieldDescriptor(
        'resource_type', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Type of resource affected"
    )
    
    resource_id: Optional[str] = FieldDescriptor(
        'resource_id', str, nullable=True,
        description="ID of the specific resource affected"
    )
    
    old_values: Optional[Dict[str, Any]] = FieldDescriptor(
        'old_values', dict, nullable=True,
        description="Old values before the change"
    )
    
    new_values: Optional[Dict[str, Any]] = FieldDescriptor(
        'new_values', dict, nullable=True,
        description="New values after the change"
    )
    
    ip_address: Optional[str] = FieldDescriptor(
        'ip_address', str, nullable=True,
        validators=[FieldValidator.max_length(45)],
        description="IP address of the client"
    )
    
    user_agent: Optional[str] = FieldDescriptor(
        'user_agent', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="User agent string"
    )
    
    session_id: Optional[str] = FieldDescriptor(
        'session_id', str, nullable=True,
        description="Session ID when action was performed"
    )
    
    success: bool = FieldDescriptor(
        'success', bool, default=True,
        description="Whether the action was successful"
    )
    
    error_message: Optional[str] = FieldDescriptor(
        'error_message', str, nullable=True,
        validators=[FieldValidator.max_length(1000)],
        description="Error message if action failed"
    )
    
    additional_data: Optional[Dict[str, Any]] = FieldDescriptor(
        'additional_data', dict, nullable=True,
        description="Additional contextual data"
    )
    
    def __str__(self) -> str:
        return f"{self.action} on {self.resource_type} by user {self.user_id or 'system'}"


class AuditLogger:
    """Service for logging audit events."""
    
    def __init__(self, repository: BaseRepository[AuditLogEntry]):
        self.repository = repository
    
    async def log_action(
        self,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """Log an audit event."""
        
        entry = AuditLogEntry(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            success=success,
            error_message=error_message,
            additional_data=additional_data
        )
        
        return await self.repository.create(entry)
    
    async def log_create(
        self,
        resource_type: str,
        resource_id: str,
        new_values: Dict[str, Any],
        user_id: Optional[str] = None,
        **kwargs
    ) -> AuditLogEntry:
        """Log a create action."""
        return await self.log_action(
            action="CREATE",
            resource_type=resource_type,
            resource_id=resource_id,
            new_values=new_values,
            user_id=user_id,
            **kwargs
        )
    
    async def log_update(
        self,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        user_id: Optional[str] = None,
        **kwargs
    ) -> AuditLogEntry:
        """Log an update action."""
        return await self.log_action(
            action="UPDATE",
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values,
            user_id=user_id,
            **kwargs
        )
    
    async def log_delete(
        self,
        resource_type: str,
        resource_id: str,
        old_values: Dict[str, Any],
        user_id: Optional[str] = None,
        **kwargs
    ) -> AuditLogEntry:
        """Log a delete action."""
        return await self.log_action(
            action="DELETE",
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            user_id=user_id,
            **kwargs
        )
    
    async def log_login(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> AuditLogEntry:
        """Log a login attempt."""
        return await self.log_action(
            action="LOGIN",
            resource_type="User",
            resource_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message
        )
    
    async def log_logout(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> AuditLogEntry:
        """Log a logout action."""
        return await self.log_action(
            action="LOGOUT",
            resource_type="User",
            resource_id=user_id,
            session_id=session_id,
            ip_address=ip_address
        )



# ==================== BUSINESS LOGIC SERVICES ====================

class ServiceError(Exception):
    """Base exception for service layer operations."""
    pass


class BusinessRuleViolationError(ServiceError):
    """Raised when a business rule is violated."""
    pass


class AuthorizationError(ServiceError):
    """Raised when user is not authorized to perform an action."""
    pass


class ServiceResult(Generic[T]):
    """Result wrapper for service operations."""
    
    def __init__(
        self, 
        success: bool, 
        data: Optional[T] = None, 
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error_message = error_message
        self.error_code = error_code
        self.metadata = metadata or {}
    
    @classmethod
    def success_result(cls, data: T, metadata: Optional[Dict[str, Any]] = None) -> 'ServiceResult[T]':
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)
    
    @classmethod
    def error_result(
        cls, 
        error_message: str, 
        error_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'ServiceResult[T]':
        """Create an error result."""
        return cls(success=False, error_message=error_message, error_code=error_code, metadata=metadata)
    
    def is_success(self) -> bool:
        """Check if the result is successful."""
        return self.success
    
    def is_error(self) -> bool:
        """Check if the result is an error."""
        return not self.success
    
    def get_data(self) -> T:
        """Get the data, raising an exception if this is an error result."""
        if self.is_error():
            raise ServiceError(self.error_message or "Operation failed")
        return self.data
    
    def get_data_or_none(self) -> Optional[T]:
        """Get the data or None if this is an error result."""
        return self.data if self.is_success() else None
    
    def __bool__(self) -> bool:
        """Boolean conversion returns success status."""
        return self.success


class BaseService(ABC):
    """Base service class with common functionality."""
    
    def __init__(
        self, 
        unit_of_work: UnitOfWork,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        self.uow = unit_of_work
        self.audit_logger = audit_logger
        self.event_dispatcher = event_dispatcher
        self._current_user_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
    
    def set_context(self, user_id: Optional[str], session_id: Optional[str] = None) -> None:
        """Set the current user context for the service."""
        self._current_user_id = user_id
        self._current_session_id = session_id
    
    async def _log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log an audit event."""
        if self.audit_logger:
            await self.audit_logger.log_action(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                old_values=old_values,
                new_values=new_values,
                user_id=self._current_user_id,
                session_id=self._current_session_id,
                success=success,
                error_message=error_message
            )
    
    async def _dispatch_event(self, event: DomainEvent) -> None:
        """Dispatch a domain event."""
        if self.event_dispatcher:
            await self.event_dispatcher.dispatch(event)
    
    def _check_authorization(self, required_permission: str) -> None:
        """Check if current user has required permission."""
        if not self._current_user_id:
            raise AuthorizationError("User not authenticated")
        
        # In a real implementation, check user permissions here
        # This is a simplified version
        pass
    
    async def _validate_business_rules(self, entity: BaseModel, operation: str) -> None:
        """Validate business rules for an entity."""
        # Override in subclasses to implement specific business rules
        pass


class UserService(BaseService):
    """Service for user management operations."""
    
    def __init__(
        self, 
        unit_of_work: UnitOfWork,
        password_manager: PasswordManager,
        session_manager: SessionManager,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None
    ):
        super().__init__(unit_of_work, audit_logger, event_dispatcher)
        self.password_manager = password_manager
        self.session_manager = session_manager
        self.user_repository = unit_of_work.get_repository(User)
    
    async def create_user(
        self, 
        username: str,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        role: UserRole = UserRole.USER
    ) -> ServiceResult[User]:
        """Create a new user account."""
        try:
            # Check if username or email already exists
            existing_users = await self.user_repository.find_by_criteria({
                'username': username
            })
            if existing_users:
                return ServiceResult.error_result(
                    "Username already exists",
                    error_code="USERNAME_EXISTS"
                )
            
            existing_users = await self.user_repository.find_by_criteria({
                'email': email
            })
            if existing_users:
                return ServiceResult.error_result(
                    "Email already exists", 
                    error_code="EMAIL_EXISTS"
                )
            
            # Validate password strength
            is_valid, errors = self.password_manager.validate_password_strength(password)
            if not is_valid:
                return ServiceResult.error_result(
                    f"Password validation failed: {', '.join(errors)}",
                    error_code="WEAK_PASSWORD"
                )
            
            # Create user
            user = User(
                username=username,
                email=email,
                password_hash=self.password_manager.hash_password(password),
                first_name=first_name,
                last_name=last_name,
                role=role
            )
            
            # Validate business rules
            await self._validate_business_rules(user, "CREATE")
            
            # Save user
            created_user = await self.user_repository.create(user)
            await self.uow.commit()
            
            # Log audit event
            await self._log_audit_event(
                action="CREATE_USER",
                resource_type="User",
                resource_id=created_user.id,
                new_values=created_user.to_dict(exclude_fields={'password_hash'})
            )
            
            # Dispatch event
            await self._dispatch_event(EntityCreatedEvent(created_user))
            
            return ServiceResult.success_result(created_user)
            
        except Exception as e:
            await self.uow.rollback()
            await self._log_audit_event(
                action="CREATE_USER",
                resource_type="User",
                success=False,
                error_message=str(e)
            )
            return ServiceResult.error_result(f"Failed to create user: {str(e)}")
    
    async def authenticate_user(
        self, 
        username_or_email: str, 
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> ServiceResult[Tuple[User, Session]]:
        """Authenticate a user and create a session."""
        try:
            # Find user by username or email
            user = None
            users = await self.user_repository.find_by_criteria({'username': username_or_email})
            if not users:
                users = await self.user_repository.find_by_criteria({'email': username_or_email})
            
            if users:
                user = users[0]
            
            # Check if user exists and is active
            if not user or not user.is_active:
                await self._log_audit_event(
                    action="LOGIN_FAILED",
                    resource_type="User",
                    success=False,
                    error_message="Invalid credentials"
                )
                return ServiceResult.error_result(
                    "Invalid credentials",
                    error_code="INVALID_CREDENTIALS"
                )
            
            # Check if account is locked
            if user.is_locked:
                await self._log_audit_event(
                    action="LOGIN_FAILED",
                    resource_type="User",
                    resource_id=user.id,
                    success=False,
                    error_message="Account is locked"
                )
                return ServiceResult.error_result(
                    "Account is temporarily locked",
                    error_code="ACCOUNT_LOCKED"
                )
            
            # Verify password
            if not self.password_manager.verify_password(password, user.password_hash):
                # Record failed attempt
                user.record_login_attempt(success=False)
                await self.user_repository.update(user)
                await self.uow.commit()
                
                await self._log_audit_event(
                    action="LOGIN_FAILED",
                    resource_type="User",
                    resource_id=user.id,
                    success=False,
                    error_message="Invalid password"
                )
                
                return ServiceResult.error_result(
                    "Invalid credentials",
                    error_code="INVALID_CREDENTIALS"
                )
            
            # Successful authentication
            user.record_login_attempt(success=True)
            await self.user_repository.update(user)
            
            # Create session
            session = await self.session_manager.create_session(
                user_id=user.id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            await self.uow.commit()
            
            # Log successful login
            await self._log_audit_event(
                action="LOGIN_SUCCESS",
                resource_type="User",
                resource_id=user.id
            )
            
            return ServiceResult.success_result((user, session))
            
        except Exception as e:
            await self.uow.rollback()
            return ServiceResult.error_result(f"Authentication failed: {str(e)}")
    
    async def logout_user(self, session_token: str) -> ServiceResult[bool]:
        """Logout a user by invalidating their session."""
        try:
            success = await self.session_manager.invalidate_session(session_token)
            
            if success:
                await self._log_audit_event(
                    action="LOGOUT",
                    resource_type="User",
                    resource_id=self._current_user_id
                )
            
            return ServiceResult.success_result(success)
            
        except Exception as e:
            return ServiceResult.error_result(f"Logout failed: {str(e)}")
    
    async def change_password(
        self, 
        user_id: str, 
        old_password: str, 
        new_password: str
    ) -> ServiceResult[bool]:
        """Change user password."""
        try:
            self._check_authorization("change_password")
            
            # Get user
            user = await self.user_repository.get_by_id(user_id)
            if not user:
                return ServiceResult.error_result("User not found", error_code="USER_NOT_FOUND")
            
            # Verify current password
            if not self.password_manager.verify_password(old_password, user.password_hash):
                return ServiceResult.error_result(
                    "Current password is incorrect",
                    error_code="INVALID_PASSWORD"
                )
            
            # Validate new password
            is_valid, errors = self.password_manager.validate_password_strength(new_password)
            if not is_valid:
                return ServiceResult.error_result(
                    f"New password validation failed: {', '.join(errors)}",
                    error_code="WEAK_PASSWORD"
                )
            
            # Update password
            old_hash = user.password_hash
            user.password_hash = self.password_manager.hash_password(new_password)
            
            await self.user_repository.update(user)
            await self.uow.commit()
            
            # Log audit event
            await self._log_audit_event(
                action="CHANGE_PASSWORD",
                resource_type="User",
                resource_id=user.id,
                old_values={'password_changed': True},
                new_values={'password_changed': True}
            )
            
            return ServiceResult.success_result(True)
            
        except Exception as e:
            await self.uow.rollback()
            await self._log_audit_event(
                action="CHANGE_PASSWORD",
                resource_type="User",
                resource_id=user_id,
                success=False,
                error_message=str(e)
            )
            return ServiceResult.error_result(f"Failed to change password: {str(e)}")
    
    async def update_user_profile(
        self, 
        user_id: str, 
        updates: Dict[str, Any]
    ) -> ServiceResult[User]:
        """Update user profile information."""
        try:
            self._check_authorization("update_user")
            
            # Get user
            user = await self.user_repository.get_by_id(user_id)
            if not user:
                return ServiceResult.error_result("User not found", error_code="USER_NOT_FOUND")
            
            # Store old values for audit
            old_values = user.to_dict(exclude_fields={'password_hash'})
            
            # Apply updates
            allowed_fields = {'first_name', 'last_name', 'email', 'preferences'}
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(user, field, value)
            
            # Validate business rules
            await self._validate_business_rules(user, "UPDATE")
            
            # Save changes
            updated_user = await self.user_repository.update(user)
            await self.uow.commit()
            
            # Log audit event
            new_values = updated_user.to_dict(exclude_fields={'password_hash'})
            await self._log_audit_event(
                action="UPDATE_USER",
                resource_type="User",
                resource_id=user.id,
                old_values=old_values,
                new_values=new_values
            )
            
            # Dispatch event
            changes = updated_user.get_changed_values()
            await self._dispatch_event(EntityUpdatedEvent(updated_user, changes))
            
            return ServiceResult.success_result(updated_user)
            
        except Exception as e:
            await self.uow.rollback()
            await self._log_audit_event(
                action="UPDATE_USER",
                resource_type="User",
                resource_id=user_id,
                success=False,
                error_message=str(e)
            )
            return ServiceResult.error_result(f"Failed to update user: {str(e)}")


# ==================== NOTIFICATION SYSTEM ====================

class NotificationChannel(enum.Enum):
    """Notification delivery channels."""
    
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    WEBHOOK = "webhook"


class NotificationPriority(enum.Enum):
    """Notification priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationTemplate(BaseModel):
    """Notification template entity."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Template name"
    )
    
    subject: str = FieldDescriptor(
        'subject', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(200)
        ],
        description="Notification subject template"
    )
    
    body: str = FieldDescriptor(
        'body', str,
        validators=[FieldValidator.required],
        description="Notification body template"
    )
    
    channel: NotificationChannel = FieldDescriptor(
        'channel', NotificationChannel,
        description="Notification channel"
    )
    
    variables: List[str] = FieldDescriptor(
        'variables', list, default=list,
        description="List of template variables"
    )
    
    is_active: bool = FieldDescriptor(
        'is_active', bool, default=True,
        description="Whether template is active"
    )
    
    def render(self, variables: Dict[str, Any]) -> Tuple[str, str]:
        """Render template with provided variables."""
        subject = self.subject
        body = self.body
        
        # Simple template variable substitution
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            subject = subject.replace(placeholder, str(var_value))
            body = body.replace(placeholder, str(var_value))
        
        return subject, body


class Notification(BaseModel):
    """Notification entity."""
    
    recipient_id: str = FieldDescriptor(
        'recipient_id', str,
        validators=[FieldValidator.required],
        description="ID of the notification recipient"
    )
    
    sender_id: Optional[str] = FieldDescriptor(
        'sender_id', str, nullable=True,
        description="ID of the notification sender"
    )
    
    channel: NotificationChannel = FieldDescriptor(
        'channel', NotificationChannel,
        description="Notification channel"
    )
    
    priority: NotificationPriority = FieldDescriptor(
        'priority', NotificationPriority, default=NotificationPriority.NORMAL,
        description="Notification priority"
    )
    
    subject: str = FieldDescriptor(
        'subject', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(200)
        ],
        description="Notification subject"
    )
    
    body: str = FieldDescriptor(
        'body', str,
        validators=[FieldValidator.required],
        description="Notification body"
    )
    
    recipient_address: str = FieldDescriptor(
        'recipient_address', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(200)
        ],
        description="Recipient address (email, phone, etc.)"
    )
    
    scheduled_at: Optional[datetime] = FieldDescriptor(
        'scheduled_at', datetime, nullable=True,
        description="When notification should be sent"
    )
    
    sent_at: Optional[datetime] = FieldDescriptor(
        'sent_at', datetime, nullable=True,
        description="When notification was actually sent"
    )
    
    status: str = FieldDescriptor(
        'status', str, default='pending',
        validators=[FieldValidator.max_length(20)],
        description="Notification status (pending, sent, failed, cancelled)"
    )
    
    attempts: int = FieldDescriptor(
        'attempts', int, default=0,
        validators=[FieldValidator.numeric_range(0, 10)],
        description="Number of delivery attempts"
    )
    
    error_message: Optional[str] = FieldDescriptor(
        'error_message', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="Error message if delivery failed"
    )
    
    metadata: Dict[str, Any] = FieldDescriptor(
        'metadata', dict, default=dict,
        description="Additional metadata"
    )
    
    @property
    def is_pending(self) -> bool:
        """Check if notification is pending."""
        return self.status == 'pending'
    
    @property
    def is_sent(self) -> bool:
        """Check if notification was sent."""
        return self.status == 'sent'
    
    @property
    def is_failed(self) -> bool:
        """Check if notification failed."""
        return self.status == 'failed'
    
    @property
    def should_send_now(self) -> bool:
        """Check if notification should be sent now."""
        if not self.is_pending:
            return False
        
        if self.scheduled_at is None:
            return True
        
        return datetime.now(timezone.utc) >= self.scheduled_at
    
    def mark_sent(self) -> None:
        """Mark notification as sent."""
        self.status = 'sent'
        self.sent_at = datetime.now(timezone.utc)
        self._mark_dirty('status')
        self._mark_dirty('sent_at')
    
    def mark_failed(self, error_message: str) -> None:
        """Mark notification as failed."""
        self.status = 'failed'
        self.error_message = error_message
        self.attempts += 1
        self._mark_dirty('status')
        self._mark_dirty('error_message')
        self._mark_dirty('attempts')
    
    def retry(self) -> None:
        """Reset notification for retry."""
        if self.attempts < 10:  # Max retry limit
            self.status = 'pending'
            self.error_message = None
            self._mark_dirty('status')
            self._mark_dirty('error_message')


class NotificationService(BaseService):
    """Service for managing notifications."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        email_config: EmailConfig,
        audit_logger: Optional[AuditLogger] = None
    ):
        super().__init__(unit_of_work, audit_logger)
        self.email_config = email_config
        self.notification_repository = unit_of_work.get_repository(Notification)
        self.template_repository = unit_of_work.get_repository(NotificationTemplate)
    
    async def send_notification(
        self,
        recipient_id: str,
        template_name: str,
        variables: Dict[str, Any],
        channel: NotificationChannel = NotificationChannel.EMAIL,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        recipient_address: Optional[str] = None
    ) -> ServiceResult[Notification]:
        """Send a notification using a template."""
        try:
            # Get template
            templates = await self.template_repository.find_by_criteria({
                'name': template_name,
                'channel': channel,
                'is_active': True
            })
            
            if not templates:
                return ServiceResult.error_result(
                    f"Template '{template_name}' not found for channel {channel.value}",
                    error_code="TEMPLATE_NOT_FOUND"
                )
            
            template = templates[0]
            
            # Render template
            subject, body = template.render(variables)
            
            # Determine recipient address if not provided
            if not recipient_address:
                recipient_address = await self._get_recipient_address(recipient_id, channel)
                if not recipient_address:
                    return ServiceResult.error_result(
                        f"No {channel.value} address found for recipient {recipient_id}",
                        error_code="NO_RECIPIENT_ADDRESS"
                    )
            
            # Create notification
            notification = Notification(
                recipient_id=recipient_id,
                sender_id=self._current_user_id,
                channel=channel,
                priority=priority,
                subject=subject,
                body=body,
                recipient_address=recipient_address,
                scheduled_at=scheduled_at
            )
            
            # Save notification
            created_notification = await self.notification_repository.create(notification)
            
            # Send immediately if not scheduled
            if not scheduled_at:
                await self._deliver_notification(created_notification)
                await self.notification_repository.update(created_notification)
            
            await self.uow.commit()
            
            # Log audit event
            await self._log_audit_event(
                action="SEND_NOTIFICATION",
                resource_type="Notification",
                resource_id=created_notification.id,
                new_values={
                    'recipient_id': recipient_id,
                    'template': template_name,
                    'channel': channel.value
                }
            )
            
            return ServiceResult.success_result(created_notification)
            
        except Exception as e:
            await self.uow.rollback()
            await self._log_audit_event(
                action="SEND_NOTIFICATION",
                resource_type="Notification",
                success=False,
                error_message=str(e)
            )
            return ServiceResult.error_result(f"Failed to send notification: {str(e)}")
    
    async def _get_recipient_address(self, recipient_id: str, channel: NotificationChannel) -> Optional[str]:
        """Get recipient address for the specified channel."""
        # In a real implementation, look up user's contact information
        # This is a simplified version
        if channel == NotificationChannel.EMAIL:
            user_repo = self.uow.get_repository(User)
            user = await user_repo.get_by_id(recipient_id)
            return user.email if user else None
        
        return None
    
    async def _deliver_notification(self, notification: Notification) -> None:
        """Deliver a notification through the appropriate channel."""
        try:
            if notification.channel == NotificationChannel.EMAIL:
                await self._send_email(notification)
            elif notification.channel == NotificationChannel.SMS:
                await self._send_sms(notification)
            elif notification.channel == NotificationChannel.PUSH:
                await self._send_push(notification)
            elif notification.channel == NotificationChannel.IN_APP:
                await self._send_in_app(notification)
            elif notification.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(notification)
            else:
                raise ValueError(f"Unsupported notification channel: {notification.channel}")
            
            notification.mark_sent()
            
        except Exception as e:
            notification.mark_failed(str(e))
            logger.error(f"Failed to deliver notification {notification.id}: {e}")
    
    async def _send_email(self, notification: Notification) -> None:
        """Send email notification."""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = f"{self.email_config.from_name} <{self.email_config.from_address}>"
            msg['To'] = notification.recipient_address
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.body, 'plain'))
            
            # Send email
            context = ssl.create_default_context()
            
            if self.email_config.use_ssl:
                server = smtplib.SMTP_SSL(self.email_config.smtp_server, self.email_config.smtp_port, context=context)
            else:
                server = smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port)
                if self.email_config.use_tls:
                    server.starttls(context=context)
            
            if self.email_config.username and self.email_config.password:
                server.login(self.email_config.username, self.email_config.password)
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            raise Exception(f"Email delivery failed: {e}")
    
    async def _send_sms(self, notification: Notification) -> None:
        """Send SMS notification."""
        # Placeholder for SMS implementation
        raise NotImplementedError("SMS delivery not implemented")
    
    async def _send_push(self, notification: Notification) -> None:
        """Send push notification."""
        # Placeholder for push notification implementation
        raise NotImplementedError("Push notification delivery not implemented")
    
    async def _send_in_app(self, notification: Notification) -> None:
        """Send in-app notification."""
        # For in-app notifications, just mark as sent
        # The notification will be displayed when user checks their notifications
        pass
    
    async def _send_webhook(self, notification: Notification) -> None:
        """Send webhook notification."""
        # Placeholder for webhook implementation
        raise NotImplementedError("Webhook delivery not implemented")
    
    async def process_scheduled_notifications(self) -> ServiceResult[int]:
        """Process all scheduled notifications that are ready to send."""
        try:
            # Find pending notifications that should be sent now
            pending_notifications = await self.notification_repository.find_by_criteria({
                'status': 'pending'
            })
            
            processed_count = 0
            
            for notification in pending_notifications:
                if notification.should_send_now:
                    await self._deliver_notification(notification)
                    await self.notification_repository.update(notification)
                    processed_count += 1
            
            await self.uow.commit()
            
            return ServiceResult.success_result(processed_count)
            
        except Exception as e:
            await self.uow.rollback()
            return ServiceResult.error_result(f"Failed to process scheduled notifications: {str(e)}")
    
    async def get_user_notifications(
        self, 
        user_id: str, 
        unread_only: bool = False,
        limit: int = 50
    ) -> ServiceResult[List[Notification]]:
        """Get notifications for a user."""
        try:
            criteria = {'recipient_id': user_id}
            
            if unread_only:
                criteria['status'] = 'sent'
                # In a real implementation, track read status
            
            # Use query builder for more complex query with limit
            query = self.notification_repository.query()
            query.where("recipient_id = ?", user_id)
            
            if unread_only:
                query.where("status = ?", 'sent')
            
            query.order_by('created_at', 'DESC').limit(limit)
            
            results = query.execute()
            notifications = [self.notification_repository._row_to_entity(row) for row in results]
            
            return ServiceResult.success_result(notifications)
            
        except Exception as e:
            return ServiceResult.error_result(f"Failed to get user notifications: {str(e)}")


# ==================== SYSTEM HEALTH AND MONITORING ====================

class HealthCheck:
    """System health check utilities."""
    
    def __init__(self, connection_manager: ConnectionManager, cache_manager: CacheManager):
        self.connection_manager = connection_manager
        self.cache_manager = cache_manager
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = datetime.now()
            
            # Test basic connectivity
            is_connected = self.connection_manager.test_connection()
            
            # Measure response time
            end_time = datetime.now()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy' if is_connected else 'unhealthy',
                'connected': is_connected,
                'response_time_ms': response_time_ms,
                'database_info': self.connection_manager.get_database_info()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    async def check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            if not self.cache_manager.config.enabled:
                return {
                    'status': 'disabled',
                    'enabled': False
                }
            
            start_time = datetime.now()
            
            # Test cache operations
            test_key = f"health_check_{int(start_time.timestamp())}"
            test_value = "test_data"
            
            await self.cache_manager.set(test_key, test_value, 60)
            retrieved_value = await self.cache_manager.get(test_key)
            await self.cache_manager.delete(test_key)
            
            end_time = datetime.now()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            
            is_working = retrieved_value == test_value
            
            return {
                'status': 'healthy' if is_working else 'unhealthy',
                'enabled': True,
                'working': is_working,
                'response_time_ms': response_time_ms,
                'backend': self.cache_manager.config.backend
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'enabled': True,
                'working': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            database_health = await self.check_database_health()
            cache_health = await self.check_cache_health()
            
            # Determine overall status
            components_healthy = (
                database_health['status'] in ('healthy', 'disabled') and
                cache_health['status'] in ('healthy', 'disabled')
            )
            
            overall_status = 'healthy' if components_healthy else 'unhealthy'
            
            return {
                'status': overall_status,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'components': {
                    'database': database_health,
                    'cache': cache_health
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }


class SystemMetrics:
    """System metrics collection and reporting."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._start_time = datetime.now(timezone.utc)
    
    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        timestamp = datetime.now(timezone.utc)
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append({
            'value': value,
            'timestamp': timestamp,
            'tags': tags or {}
        })
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        self.record_metric(name, 1, tags)
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self.record_metric(name, duration_ms, tags)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        summary = {
            'uptime_seconds': (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            'metrics_count': len(self._metrics),
            'metrics': {}
        }
        
        for name, values in self._metrics.items():
            if values:
                numeric_values = [v['value'] for v in values if isinstance(v['value'], (int, float))]
                if numeric_values:
                    summary['metrics'][name] = {
                        'count': len(values),
                        'latest': values[-1]['value'],
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': sum(numeric_values) / len(numeric_values)
                    }
        
        return summary
    
    def clear_metrics(self, older_than_minutes: int = 60) -> None:
        """Clear old metrics to prevent memory buildup."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        
        for name in list(self._metrics.keys()):
            self._metrics[name] = [
                metric for metric in self._metrics[name]
                if metric['timestamp'] > cutoff_time
            ]
            
            # Remove empty metrics
            if not self._metrics[name]:
                del self._metrics[name]


# Global metrics instance
system_metrics = SystemMetrics()


def timed_operation(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to time operations and record metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = await func(*args, **kwargs)
                success_tags = {**(tags or {}), 'status': 'success'}
                return result
            except Exception as e:
                success_tags = {**(tags or {}), 'status': 'error', 'error_type': type(e).__name__}
                raise
            finally:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                system_metrics.record_timing(metric_name, duration_ms, success_tags)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                success_tags = {**(tags or {}), 'status': 'success'}
                return result
            except Exception as e:
                success_tags = {**(tags or {}), 'status': 'error', 'error_type': type(e).__name__}
                raise
            finally:
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                system_metrics.record_timing(metric_name, duration_ms, success_tags)
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ==================== APPLICATION FACTORY ====================

class AutoERPApplication:
    """Main application class that wires together all components."""
    
    def __init__(self, config: AutoERPConfig):
        self.config = config
        self.connection_manager: Optional[ConnectionManager] = None
        self.cache_manager: Optional[CacheManager] = None
        self.event_dispatcher: Optional[EventDispatcher] = None
        self.session_manager: Optional[SessionManager] = None
        self.password_manager: Optional[PasswordManager] = None
        self.health_checker: Optional[HealthCheck] = None
        
        # Services
        self.user_service: Optional[UserService] = None
        self.notification_service: Optional[NotificationService] = None
        
        # Repositories
        self._repositories: Dict[Type, BaseRepository] = {}
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the application and all its components."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing AutoERP application...")
            
            # Initialize core components
            self.connection_manager = ConnectionManager(self.config.database)
            self.cache_manager = CacheManager(self.config.cache)
            self.event_dispatcher = EventDispatcher()
            self.password_manager = PasswordManager(self.config.security)
            
            # Initialize session manager
            self.session_manager = SessionManager(self.config.security, self.cache_manager)
            
            # Initialize health checker
            self.health_checker = HealthCheck(self.connection_manager, self.cache_manager)
            
            # Initialize services
            await self._initialize_services()
            
            # Test connections
            await self._test_connections()
            
            self._initialized = True
            logger.info("AutoERP application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AutoERP application: {e}")
            raise
    
    async def _initialize_services(self) -> None:
        """Initialize business services."""
        # Create unit of work
        unit_of_work = UnitOfWork(self.connection_manager)
        
        # Create audit logger
        audit_repository = BaseRepository(AuditLogEntry, self.connection_manager)
        audit_logger = AuditLogger(audit_repository)
        
        # Initialize services
        self.user_service = UserService(
            unit_of_work=unit_of_work,
            password_manager=self.password_manager,
            session_manager=self.session_manager,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher
        )
        
        self.notification_service = NotificationService(
            unit_of_work=unit_of_work,
            email_config=self.config.email,
            audit_logger=audit_logger
        )
    
    async def _test_connections(self) -> None:
        """Test all external connections."""
        # Test database connection
        if not self.connection_manager.test_connection():
            raise RuntimeError("Database connection test failed")
        
        # Test cache if enabled
        if self.config.cache.enabled:
            test_key = "init_test"
            await self.cache_manager.set(test_key, "test_value", 60)
            test_value = await self.cache_manager.get(test_key)
            await self.cache_manager.delete(test_key)
            
            if test_value != "test_value":
                raise RuntimeError("Cache connection test failed")
    
    def get_repository(self, model_class: Type[T]) -> BaseRepository[T]:
        """Get repository for a model class."""
        if not self._initialized:
            raise RuntimeError("Application not initialized")
        
        if model_class not in self._repositories:
            self._repositories[model_class] = BaseRepository(model_class, self.connection_manager)
        
        return self._repositories[model_class]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get application health status."""
        if not self.health_checker:
            return {'status': 'not_initialized'}
        
        return await self.health_checker.get_system_health()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics."""
        return system_metrics.get_metrics_summary()
    
    async def cleanup(self) -> None:
        """Clean up application resources."""
        logger.info("Cleaning up AutoERP application...")
        
        try:
            # Clean up session manager
            if self.session_manager:
                await self.session_manager.cleanup_expired_sessions()
            
            # Close cache manager
            if self.cache_manager:
                await self.cache_manager.clear()
            
            # Close connection manager
            if self.connection_manager:
                self.connection_manager.close()
            
            logger.info("AutoERP application cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during application cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Configuration
    'AutoERPConfig', 'DatabaseConfig', 'SecurityConfig', 'CacheConfig',
    'EmailConfig', 'BusinessConfig', 'SystemConfig',
    
    # Base Models and Mixins
    'BaseModel', 'AuditMixin', 'SoftDeleteMixin', 'ValidationError',
    'FieldDescriptor', 'FieldValidator', 'ModelRegistry',
    
    # Connection Management
    'ConnectionManager', 'ConnectionPool',
    
    # Repository Pattern
    'IRepository', 'BaseRepository', 'RepositoryError',
    'RecordNotFoundError', 'DuplicateRecordError', 'ConcurrencyError',
    
    # Unit of Work
    'UnitOfWork',
    
    # Domain Events
    'DomainEvent', 'EntityCreatedEvent', 'EntityUpdatedEvent', 'EntityDeletedEvent',
    'EventHandler', 'EventDispatcher',
    
    # Caching
    'CacheBackend', 'MemoryCacheBackend', 'CacheManager', 'cached',
    
    # Value Objects
    'Currency', 'Money', 'Address', 'EmailAddress', 'PhoneNumber', 'ContactInfo',
    'PersonName',
    
    # Entities
    'Person', 'Organization', 'User', 'Permission', 'Session', 'AuditLogEntry',
    'NotificationTemplate', 'Notification',
    
    # Enums
    'UserRole', 'NotificationChannel', 'NotificationPriority',
    
    # Services
    'ServiceResult', 'BaseService', 'UserService', 'NotificationService',
    'ServiceError', 'BusinessRuleViolationError', 'AuthorizationError',
    
    # Security
    'PasswordManager', 'SessionManager',
    
    # Audit
    'AuditLogger',
    
    # Health and Monitoring
    'HealthCheck', 'SystemMetrics', 'system_metrics', 'timed_operation',
    
    # Application
    'AutoERPApplication',
    
    # Query Builder
    'QueryBuilder'
]

# ==================== ENHANCED BASE SERVICE WITH TRANSACTION & OBSERVER PATTERN ====================

class ServiceObserver(ABC):
    """Observer interface for service operations."""
    
    @abstractmethod
    async def on_before_operation(self, operation: str, context: Dict[str, Any]) -> None:
        """Called before a service operation."""
        pass
    
    @abstractmethod
    async def on_after_operation(self, operation: str, context: Dict[str, Any], result: Any) -> None:
        """Called after a successful service operation."""
        pass
    
    @abstractmethod
    async def on_operation_error(self, operation: str, context: Dict[str, Any], error: Exception) -> None:
        """Called when a service operation fails."""
        pass


class LoggingObserver(ServiceObserver):
    """Observer that logs service operations."""
    
    def __init__(self, logger_name: str = "service_observer"):
        self.logger = logging.getLogger(logger_name)
    
    async def on_before_operation(self, operation: str, context: Dict[str, Any]) -> None:
        self.logger.info(f"Starting operation: {operation}", extra={'context': context})
    
    async def on_after_operation(self, operation: str, context: Dict[str, Any], result: Any) -> None:
        self.logger.info(f"Completed operation: {operation}", extra={'context': context, 'success': True})
    
    async def on_operation_error(self, operation: str, context: Dict[str, Any], error: Exception) -> None:
        self.logger.error(f"Failed operation: {operation} - {str(error)}", 
                         extra={'context': context, 'error': str(error)})


class MetricsObserver(ServiceObserver):
    """Observer that records metrics for service operations."""
    
    async def on_before_operation(self, operation: str, context: Dict[str, Any]) -> None:
        system_metrics.increment_counter(f"service.{operation}.started")
    
    async def on_after_operation(self, operation: str, context: Dict[str, Any], result: Any) -> None:
        system_metrics.increment_counter(f"service.{operation}.success")
    
    async def on_operation_error(self, operation: str, context: Dict[str, Any], error: Exception) -> None:
        system_metrics.increment_counter(f"service.{operation}.error", 
                                       {'error_type': type(error).__name__})


class EnhancedBaseService(ABC):
    """Enhanced base service with transaction support, observer pattern, and thread pool handling."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ):
        self.uow = unit_of_work
        self.audit_logger = audit_logger
        self.event_dispatcher = event_dispatcher
        self.thread_pool = thread_pool_executor or ThreadPoolExecutor(max_workers=4)
        
        # Observer pattern
        self._observers: List[ServiceObserver] = []
        
        # Context
        self._current_user_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._operation_context: Dict[str, Any] = {}
        
        # Add default observers
        self.add_observer(LoggingObserver())
        self.add_observer(MetricsObserver())
    
    def add_observer(self, observer: ServiceObserver) -> None:
        """Add an observer to the service."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: ServiceObserver) -> None:
        """Remove an observer from the service."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    async def _notify_observers_before(self, operation: str, context: Dict[str, Any]) -> None:
        """Notify all observers before operation."""
        for observer in self._observers:
            try:
                await observer.on_before_operation(operation, context)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    async def _notify_observers_after(self, operation: str, context: Dict[str, Any], result: Any) -> None:
        """Notify all observers after successful operation."""
        for observer in self._observers:
            try:
                await observer.on_after_operation(operation, context, result)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    async def _notify_observers_error(self, operation: str, context: Dict[str, Any], error: Exception) -> None:
        """Notify all observers of operation error."""
        for observer in self._observers:
            try:
                await observer.on_operation_error(operation, context, error)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    def set_context(self, user_id: Optional[str], session_id: Optional[str] = None, **kwargs) -> None:
        """Set the current operation context."""
        self._current_user_id = user_id
        self._current_session_id = session_id
        self._operation_context.update(kwargs)
    
    async def execute_with_transaction(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        *args,
        rollback_on_error: bool = True,
        **kwargs
    ) -> ServiceResult[Any]:
        """Execute an operation within a transaction with observer notifications."""
        
        # Prepare context
        context = {
            'operation_name': operation_name,
            'user_id': self._current_user_id,
            'session_id': self._current_session_id,
            'args': str(args),
            'kwargs': {k: str(v) for k, v in kwargs.items()},
            **self._operation_context
        }
        
        start_time = datetime.now()
        
        try:
            # Notify observers before operation
            await self._notify_observers_before(operation_name, context)
            
            # Execute operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                # Run sync operation in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, operation, *args, **kwargs
                )
            
            # Commit transaction
            await self.uow.commit()
            
            # Calculate timing
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            system_metrics.record_timing(f"service.{operation_name}", duration_ms)
            
            # Notify observers after successful operation
            await self._notify_observers_after(operation_name, context, result)
            
            return ServiceResult.success_result(result, {'duration_ms': duration_ms})
            
        except Exception as e:
            # Rollback transaction if requested
            if rollback_on_error:
                await self.uow.rollback()
            
            # Calculate timing
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            system_metrics.record_timing(f"service.{operation_name}", duration_ms, {'status': 'error'})
            
            # Notify observers of error
            await self._notify_observers_error(operation_name, context, e)
            
            # Log audit event if available
            if self.audit_logger:
                await self.audit_logger.log_action(
                    action=operation_name.upper(),
                    resource_type="Service",
                    user_id=self._current_user_id,
                    session_id=self._current_session_id,
                    success=False,
                    error_message=str(e)
                )
            
            return ServiceResult.error_result(
                f"Operation {operation_name} failed: {str(e)}",
                error_code=type(e).__name__,
                metadata={'duration_ms': duration_ms}
            )
    
    async def execute_in_background(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        *args,
        **kwargs
    ) -> asyncio.Task:
        """Execute an operation in the background using thread pool."""
        
        async def background_wrapper():
            return await self.execute_with_transaction(
                operation, operation_name, *args, **kwargs
            )
        
        return asyncio.create_task(background_wrapper())
    
    async def execute_parallel_operations(
        self,
        operations: List[Tuple[Callable, str, Tuple, Dict]],
        max_concurrency: int = 5
    ) -> List[ServiceResult]:
        """Execute multiple operations in parallel with limited concurrency."""
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_operation(operation, name, args, kwargs):
            async with semaphore:
                return await self.execute_with_transaction(operation, name, *args, **kwargs)
        
        tasks = [
            bounded_operation(op, name, args, kwargs)
            for op, name, args, kwargs in operations
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)


# ==================== DATA LOADER SERVICE ====================

class DataFormat(enum.Enum):
    """Supported data formats."""
    
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "excel"
    TSV = "tsv"
    PARQUET = "parquet"
    UNKNOWN = "unknown"


class DataSchema:
    """Data schema representation."""
    
    def __init__(self):
        self.columns: Dict[str, Type] = {}
        self.nullable_columns: Set[str] = set()
        self.primary_keys: List[str] = []
        self.relationships: Dict[str, str] = {}
        self.constraints: Dict[str, List[str]] = {}
    
    def add_column(self, name: str, data_type: Type, nullable: bool = True) -> None:
        """Add a column to the schema."""
        self.columns[name] = data_type
        if nullable:
            self.nullable_columns.add(name)
    
    def set_primary_key(self, columns: List[str]) -> None:
        """Set primary key columns."""
        self.primary_keys = columns
    
    def add_relationship(self, column: str, target_table: str) -> None:
        """Add a foreign key relationship."""
        self.relationships[column] = target_table
    
    def add_constraint(self, column: str, constraint: str) -> None:
        """Add a constraint to a column."""
        if column not in self.constraints:
            self.constraints[column] = []
        self.constraints[column].append(constraint)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            'columns': {name: dtype.__name__ for name, dtype in self.columns.items()},
            'nullable_columns': list(self.nullable_columns),
            'primary_keys': self.primary_keys,
            'relationships': self.relationships,
            'constraints': self.constraints
        }


class DataCleaningRule:
    """Data cleaning rule."""
    
    def __init__(
        self,
        name: str,
        column: Optional[str] = None,
        rule_type: str = "custom",
        parameters: Optional[Dict[str, Any]] = None,
        function: Optional[Callable] = None
    ):
        self.name = name
        self.column = column
        self.rule_type = rule_type
        self.parameters = parameters or {}
        self.function = function
    
    def apply(self, data: Any) -> Any:
        """Apply the cleaning rule to data."""
        if self.function:
            return self.function(data, **self.parameters)
        
        # Built-in cleaning rules
        if self.rule_type == "trim_whitespace":
            return str(data).strip() if data else data
        elif self.rule_type == "lowercase":
            return str(data).lower() if data else data
        elif self.rule_type == "uppercase":
            return str(data).upper() if data else data
        elif self.rule_type == "remove_nulls":
            return None if data in ('', 'NULL', 'null', 'N/A', 'n/a') else data
        elif self.rule_type == "standardize_phone":
            if data:
                digits = ''.join(c for c in str(data) if c.isdigit())
                return digits if len(digits) >= 10 else None
            return None
        elif self.rule_type == "standardize_email":
            if data and '@' in str(data):
                return str(data).lower().strip()
            return None
        
        return data


class DataLoaderService(EnhancedBaseService):
    """Service for loading and processing data from various sources."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None,
        batch_size: int = 1000
    ):
        super().__init__(unit_of_work, audit_logger, event_dispatcher, thread_pool_executor)
        self.batch_size = batch_size
        self._cleaning_rules: Dict[str, List[DataCleaningRule]] = {}
    
    async def detect_format(self, file_path: Union[str, Path], content: Optional[bytes] = None) -> DataFormat:
        """Detect file format from path or content."""
        
        def _detect_format_sync():
            path = Path(file_path) if isinstance(file_path, str) else file_path
            
            # Check file extension first
            extension = path.suffix.lower()
            format_map = {
                '.csv': DataFormat.CSV,
                '.json': DataFormat.JSON,
                '.xml': DataFormat.XML,
                '.xlsx': DataFormat.EXCEL,
                '.xls': DataFormat.EXCEL,
                '.tsv': DataFormat.TSV,
                '.parquet': DataFormat.PARQUET,
                '.txt': DataFormat.CSV  # Assume CSV for txt files
            }
            
            if extension in format_map:
                return format_map[extension]
            
            # If no extension or unknown, try to detect from content
            if content:
                content_str = content[:1000].decode('utf-8', errors='ignore')
                
                # JSON detection
                if content_str.strip().startswith(('{', '[')):
                    try:
                        json.loads(content_str)
                        return DataFormat.JSON
                    except:
                        pass
                
                # XML detection
                if content_str.strip().startswith('<'):
                    return DataFormat.XML
                
                # CSV/TSV detection (count delimiters)
                comma_count = content_str.count(',')
                tab_count = content_str.count('\t')
                
                if tab_count > comma_count:
                    return DataFormat.TSV
                elif comma_count > 0:
                    return DataFormat.CSV
            
            return DataFormat.UNKNOWN
        
        return await self.execute_with_transaction(
            _detect_format_sync,
            "detect_format"
        ).then(lambda r: r.get_data())
    
    async def detect_schema(
        self,
        file_path: Union[str, Path],
        data_format: DataFormat,
        sample_size: int = 100
    ) -> ServiceResult[DataSchema]:
        """Detect schema from data sample."""
        
        def _detect_schema_sync():
            schema = DataSchema()
            
            try:
                if data_format == DataFormat.CSV:
                    return self._detect_csv_schema(file_path, sample_size)
                elif data_format == DataFormat.JSON:
                    return self._detect_json_schema(file_path, sample_size)
                elif data_format == DataFormat.XML:
                    return self._detect_xml_schema(file_path, sample_size)
                elif data_format == DataFormat.EXCEL:
                    return self._detect_excel_schema(file_path, sample_size)
                else:
                    raise ValueError(f"Schema detection not supported for format: {data_format}")
                    
            except Exception as e:
                raise Exception(f"Schema detection failed: {str(e)}")
        
        return await self.execute_with_transaction(_detect_schema_sync, "detect_schema")
    
    def _detect_csv_schema(self, file_path: Union[str, Path], sample_size: int) -> DataSchema:
        """Detect schema from CSV file."""
        import csv
        
        schema = DataSchema()
        
        with open(file_path, 'r', encoding='utf-8') as file:
            # Detect delimiter
            sample = file.read(1024)
            file.seek(0)
            
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            # Sample data for type inference
            samples = []
            for i, row in enumerate(reader):
                if i >= sample_size:
                    break
                samples.append(row)
            
            if not samples:
                return schema
            
            # Infer types from samples
            for column in samples[0].keys():
                column_values = [row.get(column, '') for row in samples]
                non_empty_values = [v for v in column_values if v and str(v).strip()]
                
                if not non_empty_values:
                    schema.add_column(column, str, nullable=True)
                    continue
                
                # Type inference logic
                data_type = self._infer_column_type(non_empty_values)
                nullable = len(non_empty_values) < len(column_values)
                
                schema.add_column(column, data_type, nullable=nullable)
        
        return schema
    
    def _detect_json_schema(self, file_path: Union[str, Path], sample_size: int) -> DataSchema:
        """Detect schema from JSON file."""
        schema = DataSchema()
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list) and data:
            # Array of objects
            sample_objects = data[:sample_size]
            
            # Get all possible keys
            all_keys = set()
            for obj in sample_objects:
                if isinstance(obj, dict):
                    all_keys.update(obj.keys())
            
            # Infer types for each key
            for key in all_keys:
                values = []
                for obj in sample_objects:
                    if isinstance(obj, dict) and key in obj:
                        values.append(obj[key])
                
                if values:
                    data_type = self._infer_column_type(values)
                    nullable = len(values) < len(sample_objects)
                    schema.add_column(key, data_type, nullable=nullable)
        
        elif isinstance(data, dict):
            # Single object - treat keys as columns
            for key, value in data.items():
                data_type = type(value) if value is not None else str
                schema.add_column(key, data_type, nullable=value is None)
        
        return schema
    
    def _detect_xml_schema(self, file_path: Union[str, Path], sample_size: int) -> DataSchema:
        """Detect schema from XML file - simplified implementation."""
        schema = DataSchema()
        
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find repeating elements (likely records)
            child_tags = [child.tag for child in root]
            if child_tags:
                # Use the most common child tag as record type
                from collections import Counter
                most_common_tag = Counter(child_tags).most_common(1)[0][0]
                
                # Sample records
                records = [child for child in root if child.tag == most_common_tag][:sample_size]
                
                # Get all possible attributes and child elements
                all_fields = set()
                for record in records:
                    # Add attributes
                    all_fields.update(record.attrib.keys())
                    # Add child element tags
                    all_fields.update([child.tag for child in record])
                
                # Infer types (simplified - everything as string for XML)
                for field in all_fields:
                    schema.add_column(field, str, nullable=True)
        
        except ImportError:
            raise Exception("XML parsing requires xml module")
        except Exception as e:
            raise Exception(f"XML schema detection failed: {str(e)}")
        
        return schema
    
    def _detect_excel_schema(self, file_path: Union[str, Path], sample_size: int) -> DataSchema:
        """Detect schema from Excel file - simplified implementation."""
        schema = DataSchema()
        
        try:
            # This would require openpyxl or similar library
            # For now, return basic schema
            schema.add_column("data", str, nullable=True)
            return schema
        
        except Exception as e:
            raise Exception(f"Excel schema detection failed: {str(e)}")
    
    def _infer_column_type(self, values: List[Any]) -> Type:
        """Infer the most appropriate type for a list of values."""
        
        # Try int first
        try:
            for value in values:
                int(str(value))
            return int
        except (ValueError, TypeError):
            pass
        
        # Try float
        try:
            for value in values:
                float(str(value))
            return float
        except (ValueError, TypeError):
            pass
        
        # Try bool
        bool_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        if all(str(value).lower() in bool_values for value in values):
            return bool
        
        # Try date
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            if all(re.match(pattern, str(value).strip()) for value in values):
                return date
        
        # Default to string
        return str
    
    async def clean_data(
        self,
        data: List[Dict[str, Any]],
        cleaning_rules: Optional[List[DataCleaningRule]] = None
    ) -> ServiceResult[List[Dict[str, Any]]]:
        """Clean data using specified rules."""
        
        def _clean_data_sync():
            rules = cleaning_rules or []
            cleaned_data = []
            
            for record in data:
                cleaned_record = {}
                
                for column, value in record.items():
                    cleaned_value = value
                    
                    # Apply column-specific rules
                    column_rules = [r for r in rules if r.column == column or r.column is None]
                    
                    for rule in column_rules:
                        try:
                            cleaned_value = rule.apply(cleaned_value)
                        except Exception as e:
                            logger.warning(f"Cleaning rule '{rule.name}' failed for column '{column}': {e}")
                    
                    cleaned_record[column] = cleaned_value
                
                cleaned_data.append(cleaned_record)
            
            return cleaned_data
        
        return await self.execute_with_transaction(_clean_data_sync, "clean_data")
    
    def add_cleaning_rule(self, table_name: str, rule: DataCleaningRule) -> None:
        """Add a cleaning rule for a specific table."""
        if table_name not in self._cleaning_rules:
            self._cleaning_rules[table_name] = []
        
        self._cleaning_rules[table_name].append(rule)
    
    async def batch_insert(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        model_class: Optional[Type[BaseModel]] = None,
        conflict_resolution: str = "skip"  # skip, update, error
    ) -> ServiceResult[Dict[str, Any]]:
        """Insert data in batches with conflict resolution."""
        
        async def _batch_insert_async():
            total_records = len(data)
            successful_inserts = 0
            failed_inserts = 0
            errors = []
            
            # Get repository
            if model_class:
                repository = self.uow.get_repository(model_class)
            else:
                # Create a dynamic repository based on table name
                repository = BaseRepository(BaseModel, self.uow.connection_manager)
                repository.table_name = table_name
            
            # Process data in batches
            for i in range(0, total_records, self.batch_size):
                batch = data[i:i + self.batch_size]
                batch_results = await self._process_batch(
                    repository, batch, model_class, conflict_resolution
                )
                
                successful_inserts += batch_results['success_count']
                failed_inserts += batch_results['error_count']
                errors.extend(batch_results['errors'])
            
            return {
                'total_records': total_records,
                'successful_inserts': successful_inserts,
                'failed_inserts': failed_inserts,
                'errors': errors[:100],  # Limit error list
                'success_rate': successful_inserts / total_records if total_records > 0 else 0
            }
        
        return await self.execute_with_transaction(_batch_insert_async, "batch_insert")
    
    async def _process_batch(
        self,
        repository: BaseRepository,
        batch_data: List[Dict[str, Any]],
        model_class: Optional[Type[BaseModel]],
        conflict_resolution: str
    ) -> Dict[str, Any]:
        """Process a single batch of data."""
        
        success_count = 0
        error_count = 0
        errors = []
        
        for record_data in batch_data:
            try:
                if model_class:
                    # Create model instance
                    entity = model_class.from_dict(record_data)
                    
                    # Check for existing record if conflict resolution is needed
                    if conflict_resolution in ("skip", "update") and hasattr(entity, 'id'):
                        existing = await repository.get_by_id(entity.id)
                        
                        if existing:
                            if conflict_resolution == "skip":
                                continue  # Skip existing records
                            elif conflict_resolution == "update":
                                # Update existing record
                                for field, value in record_data.items():
                                    if hasattr(existing, field):
                                        setattr(existing, field, value)
                                await repository.update(existing)
                                success_count += 1
                                continue
                    
                    # Create new record
                    await repository.create(entity)
                    success_count += 1
                
                else:
                    # Raw SQL insert (simplified)
                    # In a real implementation, build proper SQL insert statement
                    success_count += 1
            
            except Exception as e:
                error_count += 1
                errors.append({
                    'record': record_data,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        return {
            'success_count': success_count,
            'error_count': error_count,
            'errors': errors
        }


# ==================== CRUD SERVICE ====================

class PaginationParams:
    """Pagination parameters."""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 50,
        max_page_size: int = 1000
    ):
        self.page = max(1, page)
        self.page_size = min(max(1, page_size), max_page_size)
        self.offset = (self.page - 1) * self.page_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'page': self.page,
            'page_size': self.page_size,
            'offset': self.offset
        }


class SortParams:
    """Sort parameters."""
    
    def __init__(self, field: str, direction: str = 'ASC'):
        self.field = field
        self.direction = direction.upper() if direction.upper() in ('ASC', 'DESC') else 'ASC'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'direction': self.direction
        }


class FilterParams:
    """Filter parameters."""
    
    def __init__(self, filters: Optional[Dict[str, Any]] = None):
        self.filters = filters or {}
    
    def add_filter(self, field: str, value: Any, operator: str = 'eq') -> None:
        """Add a filter condition."""
        if field not in self.filters:
            self.filters[field] = []
        
        self.filters[field].append({
            'value': value,
            'operator': operator
        })
    
    def to_dict(self) -> Dict[str, Any]:
        return {'filters': self.filters}


class PaginatedResult(Generic[T]):
    """Result with pagination information."""
    
    def __init__(
        self,
        items: List[T],
        total_count: int,
        pagination: PaginationParams
    ):
        self.items = items
        self.total_count = total_count
        self.pagination = pagination
    
    @property
    def total_pages(self) -> int:
        """Calculate total pages."""
        if self.pagination.page_size == 0:
            return 0
        return (self.total_count + self.pagination.page_size - 1) // self.pagination.page_size
    
    @property
    def has_next(self) -> bool:
        """Check if there are more pages."""
        return self.pagination.page < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        """Check if there are previous pages."""
        return self.pagination.page > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'items': [item.to_dict() if hasattr(item, 'to_dict') else item for item in self.items],
            'pagination': {
                'page': self.pagination.page,
                'page_size': self.pagination.page_size,
                'total_count': self.total_count,
                'total_pages': self.total_pages,
                'has_next': self.has_next,
                'has_previous': self.has_previous
            }
        }


class CRUDService(EnhancedBaseService, Generic[T]):
    """Generic CRUD service with pagination, filtering, and sorting."""
    
    def __init__(
        self,
        model_class: Type[T],
        unit_of_work: UnitOfWork,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ):
        super().__init__(unit_of_work, audit_logger, event_dispatcher, thread_pool_executor)
        self.model_class = model_class
        self.repository = unit_of_work.get_repository(model_class)
    
    async def create_record(self, data: Dict[str, Any]) -> ServiceResult[T]:
        """Create a new record."""
        
        async def _create_operation():
            # Create entity from data
            entity = self.model_class.from_dict(data)
            
            # Validate business rules
            await self._validate_create_rules(entity)
            
            # Create record
            created_entity = await self.repository.create(entity)
            
            # Log audit event
            if self.audit_logger:
                await self.audit_logger.log_create(
                    resource_type=self.model_class.__name__,
                    resource_id=created_entity.id,
                    new_values=created_entity.to_dict(),
                    user_id=self._current_user_id
                )
            
            # Dispatch event
            if self.event_dispatcher:
                await self.event_dispatcher.dispatch(EntityCreatedEvent(created_entity))
            
            return created_entity
        
        return await self.execute_with_transaction(_create_operation, "create_record")
    
    async def read_records(
        self,
        pagination: Optional[PaginationParams] = None,
        filters: Optional[FilterParams] = None,
        sort: Optional[SortParams] = None,
        include_deleted: bool = False
    ) -> ServiceResult[PaginatedResult[T]]:
        """Read records with pagination, filtering, and sorting."""
        
        async def _read_operation():
            # Set defaults
            pagination = pagination or PaginationParams()
            
            # Build query
            query = self.repository.query()
            
            # Apply filters
            if filters:
                await self._apply_filters(query, filters)
            
            # Exclude soft deleted records by default
            if not include_deleted and hasattr(self.model_class, 'is_deleted'):
                query.where("is_deleted = ?", False)
            
            # Get total count for pagination
            total_count = len(query.execute())  # Simplified count
            
            # Apply sorting
            if sort:
                query.order_by(sort.field, sort.direction)
            else:
                # Default sort by creation date if available
                if hasattr(self.model_class, 'created_at'):
                    query.order_by('created_at', 'DESC')
            
            # Apply pagination
            query.limit(pagination.page_size).offset(pagination.offset)
            
            # Execute query
            results = query.execute()
            entities = [self.repository._row_to_entity(row) for row in results]
            
            return PaginatedResult(entities, total_count, pagination)
        
        return await self.execute_with_transaction(_read_operation, "read_records")
    
    async def update_record(self, record_id: str, data: Dict[str, Any]) -> ServiceResult[T]:
        """Update an existing record."""
        
        async def _update_operation():
            # Get existing record
            existing_entity = await self.repository.get_by_id(record_id)
            if not existing_entity:
                raise RecordNotFoundError(f"Record with ID {record_id} not found")
            
            # Store old values for audit
            old_values = existing_entity.to_dict()
            
            # Update entity with new data
            for field, value in data.items():
                if hasattr(existing_entity, field):
                    setattr(existing_entity, field, value)
            
            # Validate business rules
            await self._validate_update_rules(existing_entity, data)
            
            # Update record
            updated_entity = await self.repository.update(existing_entity)
            
            # Log audit event
            if self.audit_logger:
                await self.audit_logger.log_update(
                    resource_type=self.model_class.__name__,
                    resource_id=record_id,
                    old_values=old_values,
                    new_values=updated_entity.to_dict(),
                    user_id=self._current_user_id
                )
            
            # Dispatch event
            if self.event_dispatcher:
                changes = updated_entity.get_changed_values()
                await self.event_dispatcher.dispatch(EntityUpdatedEvent(updated_entity, changes))
            
            return updated_entity
        
        return await self.execute_with_transaction(_update_operation, "update_record")
    
    async def delete_record(self, record_id: str, soft_delete: bool = True) -> ServiceResult[bool]:
        """Delete a record (soft or hard delete)."""
        
        async def _delete_operation():
            # Get existing record
            existing_entity = await self.repository.get_by_id(record_id)
            if not existing_entity:
                raise RecordNotFoundError(f"Record with ID {record_id} not found")
            
            # Store old values for audit
            old_values = existing_entity.to_dict()
            
            # Validate delete rules
            await self._validate_delete_rules(existing_entity)
            
            # Perform delete
            if soft_delete and hasattr(existing_entity, 'soft_delete'):
                existing_entity.soft_delete(self._current_user_id)
                await self.repository.update(existing_entity)
                deleted = True
            else:
                deleted = await self.repository.delete(record_id)
            
            # Log audit event
            if self.audit_logger and deleted:
                await self.audit_logger.log_delete(
                    resource_type=self.model_class.__name__,
                    resource_id=record_id,
                    old_values=old_values,
                    user_id=self._current_user_id
                )
            
            # Dispatch event
            if self.event_dispatcher and deleted:
                await self.event_dispatcher.dispatch(
                    EntityDeletedEvent(self.model_class.__name__, record_id)
                )
            
            return deleted
        
        return await self.execute_with_transaction(_delete_operation, "delete_record")
    
    async def _apply_filters(self, query: QueryBuilder, filters: FilterParams) -> None:
        """Apply filter parameters to query."""
        for field, conditions in filters.filters.items():
            for condition in conditions:
                operator = condition.get('operator', 'eq')
                value = condition['value']
                
                if operator == 'eq':
                    query.where(f"{field} = ?", value)
                elif operator == 'ne':
                    query.where(f"{field} != ?", value)
                elif operator == 'gt':
                    query.where(f"{field} > ?", value)
                elif operator == 'gte':
                    query.where(f"{field} >= ?", value)
                elif operator == 'lt':
                    query.where(f"{field} < ?", value)
                elif operator == 'lte':
                    query.where(f"{field} <= ?", value)
                elif operator == 'like':
                    query.where(f"{field} LIKE ?", f"%{value}%")
                elif operator == 'in':
                    if isinstance(value, (list, tuple)):
                        placeholders = ','.join(['?' for _ in value])
                        query.where(f"{field} IN ({placeholders})", *value)
    
    async def _validate_create_rules(self, entity: T) -> None:
        """Validate business rules for create operation."""
        # Override in subclasses for specific validation
        pass
    
    async def _validate_update_rules(self, entity: T, update_data: Dict[str, Any]) -> None:
        """Validate business rules for update operation."""
        # Override in subclasses for specific validation
        pass
    
    async def _validate_delete_rules(self, entity: T) -> None:
        """Validate business rules for delete operation."""
        # Override in subclasses for specific validation
        pass


# ==================== BUSINESS ENTITY SERVICES ====================

class UserCRUDService(CRUDService[User]):
    """CRUD service specifically for User entities."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        password_manager: PasswordManager,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ):
        super().__init__(User, unit_of_work, audit_logger, event_dispatcher, thread_pool_executor)
        self.password_manager = password_manager
    
    async def _validate_create_rules(self, entity: User) -> None:
        """Validate business rules for user creation."""
        # Check username uniqueness
        existing_users = await self.repository.find_by_criteria({'username': entity.username})
        if existing_users:
            raise BusinessRuleViolationError(f"Username '{entity.username}' already exists")
        
        # Check email uniqueness
        existing_users = await self.repository.find_by_criteria({'email': entity.email})
        if existing_users:
            raise BusinessRuleViolationError(f"Email '{entity.email}' already exists")
        
        # Validate password if it's being set
        if hasattr(entity, '_raw_password'):
            is_valid, errors = self.password_manager.validate_password_strength(entity._raw_password)
            if not is_valid:
                raise BusinessRuleViolationError(f"Password validation failed: {'; '.join(errors)}")
    
    async def _validate_update_rules(self, entity: User, update_data: Dict[str, Any]) -> None:
        """Validate business rules for user updates."""
        # Check username uniqueness if being changed
        if 'username' in update_data:
            existing_users = await self.repository.find_by_criteria({'username': update_data['username']})
            if existing_users and existing_users[0].id != entity.id:
                raise BusinessRuleViolationError(f"Username '{update_data['username']}' already exists")
        
        # Check email uniqueness if being changed
        if 'email' in update_data:
            existing_users = await self.repository.find_by_criteria({'email': update_data['email']})
            if existing_users and existing_users[0].id != entity.id:
                raise BusinessRuleViolationError(f"Email '{update_data['email']}' already exists")
    
    async def _validate_delete_rules(self, entity: User) -> None:
        """Validate business rules for user deletion."""
        # Don't allow deletion of super admin users
        if entity.role == UserRole.SUPER_ADMIN:
            super_admin_count = len(await self.repository.find_by_criteria({'role': UserRole.SUPER_ADMIN}))
            if super_admin_count <= 1:
                raise BusinessRuleViolationError("Cannot delete the last super admin user")


class OrganizationCRUDService(CRUDService[Organization]):
    """CRUD service specifically for Organization entities."""
    
    async def _validate_create_rules(self, entity: Organization) -> None:
        """Validate business rules for organization creation."""
        # Check name uniqueness
        existing_orgs = await self.repository.find_by_criteria({'name': entity.name})
        if existing_orgs:
            raise BusinessRuleViolationError(f"Organization '{entity.name}' already exists")
        
        # Check tax ID uniqueness if provided
        if entity.tax_id:
            existing_orgs = await self.repository.find_by_criteria({'tax_id': entity.tax_id})
            if existing_orgs:
                raise BusinessRuleViolationError(f"Tax ID '{entity.tax_id}' already exists")
    
    async def _validate_update_rules(self, entity: Organization, update_data: Dict[str, Any]) -> None:
        """Validate business rules for organization updates."""
        # Check name uniqueness if being changed
        if 'name' in update_data:
            existing_orgs = await self.repository.find_by_criteria({'name': update_data['name']})
            if existing_orgs and existing_orgs[0].id != entity.id:
                raise BusinessRuleViolationError(f"Organization '{update_data['name']}' already exists")
        
        # Check tax ID uniqueness if being changed
        if 'tax_id' in update_data and update_data['tax_id']:
            existing_orgs = await self.repository.find_by_criteria({'tax_id': update_data['tax_id']})
            if existing_orgs and existing_orgs[0].id != entity.id:
                raise BusinessRuleViolationError(f"Tax ID '{update_data['tax_id']}' already exists")


# ==================== WORKFLOW ENGINE ====================

class WorkflowState(enum.Enum):
    """Workflow execution states."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStep(BaseModel):
    """Individual step in a workflow."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Step name"
    )
    
    step_type: str = FieldDescriptor(
        'step_type', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(50)
        ],
        description="Type of step (action, condition, loop, etc.)"
    )
    
    configuration: Dict[str, Any] = FieldDescriptor(
        'configuration', dict, default=dict,
        description="Step configuration parameters"
    )
    
    dependencies: List[str] = FieldDescriptor(
        'dependencies', list, default=list,
        description="List of step names this step depends on"
    )
    
    timeout_seconds: Optional[int] = FieldDescriptor(
        'timeout_seconds', int, nullable=True,
        validators=[FieldValidator.numeric_range(1, 86400)],  # Max 24 hours
        description="Step timeout in seconds"
    )
    
    retry_count: int = FieldDescriptor(
        'retry_count', int, default=0,
        validators=[FieldValidator.numeric_range(0, 10)],
        description="Number of retry attempts"
    )
    
    is_critical: bool = FieldDescriptor(
        'is_critical', bool, default=True,
        description="Whether workflow should fail if this step fails"
    )


class WorkflowDefinition(BaseModel):
    """Workflow definition containing steps and configuration."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Workflow name"
    )
    
    description: Optional[str] = FieldDescriptor(
        'description', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="Workflow description"
    )
    
    steps: List[WorkflowStep] = FieldDescriptor(
        'steps', list, default=list,
        description="List of workflow steps"
    )
    
    timeout_seconds: Optional[int] = FieldDescriptor(
        'timeout_seconds', int, nullable=True,
        validators=[FieldValidator.numeric_range(1, 86400)],
        description="Overall workflow timeout"
    )
    
    max_concurrent_steps: int = FieldDescriptor(
        'max_concurrent_steps', int, default=5,
        validators=[FieldValidator.numeric_range(1, 50)],
        description="Maximum number of steps to run concurrently"
    )
    
    variables: Dict[str, Any] = FieldDescriptor(
        'variables', dict, default=dict,
        description="Workflow variables and their default values"
    )
    
    is_active: bool = FieldDescriptor(
        'is_active', bool, default=True,
        description="Whether the workflow is active"
    )
    
    def validate_workflow(self) -> Tuple[bool, List[str]]:
        """Validate the workflow definition."""
        errors = []
        
        if not self.steps:
            errors.append("Workflow must have at least one step")
            return False, errors
        
        step_names = [step.name for step in self.steps]
        
        # Check for duplicate step names
        if len(step_names) != len(set(step_names)):
            errors.append("Step names must be unique")
        
        # Check dependencies
        for step in self.steps:
            for dependency in step.dependencies:
                if dependency not in step_names:
                    errors.append(f"Step '{step.name}' depends on non-existent step '{dependency}'")
        
        # Check for circular dependencies (simplified)
        def has_circular_dependency(step_name: str, visited: Set[str] = None) -> bool:
            if visited is None:
                visited = set()
            
            if step_name in visited:
                return True
            
            visited.add(step_name)
            
            step = next((s for s in self.steps if s.name == step_name), None)
            if step:
                for dependency in step.dependencies:
                    if has_circular_dependency(dependency, visited.copy()):
                        return True
            
            return False
        
        for step in self.steps:
            if has_circular_dependency(step.name):
                errors.append(f"Circular dependency detected involving step '{step.name}'")
        
        return len(errors) == 0, errors


class WorkflowExecution(BaseModel):
    """Workflow execution instance."""
    
    workflow_id: str = FieldDescriptor(
        'workflow_id', str,
        validators=[FieldValidator.required],
        description="ID of the workflow definition"
    )
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Execution name"
    )
    
    state: WorkflowState = FieldDescriptor(
        'state', WorkflowState, default=WorkflowState.PENDING,
        description="Current execution state"
    )
    
    started_at: Optional[datetime] = FieldDescriptor(
        'started_at', datetime, nullable=True,
        description="When execution started"
    )
    
    completed_at: Optional[datetime] = FieldDescriptor(
        'completed_at', datetime, nullable=True,
        description="When execution completed"
    )
    
    variables: Dict[str, Any] = FieldDescriptor(
        'variables', dict, default=dict,
        description="Runtime variables"
    )
    
    step_results: Dict[str, Dict[str, Any]] = FieldDescriptor(
        'step_results', dict, default=dict,
        description="Results from completed steps"
    )
    
    error_message: Optional[str] = FieldDescriptor(
        'error_message', str, nullable=True,
        validators=[FieldValidator.max_length(1000)],
        description="Error message if execution failed"
    )
    
    progress_percentage: float = FieldDescriptor(
        'progress_percentage', float, default=0.0,
        validators=[FieldValidator.numeric_range(0.0, 100.0)],
        description="Execution progress percentage"
    )
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        return end_time - self.started_at
    
    @property
    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.state == WorkflowState.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.state in (WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED)
    
    def update_progress(self, completed_steps: int, total_steps: int) -> None:
        """Update execution progress."""
        if total_steps > 0:
            self.progress_percentage = (completed_steps / total_steps) * 100
        self._mark_dirty('progress_percentage')


class WorkflowStepHandler(ABC):
    """Abstract base class for workflow step handlers."""
    
    @abstractmethod
    async def execute(
        self, 
        step: WorkflowStep, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the workflow step."""
        pass
    
    @abstractmethod
    def get_step_type(self) -> str:
        """Get the step type this handler supports."""
        pass


class ActionStepHandler(WorkflowStepHandler):
    """Handler for action steps."""
    
    def get_step_type(self) -> str:
        return "action"
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action step."""
        action_type = step.configuration.get('action_type', 'noop')
        
        if action_type == 'noop':
            # No operation - just return success
            return {'status': 'success', 'message': 'No operation performed'}
        
        elif action_type == 'delay':
            # Delay for specified seconds
            delay_seconds = step.configuration.get('delay_seconds', 1)
            await asyncio.sleep(delay_seconds)
            return {'status': 'success', 'message': f'Delayed for {delay_seconds} seconds'}
        
        elif action_type == 'log':
            # Log a message
            message = step.configuration.get('message', 'Workflow step executed')
            logger.info(f"Workflow step '{step.name}': {message}")
            return {'status': 'success', 'message': f'Logged: {message}'}
        
        elif action_type == 'set_variable':
            # Set a variable in the context
            var_name = step.configuration.get('variable_name')
            var_value = step.configuration.get('variable_value')
            if var_name:
                context['variables'][var_name] = var_value
                return {'status': 'success', 'message': f'Set variable {var_name} = {var_value}'}
            else:
                raise ValueError("Variable name not specified")
        
        else:
            raise ValueError(f"Unknown action type: {action_type}")


class ConditionStepHandler(WorkflowStepHandler):
    """Handler for condition steps."""
    
    def get_step_type(self) -> str:
        return "condition"
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a condition step."""
        condition_type = step.configuration.get('condition_type', 'always_true')
        
        if condition_type == 'always_true':
            return {'status': 'success', 'result': True}
        
        elif condition_type == 'variable_equals':
            var_name = step.configuration.get('variable_name')
            expected_value = step.configuration.get('expected_value')
            
            if var_name in context['variables']:
                actual_value = context['variables'][var_name]
                result = actual_value == expected_value
                return {
                    'status': 'success', 
                    'result': result,
                    'message': f'Variable {var_name}: {actual_value} == {expected_value} -> {result}'
                }
            else:
                return {'status': 'success', 'result': False, 'message': f'Variable {var_name} not found'}
        
        else:
            raise ValueError(f"Unknown condition type: {condition_type}")


class WorkflowEngine(EnhancedBaseService):
    """Workflow execution engine."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ):
        super().__init__(unit_of_work, audit_logger, event_dispatcher, thread_pool_executor)
        self.definition_repository = unit_of_work.get_repository(WorkflowDefinition)
        self.execution_repository = unit_of_work.get_repository(WorkflowExecution)
        
        # Register default step handlers
        self._step_handlers: Dict[str, WorkflowStepHandler] = {}
        self.register_step_handler(ActionStepHandler())
        self.register_step_handler(ConditionStepHandler())
        
        # Active executions
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}
    
    def register_step_handler(self, handler: WorkflowStepHandler) -> None:
        """Register a step handler."""
        self._step_handlers[handler.get_step_type()] = handler
    
    async def create_workflow_definition(
        self, 
        definition: WorkflowDefinition
    ) -> ServiceResult[WorkflowDefinition]:
        """Create a new workflow definition."""
        
        async def _create_definition():
            # Validate workflow
            is_valid, errors = definition.validate_workflow()
            if not is_valid:
                raise BusinessRuleViolationError(f"Invalid workflow: {'; '.join(errors)}")
            
            # Check name uniqueness
            existing = await self.definition_repository.find_by_criteria({'name': definition.name})
            if existing:
                raise BusinessRuleViolationError(f"Workflow '{definition.name}' already exists")
            
            return await self.definition_repository.create(definition)
        
        return await self.execute_with_transaction(_create_definition, "create_workflow_definition")
    
    async def start_workflow_execution(
        self, 
        workflow_id: str, 
        execution_name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> ServiceResult[WorkflowExecution]:
        """Start a workflow execution."""
        
        async def _start_execution():
            # Get workflow definition
            definition = await self.definition_repository.get_by_id(workflow_id)
            if not definition:
                raise RecordNotFoundError(f"Workflow definition {workflow_id} not found")
            
            if not definition.is_active:
                raise BusinessRuleViolationError("Workflow is not active")
            
            # Create execution instance
            execution_variables = definition.variables.copy()
            if variables:
                execution_variables.update(variables)
            
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                name=execution_name,
                state=WorkflowState.PENDING,
                variables=execution_variables
            )
            
            # Save execution
            created_execution = await self.execution_repository.create(execution)
            
            # Start execution in background
            task = asyncio.create_task(self._execute_workflow(created_execution, definition))
            self._execution_tasks[created_execution.id] = task
            
            return created_execution
        
        return await self.execute_with_transaction(_start_execution, "start_workflow_execution")
    
    async def _execute_workflow(
        self, 
        execution: WorkflowExecution, 
        definition: WorkflowDefinition
    ) -> None:
        """Execute a workflow (internal method)."""
        
        try:
            # Mark as running
            execution.state = WorkflowState.RUNNING
            execution.started_at = datetime.now(timezone.utc)
            await self.execution_repository.update(execution)
            
            self._active_executions[execution.id] = execution
            
            # Prepare execution context
            context = {
                'execution_id': execution.id,
                'variables': execution.variables.copy(),
                'step_results': execution.step_results.copy()
            }
            
            # Execute steps
            await self._execute_steps(execution, definition, context)
            
            # Mark as completed
            execution.state = WorkflowState.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            execution.progress_percentage = 100.0
            
        except Exception as e:
            # Mark as failed
            execution.state = WorkflowState.FAILED
            execution.completed_at = datetime.now(timezone.utc)
            execution.error_message = str(e)
            logger.error(f"Workflow execution {execution.id} failed: {e}")
        
        finally:
            # Update execution
            await self.execution_repository.update(execution)
            
            # Clean up
            self._active_executions.pop(execution.id, None)
            self._execution_tasks.pop(execution.id, None)
    
    async def _execute_steps(
        self, 
        execution: WorkflowExecution, 
        definition: WorkflowDefinition, 
        context: Dict[str, Any]
    ) -> None:
        """Execute workflow steps with dependency resolution."""
        
        completed_steps = set()
        failed_steps = set()
        total_steps = len(definition.steps)
        
        # Create dependency graph
        step_map = {step.name: step for step in definition.steps}
        
        while len(completed_steps) < total_steps and not failed_steps:
            # Find steps ready to execute
            ready_steps = []
            
            for step in definition.steps:
                if (step.name not in completed_steps and 
                    step.name not in failed_steps and
                    all(dep in completed_steps for dep in step.dependencies)):
                    ready_steps.append(step)
            
            if not ready_steps:
                if failed_steps:
                    break  # Some steps failed
                else:
                    # Circular dependency or other issue
                    raise RuntimeError("No steps ready to execute - possible circular dependency")
            
            # Execute ready steps (with concurrency limit)
            semaphore = asyncio.Semaphore(definition.max_concurrent_steps)
            
            async def execute_step_with_semaphore(step: WorkflowStep):
                async with semaphore:
                    return await self._execute_single_step(step, context, execution)
            
            # Execute steps concurrently
            step_tasks = [execute_step_with_semaphore(step) for step in ready_steps]
            step_results = await asyncio.gather(*step_tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    if step.is_critical:
                        failed_steps.add(step.name)
                        raise result
                    else:
                        # Non-critical step failed, mark as completed with error
                        execution.step_results[step.name] = {
                            'status': 'failed',
                            'error': str(result),
                            'is_critical': False
                        }
                        completed_steps.add(step.name)
                else:
                    execution.step_results[step.name] = result
                    completed_steps.add(step.name)
            
            # Update progress
            execution.update_progress(len(completed_steps), total_steps)
            await self.execution_repository.update(execution)
        
        if failed_steps:
            raise RuntimeError(f"Critical steps failed: {failed_steps}")
    
    async def _execute_single_step(
        self, 
        step: WorkflowStep, 
        context: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        # Get step handler
        handler = self._step_handlers.get(step.step_type)
        if not handler:
            raise ValueError(f"No handler registered for step type: {step.step_type}")
        
        # Set up timeout
        timeout = step.timeout_seconds
        
        # Execute step with retries
        for attempt in range(step.retry_count + 1):
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        handler.execute(step, context),
                        timeout=timeout
                    )
                else:
                    result = await handler.execute(step, context)
                
                # Add execution metadata
                result.update({
                    'step_name': step.name,
                    'step_type': step.step_type,
                    'execution_id': execution.id,
                    'attempt': attempt + 1,
                    'executed_at': datetime.now(timezone.utc).isoformat()
                })
                
                return result
                
            except asyncio.TimeoutError:
                if attempt < step.retry_count:
                    logger.warning(f"Step {step.name} timed out, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Step {step.name} timed out after {step.retry_count + 1} attempts")
            
            except Exception as e:
                if attempt < step.retry_count:
                    logger.warning(f"Step {step.name} failed, retrying (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def get_execution_status(self, execution_id: str) -> ServiceResult[WorkflowExecution]:
        """Get the status of a workflow execution."""
        
        async def _get_status():
            # Check active executions first
            if execution_id in self._active_executions:
                return self._active_executions[execution_id]
            
            # Get from repository
            execution = await self.execution_repository.get_by_id(execution_id)
            if not execution:
                raise RecordNotFoundError(f"Workflow execution {execution_id} not found")
            
            return execution
        
        return await self.execute_with_transaction(_get_status, "get_execution_status")
    
    async def cancel_execution(self, execution_id: str) -> ServiceResult[bool]:
        """Cancel a running workflow execution."""
        
        async def _cancel_execution():
            # Get execution
            execution = await self.execution_repository.get_by_id(execution_id)
            if not execution:
                raise RecordNotFoundError(f"Workflow execution {execution_id} not found")
            
            if not execution.is_running:
                raise BusinessRuleViolationError("Execution is not running")
            
            # Cancel task if running
            task = self._execution_tasks.get(execution_id)
            if task and not task.done():
                task.cancel()
            
            # Update execution state
            execution.state = WorkflowState.CANCELLED
            execution.completed_at = datetime.now(timezone.utc)
            await self.execution_repository.update(execution)
            
            # Clean up
            self._active_executions.pop(execution_id, None)
            self._execution_tasks.pop(execution_id, None)
            
            return True
        
        return await self.execute_with_transaction(_cancel_execution, "cancel_execution")


# ==================== REPORTING SERVICE ====================

class ReportParameter:
    """Report parameter definition."""
    
    def __init__(
        self,
        name: str,
        parameter_type: Type,
        required: bool = False,
        default_value: Any = None,
        description: str = "",
        choices: Optional[List[Any]] = None
    ):
        self.name = name
        self.parameter_type = parameter_type
        self.required = required
        self.default_value = default_value
        self.description = description
        self.choices = choices or []
    
    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a parameter value."""
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None
        
        # Type validation
        if not isinstance(value, self.parameter_type):
            try:
                # Try to convert
                converted_value = self.parameter_type(value)
                value = converted_value
            except (ValueError, TypeError):
                return False, f"Parameter '{self.name}' must be of type {self.parameter_type.__name__}"
        
        # Choice validation
        if self.choices and value not in self.choices:
            return False, f"Parameter '{self.name}' must be one of: {self.choices}"
        
        return True, None


class ReportDefinition(BaseModel):
    """Report definition."""
    
    name: str = FieldDescriptor(
        'name', str,
        validators=[
            FieldValidator.required,
            FieldValidator.max_length(100)
        ],
        description="Report name"
    )
    
    description: Optional[str] = FieldDescriptor(
        'description', str, nullable=True,
        validators=[FieldValidator.max_length(500)],
        description="Report description"
    )
    
    query: str = FieldDescriptor(
        'query', str,
        validators=[FieldValidator.required],
        description="SQL query for the report"
    )
    
    parameters: List[Dict[str, Any]] = FieldDescriptor(
        'parameters', list, default=list,
        description="Report parameters"
    )
    
    output_format: str = FieldDescriptor(
        'output_format', str, default='json',
        validators=[FieldValidator.max_length(20)],
        description="Default output format"
    )
    
    is_public: bool = FieldDescriptor(
        'is_public', bool, default=False,
        description="Whether report is publicly accessible"
    )
    
    cache_duration_seconds: int = FieldDescriptor(
        'cache_duration_seconds', int, default=300,
        validators=[FieldValidator.numeric_range(0, 86400)],
        description="Cache duration in seconds"
    )
    
    def get_parameter_definitions(self) -> List[ReportParameter]:
        """Get report parameter definitions."""
        params = []
        
        for param_dict in self.parameters:
            param_type = param_dict.get('type', 'str')
            type_map = {
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'date': date,
                'datetime': datetime
            }
            
            params.append(ReportParameter(
                name=param_dict['name'],
                parameter_type=type_map.get(param_type, str),
                required=param_dict.get('required', False),
                default_value=param_dict.get('default_value'),
                description=param_dict.get('description', ''),
                choices=param_dict.get('choices', [])
            ))
        
        return params


class ReportExecution(BaseModel):
    """Report execution instance."""
    
    report_id: str = FieldDescriptor(
        'report_id', str,
        validators=[FieldValidator.required],
        description="ID of the report definition"
    )
    
    parameters: Dict[str, Any] = FieldDescriptor(
        'parameters', dict, default=dict,
        description="Execution parameters"
    )
    
    status: str = FieldDescriptor(
        'status', str, default='pending',
        validators=[FieldValidator.max_length(20)],
        description="Execution status"
    )
    
    started_at: Optional[datetime] = FieldDescriptor(
        'started_at', datetime, nullable=True,
        description="When execution started"
    )
    
    completed_at: Optional[datetime] = FieldDescriptor(
        'completed_at', datetime, nullable=True,
        description="When execution completed"
    )
    
    result_data: Optional[List[Dict[str, Any]]] = FieldDescriptor(
        'result_data', list, nullable=True,
        description="Report result data"
    )
    
    error_message: Optional[str] = FieldDescriptor(
        'error_message', str, nullable=True,
        validators=[FieldValidator.max_length(1000)],
        description="Error message if execution failed"
    )
    
    row_count: int = FieldDescriptor(
        'row_count', int, default=0,
        validators=[FieldValidator.numeric_range(0, 1000000)],
        description="Number of result rows"
    )
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate execution duration."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.now(timezone.utc)
        return end_time - self.started_at
    
    @property
    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self.status in ('completed', 'failed')


class ReportingService(EnhancedBaseService):
    """Service for generating and managing reports."""
    
    def __init__(
        self,
        unit_of_work: UnitOfWork,
        connection_manager: ConnectionManager,
        cache_manager: CacheManager,
        audit_logger: Optional[AuditLogger] = None,
        event_dispatcher: Optional[EventDispatcher] = None,
        thread_pool_executor: Optional[ThreadPoolExecutor] = None
    ):
        super().__init__(unit_of_work, audit_logger, event_dispatcher, thread_pool_executor)
        self.connection_manager = connection_manager
        self.cache_manager = cache_manager
        self.definition_repository = unit_of_work.get_repository(ReportDefinition)
        self.execution_repository = unit_of_work.get_repository(ReportExecution)
    
    async def create_report_definition(
        self, 
        definition: ReportDefinition
    ) -> ServiceResult[ReportDefinition]:
        """Create a new report definition."""
        
        async def _create_definition():
            # Check name uniqueness
            existing = await self.definition_repository.find_by_criteria({'name': definition.name})
            if existing:
                raise BusinessRuleViolationError(f"Report '{definition.name}' already exists")
            
            # Validate SQL query (basic validation)
            await self._validate_query(definition.query)
            
            return await self.definition_repository.create(definition)
        
        return await self.execute_with_transaction(_create_definition, "create_report_definition")
    
    async def execute_report(
        self,
        report_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        output_format: str = 'json',
        use_cache: bool = True
    ) -> ServiceResult[ReportExecution]:
        """Execute a report."""
        
        async def _execute_report():
            # Get report definition
            definition = await self.definition_repository.get_by_id(report_id)
            if not definition:
                raise RecordNotFoundError(f"Report definition {report_id} not found")
            
            # Validate parameters
            validated_params = await self._validate_parameters(definition, parameters or {})
            
            # Check cache first
            cache_key = None
            if use_cache and definition.cache_duration_seconds > 0:
                cache_key = self._generate_cache_key(report_id, validated_params)
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return ReportExecution.from_dict(cached_result)
            
            # Create execution instance
            execution = ReportExecution(
                report_id=report_id,
                parameters=validated_params,
                status='running',
                started_at=datetime.now(timezone.utc)
            )
            
            # Save execution
            created_execution = await self.execution_repository.create(execution)
            
            try:
                # Execute query
                result_data = await self._execute_query(definition, validated_params)
                
                # Update execution with results
                created_execution.status = 'completed'
                created_execution.completed_at = datetime.now(timezone.utc)
                created_execution.result_data = result_data
                created_execution.row_count = len(result_data)
                
                # Cache result if enabled
                if cache_key and definition.cache_duration_seconds > 0:
                    await self.cache_manager.set(
                        cache_key,
                        created_execution.to_dict(),
                        definition.cache_duration_seconds
                    )
                
            except Exception as e:
                # Update execution with error
                created_execution.status = 'failed'
                created_execution.completed_at = datetime.now(timezone.utc)
                created_execution.error_message = str(e)
                logger.error(f"Report execution {created_execution.id} failed: {e}")
            
            # Save final execution state
            await self.execution_repository.update(created_execution)
            
            return created_execution
        
        return await self.execute_with_transaction(_execute_report, "execute_report")
    
    async def _validate_query(self, query: str) -> None:
        """Validate SQL query (basic validation)."""
        # Basic SQL injection prevention
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        upper_query = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in upper_query:
                raise BusinessRuleViolationError(f"Query contains forbidden keyword: {keyword}")
        
        # Must start with SELECT
        if not upper_query.strip().startswith('SELECT'):
            raise BusinessRuleViolationError("Query must start with SELECT")
    
    async def _validate_parameters(
        self, 
        definition: ReportDefinition, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and process report parameters."""
        
        param_definitions = definition.get_parameter_definitions()
        validated_params = {}
        
        for param_def in param_definitions:
            value = parameters.get(param_def.name, param_def.default_value)
            
            is_valid, error_msg = param_def.validate_value(value)
            if not is_valid:
                raise ValidationError(error_msg)
            
            validated_params[param_def.name] = value
        
        return validated_params
    
    async def _execute_query(
        self, 
        definition: ReportDefinition, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute the report query."""
        
        # Replace parameters in query
        query = definition.query
        for param_name, param_value in parameters.items():
            placeholder = f"{{{param_name}}}"
            # Simple string replacement - in production use proper parameter binding
            query = query.replace(placeholder, str(param_value))
        
        # Execute query
        results = self.connection_manager.execute_query(query)
        
        # Convert to list of dictionaries
        if results:
            if hasattr(results[0], 'keys'):  # Row-like objects
                return [dict(row) for row in results]
            else:  # Tuples
                # This is simplified - in reality we'd need column names
                return [{'column_' + str(i): value for i, value in enumerate(row)} for row in results]
        
        return []
    
    def _generate_cache_key(self, report_id: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for report results."""
        param_str = json.dumps(parameters, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"report:{report_id}:{param_hash}"


# ==================== MODULE EXPORTS CONTINUATION ====================

__all__.extend([
    # Enhanced Services
    'ServiceObserver', 'LoggingObserver', 'MetricsObserver', 'EnhancedBaseService',
    
    # Data Loading
    'DataFormat', 'DataSchema', 'DataCleaningRule', 'DataLoaderService',
    
    # CRUD Services
    'PaginationParams', 'SortParams', 'FilterParams', 'PaginatedResult', 'CRUDService',
    'UserCRUDService', 'OrganizationCRUDService',
    
    # Workflow Engine
    'WorkflowState', 'WorkflowStep', 'WorkflowDefinition', 'WorkflowExecution',
    'WorkflowStepHandler', 'ActionStepHandler', 'ConditionStepHandler', 'WorkflowEngine',
    
    # Reporting
    'ReportParameter', 'ReportDefinition', 'ReportExecution', 'ReportingService'
])


# ==================== APPLICATION FACTORY EXTENSION ====================

class EnhancedAutoERPApplication(AutoERPApplication):
    """Enhanced application with additional services."""
    
    def __init__(self, config: AutoERPConfig):
        super().__init__(config)
        
        # Enhanced services
        self.data_loader_service: Optional[DataLoaderService] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.reporting_service: Optional[ReportingService] = None
        
        # CRUD services
        self.user_crud_service: Optional[UserCRUDService] = None
        self.organization_crud_service: Optional[OrganizationCRUDService] = None
    
    async def _initialize_services(self) -> None:
        """Initialize all services including enhanced ones."""
        await super()._initialize_services()
        
        # Create unit of work
        unit_of_work = UnitOfWork(self.connection_manager)
        
        # Create audit logger
        audit_repository = BaseRepository(AuditLogEntry, self.connection_manager)
        audit_logger = AuditLogger(audit_repository)
        
        # Initialize enhanced services
        self.data_loader_service = DataLoaderService(
            unit_of_work=unit_of_work,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher,
            thread_pool_executor=self.thread_pool
        )
        
        self.workflow_engine = WorkflowEngine(
            unit_of_work=unit_of_work,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher,
            thread_pool_executor=self.thread_pool
        )
        
        self.reporting_service = ReportingService(
            unit_of_work=unit_of_work,
            connection_manager=self.connection_manager,
            cache_manager=self.cache_manager,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher,
            thread_pool_executor=self.thread_pool
        )
        
        # Initialize CRUD services
        self.user_crud_service = UserCRUDService(
            unit_of_work=unit_of_work,
            password_manager=self.password_manager,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher,
            thread_pool_executor=self.thread_pool
        )
        
        self.organization_crud_service = OrganizationCRUDService(
            unit_of_work=unit_of_work,
            audit_logger=audit_logger,
            event_dispatcher=self.event_dispatcher,
            thread_pool_executor=self.thread_pool
        )

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def example_usage():
        """Example of how to use the AutoERP core module."""
        
        # Create configuration
        config = AutoERPConfig()
        
        # Initialize application
        async with AutoERPApplication(config) as app:
            # Create a user
            result = await app.user_service.create_user(
                username="john_doe",
                email="john@example.com",
                password="SecurePass123!",
                first_name="John",
                last_name="Doe"
            )
            
            if result.is_success():
                user = result.get_data()
                print(f"Created user: {user}")
                
                # Authenticate user
                auth_result = await app.user_service.authenticate_user(
                    username_or_email="john_doe",
                    password="SecurePass123!"
                )
                
                if auth_result.is_success():
                    user, session = auth_result.get_data()
                    print(f"Authenticated user: {user.full_name}")
                    print(f"Session token: {session.session_token}")
                
            # Get health status
            health = await app.get_health_status()
            print(f"Application health: {health['status']}")
            
            # Get metrics
            metrics = app.get_metrics()
            print(f"Application uptime: {metrics['uptime_seconds']} seconds")
    
    # Run example
    # asyncio.run(example_usage())
    
    print("AutoERP Core module loaded successfully")