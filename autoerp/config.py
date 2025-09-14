"""
AutoERP Configuration Module
============================

This module provides comprehensive configuration management for the AutoERP system,
supporting multiple configuration sources (YAML, JSON, environment variables, CLI arguments)
and environment-specific settings with validation.

Author: AutoERP Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import yaml
import argparse
import logging
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Any, Optional, List, Union, Type, get_type_hints
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import re
from urllib.parse import urlparse
import socket
import warnings

# Third-party imports
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    warnings.warn("python-dotenv not installed. Environment file loading disabled.")
    load_dotenv = lambda *args, **kwargs: None
    find_dotenv = lambda *args, **kwargs: None

# ==================== ENUMS AND CONSTANTS ====================

class Environment(Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging" 
    PRODUCTION = "production"


class DatabaseEngine(Enum):
    """Supported database engines."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CacheBackend(Enum):
    """Supported cache backends."""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"


# Configuration file patterns
CONFIG_FILE_PATTERNS = [
    "config.yml", "config.yaml",
    "config.json",
    ".env", ".env.local",
    f"config.{Environment.DEVELOPMENT.value}.yml",
    f"config.{Environment.TESTING.value}.yml", 
    f"config.{Environment.STAGING.value}.yml",
    f"config.{Environment.PRODUCTION.value}.yml"
]

# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    Path.cwd(),
    Path.cwd() / "config",
    Path.cwd() / "configs",
    Path.home() / ".autoerp",
    Path("/etc/autoerp")
]

# ==================== VALIDATION UTILITIES ====================

class ValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error for field '{field}': {message}")


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_port(port: int, field_name: str = "port") -> None:
        """Validate port number."""
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValidationError(field_name, port, "Port must be between 1 and 65535")
    
    @staticmethod
    def validate_url(url: str, field_name: str = "url") -> None:
        """Validate URL format."""
        if not url:
            return
        
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValidationError(field_name, url, "Invalid URL format")
        except Exception:
            raise ValidationError(field_name, url, "Invalid URL format")
    
    @staticmethod
    def validate_email(email: str, field_name: str = "email") -> None:
        """Validate email format."""
        if not email:
            return
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise ValidationError(field_name, email, "Invalid email format")
    
    @staticmethod
    def validate_path(path: Union[str, Path], field_name: str = "path", must_exist: bool = False) -> None:
        """Validate file/directory path."""
        if not path:
            return
        
        path_obj = Path(path)
        
        if must_exist and not path_obj.exists():
            raise ValidationError(field_name, str(path), "Path does not exist")
        
        # Check if parent directory is writable
        parent = path_obj.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                raise ValidationError(field_name, str(path), "Cannot create parent directory")
    
    @staticmethod
    def validate_host(host: str, field_name: str = "host") -> None:
        """Validate hostname or IP address."""
        if not host:
            return
        
        # Try to resolve hostname
        try:
            socket.gethostbyname(host)
        except socket.gaierror:
            # Check if it's a valid IP address format
            try:
                socket.inet_aton(host)
            except socket.error:
                raise ValidationError(field_name, host, "Invalid hostname or IP address")
    
    @staticmethod
    def validate_positive_int(value: int, field_name: str) -> None:
        """Validate positive integer."""
        if not isinstance(value, int) or value <= 0:
            raise ValidationError(field_name, value, "Must be a positive integer")
    
    @staticmethod
    def validate_non_negative_int(value: int, field_name: str) -> None:
        """Validate non-negative integer."""
        if not isinstance(value, int) or value < 0:
            raise ValidationError(field_name, value, "Must be a non-negative integer")
    
    @staticmethod
    def validate_string_length(value: str, field_name: str, min_length: int = 0, max_length: int = None) -> None:
        """Validate string length."""
        if not isinstance(value, str):
            raise ValidationError(field_name, value, "Must be a string")
        
        if len(value) < min_length:
            raise ValidationError(field_name, value, f"Must be at least {min_length} characters")
        
        if max_length and len(value) > max_length:
            raise ValidationError(field_name, value, f"Must be no more than {max_length} characters")


# ==================== DATABASE SETTINGS ====================

@dataclass
class DatabaseSettings:
    """Database connection and configuration settings."""
    
    # Connection settings
    engine: DatabaseEngine = DatabaseEngine.SQLITE
    host: str = "localhost"
    port: int = 5432
    database: str = "autoerp"
    username: str = ""
    password: str = ""
    
    # SQLite specific
    sqlite_file: str = "autoerp.db"
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    
    # Connection options
    connect_timeout: int = 10
    read_timeout: int = 30
    write_timeout: int = 30
    
    # SSL settings
    ssl_mode: str = "prefer"
    ssl_cert: str = ""
    ssl_key: str = ""
    ssl_ca: str = ""
    
    # Charset and collation
    charset: str = "utf8mb4"
    collation: str = ""
    
    # Migration settings
    auto_migrate: bool = True
    migration_timeout: int = 300
    
    # Backup settings
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7
    backup_path: str = "backups"
    backup_compress: bool = True
    
    # Performance settings
    query_timeout: int = 60
    slow_query_threshold: float = 1.0
    log_slow_queries: bool = True
    
    # Development settings
    echo_queries: bool = False
    
    def validate(self) -> None:
        """Validate database settings."""
        # Validate engine-specific settings
        if self.engine == DatabaseEngine.SQLITE:
            ConfigValidator.validate_path(self.sqlite_file, "sqlite_file")
        else:
            ConfigValidator.validate_host(self.host, "host")
            ConfigValidator.validate_port(self.port, "port")
            ConfigValidator.validate_string_length(self.database, "database", min_length=1)
            ConfigValidator.validate_string_length(self.username, "username", min_length=1)
        
        # Validate pool settings
        ConfigValidator.validate_positive_int(self.pool_size, "pool_size")
        ConfigValidator.validate_non_negative_int(self.max_overflow, "max_overflow")
        ConfigValidator.validate_positive_int(self.pool_timeout, "pool_timeout")
        ConfigValidator.validate_positive_int(self.pool_recycle, "pool_recycle")
        
        # Validate timeout settings
        ConfigValidator.validate_positive_int(self.connect_timeout, "connect_timeout")
        ConfigValidator.validate_positive_int(self.read_timeout, "read_timeout")
        ConfigValidator.validate_positive_int(self.write_timeout, "write_timeout")
        ConfigValidator.validate_positive_int(self.query_timeout, "query_timeout")
        
        # Validate SSL certificates if provided
        if self.ssl_cert:
            ConfigValidator.validate_path(self.ssl_cert, "ssl_cert", must_exist=True)
        if self.ssl_key:
            ConfigValidator.validate_path(self.ssl_key, "ssl_key", must_exist=True)
        if self.ssl_ca:
            ConfigValidator.validate_path(self.ssl_ca, "ssl_ca", must_exist=True)
        
        # Validate backup settings
        if self.backup_enabled:
            ConfigValidator.validate_path(self.backup_path, "backup_path")
            ConfigValidator.validate_positive_int(self.backup_interval_hours, "backup_interval_hours")
            ConfigValidator.validate_positive_int(self.backup_retention_days, "backup_retention_days")
    
    def get_connection_url(self) -> str:
        """Generate database connection URL."""
        if self.engine == DatabaseEngine.SQLITE:
            return f"sqlite:///{self.sqlite_file}"
        elif self.engine == DatabaseEngine.POSTGRESQL:
            password_part = f":{self.password}" if self.password else ""
            return f"postgresql://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"
        elif self.engine == DatabaseEngine.MYSQL:
            password_part = f":{self.password}" if self.password else ""
            return f"mysql://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database engine: {self.engine}")
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters as dictionary."""
        params = {
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
        }
        
        if self.engine != DatabaseEngine.SQLITE:
            params.update({
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "user": self.username,
                "password": self.password,
            })
            
            if self.charset:
                params["charset"] = self.charset
            
            if self.ssl_mode != "disable":
                params["sslmode"] = self.ssl_mode
                if self.ssl_cert:
                    params["sslcert"] = self.ssl_cert
                if self.ssl_key:
                    params["sslkey"] = self.ssl_key
                if self.ssl_ca:
                    params["sslca"] = self.ssl_ca
        
        return params


# ==================== CACHE SETTINGS ====================

@dataclass
class CacheSettings:
    """Cache configuration settings."""
    
    # General settings
    enabled: bool = True
    backend: CacheBackend = CacheBackend.MEMORY
    default_timeout: int = 300  # 5 minutes
    key_prefix: str = "autoerp"
    
    # Memory cache settings
    max_entries: int = 10000
    cleanup_interval: int = 600  # 10 minutes
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ssl: bool = False
    redis_socket_timeout: float = 5.0
    redis_connection_pool_size: int = 10
    
    # Memcached settings
    memcached_servers: List[str] = field(default_factory=lambda: ["localhost:11211"])
    memcached_binary: bool = True
    
    # Serialization
    serializer: str = "pickle"  # pickle, json, msgpack
    compression: bool = False
    compression_level: int = 6
    
    def validate(self) -> None:
        """Validate cache settings."""
        ConfigValidator.validate_positive_int(self.default_timeout, "default_timeout")
        ConfigValidator.validate_string_length(self.key_prefix, "key_prefix", min_length=1, max_length=50)
        
        if self.backend == CacheBackend.REDIS:
            ConfigValidator.validate_host(self.redis_host, "redis_host")
            ConfigValidator.validate_port(self.redis_port, "redis_port")
            ConfigValidator.validate_non_negative_int(self.redis_db, "redis_db")
            ConfigValidator.validate_positive_int(self.redis_connection_pool_size, "redis_connection_pool_size")
        
        elif self.backend == CacheBackend.MEMCACHED:
            for i, server in enumerate(self.memcached_servers):
                if ":" not in server:
                    raise ValidationError(f"memcached_servers[{i}]", server, "Must be in format 'host:port'")
        
        if self.serializer not in ("pickle", "json", "msgpack"):
            raise ValidationError("serializer", self.serializer, "Must be 'pickle', 'json', or 'msgpack'")


# ==================== SECURITY SETTINGS ====================

@dataclass
class SecuritySettings:
    """Security configuration settings."""
    
    # JWT settings
    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 1440  # 24 hours
    jwt_refresh_expiration_days: int = 7
    
    # Password policy
    password_min_length: int = 8
    password_max_length: int = 128
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_special: bool = True
    password_special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    password_history_count: int = 5
    password_expiry_days: int = 90
    
    # Session settings
    session_timeout_minutes: int = 60
    session_absolute_timeout_hours: int = 24
    max_concurrent_sessions: int = 3
    
    # Login security
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    lockout_escalation: bool = True
    
    # Two-factor authentication
    enable_2fa: bool = False
    totp_issuer: str = "AutoERP"
    backup_codes_count: int = 10
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 200
    
    # IP security
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # CSRF protection
    csrf_enabled: bool = True
    csrf_token_timeout: int = 3600
    
    # Content Security Policy
    csp_enabled: bool = True
    csp_policy: str = "default-src 'self'"
    
    # HTTPS enforcement
    force_https: bool = False
    hsts_max_age: int = 31536000  # 1 year
    
    def validate(self) -> None:
        """Validate security settings."""
        if not self.secret_key:
            raise ValidationError("secret_key", "", "Secret key is required")
        
        ConfigValidator.validate_string_length(self.secret_key, "secret_key", min_length=32)
        
        # Password policy validation
        ConfigValidator.validate_positive_int(self.password_min_length, "password_min_length")
        ConfigValidator.validate_positive_int(self.password_max_length, "password_max_length")
        
        if self.password_min_length > self.password_max_length:
            raise ValidationError("password_min_length", self.password_min_length, 
                                "Cannot be greater than password_max_length")
        
        # Session validation
        ConfigValidator.validate_positive_int(self.session_timeout_minutes, "session_timeout_minutes")
        ConfigValidator.validate_positive_int(self.session_absolute_timeout_hours, "session_absolute_timeout_hours")
        ConfigValidator.validate_positive_int(self.max_concurrent_sessions, "max_concurrent_sessions")
        
        # Login security validation
        ConfigValidator.validate_positive_int(self.max_login_attempts, "max_login_attempts")
        ConfigValidator.validate_positive_int(self.lockout_duration_minutes, "lockout_duration_minutes")
        
        # Rate limiting validation
        if self.rate_limit_enabled:
            ConfigValidator.validate_positive_int(self.rate_limit_requests_per_minute, "rate_limit_requests_per_minute")
            ConfigValidator.validate_positive_int(self.rate_limit_burst, "rate_limit_burst")
        
        # IP validation
        for ip in self.ip_whitelist + self.ip_blacklist:
            try:
                socket.inet_aton(ip.split('/')[0])  # Basic IP validation
            except socket.error:
                raise ValidationError("ip_address", ip, "Invalid IP address format")


# ==================== LOGGING SETTINGS ====================

@dataclass
class LoggingSettings:
    """Logging configuration settings."""
    
    # General settings
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    file_enabled: bool = True
    file_path: str = "logs/autoerp.log"
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    file_rotation: str = "size"  # size, time
    
    # Console logging
    console_enabled: bool = True
    console_format: str = "%(levelname)s - %(name)s - %(message)s"
    
    # Syslog
    syslog_enabled: bool = False
    syslog_address: str = "/dev/log"
    syslog_facility: str = "local0"
    
    # Remote logging
    remote_enabled: bool = False
    remote_host: str = ""
    remote_port: int = 514
    remote_protocol: str = "tcp"  # tcp, udp
    
    # Structured logging
    structured_logging: bool = False
    log_format: str = "text"  # text, json
    
    # Performance logging
    log_sql_queries: bool = False
    log_slow_queries_only: bool = True
    slow_query_threshold: float = 1.0
    
    # Security logging
    log_authentication: bool = True
    log_authorization: bool = True
    log_data_access: bool = False
    
    # Logger-specific levels
    logger_levels: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate logging settings."""
        if self.file_enabled:
            ConfigValidator.validate_path(self.file_path, "file_path")
            ConfigValidator.validate_positive_int(self.file_max_size, "file_max_size")
            ConfigValidator.validate_positive_int(self.file_backup_count, "file_backup_count")
        
        if self.remote_enabled:
            ConfigValidator.validate_host(self.remote_host, "remote_host")
            ConfigValidator.validate_port(self.remote_port, "remote_port")
            
            if self.remote_protocol not in ("tcp", "udp"):
                raise ValidationError("remote_protocol", self.remote_protocol, "Must be 'tcp' or 'udp'")
        
        if self.log_format not in ("text", "json"):
            raise ValidationError("log_format", self.log_format, "Must be 'text' or 'json'")
        
        # Validate logger levels
        valid_levels = {level.value for level in LogLevel}
        for logger_name, level in self.logger_levels.items():
            if level not in valid_levels:
                raise ValidationError(f"logger_levels.{logger_name}", level, f"Must be one of: {valid_levels}")


# ==================== EMAIL SETTINGS ====================

@dataclass
class EmailSettings:
    """Email configuration settings."""
    
    # SMTP settings
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout: int = 30
    
    # Default sender
    default_from_email: str = "noreply@autoerp.com"
    default_from_name: str = "AutoERP System"
    
    # Email templates
    template_dir: str = "templates/email"
    default_template: str = "default.html"
    
    # Queue settings
    queue_enabled: bool = True
    queue_backend: str = "database"  # database, redis, memory
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60
    
    # Rate limiting
    rate_limit_enabled: bool = True
    max_emails_per_minute: int = 100
    max_emails_per_hour: int = 1000
    
    # Bounce handling
    bounce_handling_enabled: bool = False
    bounce_email: str = ""
    
    # Monitoring
    track_opens: bool = False
    track_clicks: bool = False
    
    def validate(self) -> None:
        """Validate email settings."""
        ConfigValidator.validate_host(self.smtp_host, "smtp_host")
        ConfigValidator.validate_port(self.smtp_port, "smtp_port")
        ConfigValidator.validate_email(self.default_from_email, "default_from_email")
        ConfigValidator.validate_positive_int(self.smtp_timeout, "smtp_timeout")
        
        if self.smtp_use_tls and self.smtp_use_ssl:
            raise ValidationError("smtp_use_ssl", True, "Cannot use both TLS and SSL")
        
        if self.queue_enabled:
            ConfigValidator.validate_positive_int(self.max_retry_attempts, "max_retry_attempts")
            ConfigValidator.validate_positive_int(self.retry_delay_seconds, "retry_delay_seconds")
        
        if self.rate_limit_enabled:
            ConfigValidator.validate_positive_int(self.max_emails_per_minute, "max_emails_per_minute")
            ConfigValidator.validate_positive_int(self.max_emails_per_hour, "max_emails_per_hour")
        
        if self.bounce_handling_enabled:
            ConfigValidator.validate_email(self.bounce_email, "bounce_email")
        
        ConfigValidator.validate_path(self.template_dir, "template_dir")


# ==================== MAIN APP SETTINGS ====================

@dataclass
class AppSettings:
    """Main application settings container."""
    
    # Environment and basic info
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    testing: bool = False
    app_name: str = "AutoERP"
    app_version: str = "1.0.0"
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    
    # Application URLs
    base_url: str = "http://localhost:8000"
    api_prefix: str = "/api/v1"
    
    # Timezone and localization
    timezone: str = "UTC"
    locale: str = "en_US"
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File handling
    upload_dir: str = "uploads"
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"])
    
    # Session settings
    session_cookie_name: str = "autoerp_session"
    session_cookie_secure: bool = False
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "Lax"
    
    # Component settings
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    email: EmailSettings = field(default_factory=EmailSettings)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "user_registration": True,
        "email_verification": True,
        "two_factor_auth": False,
        "audit_logging": True,
        "data_export": True,
        "api_rate_limiting": True,
        "maintenance_mode": False
    })
    
    def validate_configuration(self) -> List[ValidationError]:
        """Validate all configuration settings and return list of errors."""
        errors = []
        
        try:
            # Validate basic settings
            ConfigValidator.validate_host(self.host, "host")
            ConfigValidator.validate_port(self.port, "port")
            ConfigValidator.validate_positive_int(self.workers, "workers")
            ConfigValidator.validate_url(self.base_url, "base_url")
            ConfigValidator.validate_string_length(self.app_name, "app_name", min_length=1, max_length=100)
            ConfigValidator.validate_path(self.upload_dir, "upload_dir")
            ConfigValidator.validate_positive_int(self.max_upload_size, "max_upload_size")
            
            # Validate cookie settings
            if self.session_cookie_samesite not in ("Strict", "Lax", "None"):
                errors.append(ValidationError("session_cookie_samesite", self.session_cookie_samesite,
                                            "Must be 'Strict', 'Lax', or 'None'"))
            
        except ValidationError as e:
            errors.append(e)
        
        # Validate component settings
        try:
            self.database.validate()
        except ValidationError as e:
            errors.append(ValidationError(f"database.{e.field}", e.value, e.message))
        
        try:
            self.cache.validate()
        except ValidationError as e:
            errors.append(ValidationError(f"cache.{e.field}", e.value, e.message))
        
        try:
            self.security.validate()
        except ValidationError as e:
            errors.append(ValidationError(f"security.{e.field}", e.value, e.message))
        
        try:
            self.logging.validate()
        except ValidationError as e:
            errors.append(ValidationError(f"logging.{e.field}", e.value, e.message))
        
        try:
            self.email.validate()
        except ValidationError as e:
            errors.append(ValidationError(f"email.{e.field}", e.value, e.message))
        
        # Environment-specific validation
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                errors.append(ValidationError("debug", True, "Debug mode should be disabled in production"))
            
            if self.security.secret_key == "":
                errors.append(ValidationError("security.secret_key", "", "Secret key is required in production"))
            
            if not self.session_cookie_secure and self.base_url.startswith("https://"):
                errors.append(ValidationError("session_cookie_secure", False, 
                                            "Secure cookies should be enabled for HTTPS"))
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate_configuration()) == 0
    
    def get_validation_summary(self) -> str:
        """Get a summary of validation results."""
        errors = self.validate_configuration()
        
        if not errors:
            return "✅ Configuration is valid"
        
        summary = f"❌ Configuration has {len(errors)} error(s):\n"
        for i, error in enumerate(errors, 1):
            summary += f"  {i}. {error}\n"
        
        return summary


# ==================== CONFIGURATION LOADER ====================

class ConfigurationLoader:
    """Loads configuration from multiple sources with priority order."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_configuration(
        self,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
        cli_args: Optional[List[str]] = None
    ) -> AppSettings:
        """
        Load configuration from multiple sources in priority order:
        1. CLI arguments (highest priority)
        2. Environment variables
        3. Configuration file (YAML/JSON)
        4. .env file
        5. Default values (lowest priority)
        """
        
        # Start with default configuration
        config_data = self._get_default_config()
        
        # Load from .env file
        if env_file or find_dotenv():
            env_path = env_file or find_dotenv()
            self._load_env_file(env_path)
            self.logger.info(f"Loaded environment file: {env_path}")
        
        # Load from configuration file
        config_file_data = self._load_config_file(config_file)
        if config_file_data:
            config_data = self._deep_merge(config_data, config_file_data)
            self.logger.info(f"Loaded configuration file")
        
        # Load from environment variables
        env_data = self._load_from_env_vars()
        if env_data:
            config_data = self._deep_merge(config_data, env_data)
            self.logger.info("Loaded environment variables")
        
        # Load from CLI arguments
        if cli_args:
            cli_data = self._load_from_cli_args(cli_args)
            if cli_data:
                config_data = self._deep_merge(config_data, cli_data)
                self.logger.info("Loaded CLI arguments")
        
        # Convert to AppSettings object
        return self._create_app_settings(config_data)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on environment."""
        env = Environment(os.getenv("AUTOERP_ENV", "development"))
        
        if env == Environment.DEVELOPMENT:
            return self._get_development_config()
        elif env == Environment.TESTING:
            return self._get_testing_config()
        elif env == Environment.STAGING:
            return self._get_staging_config()
        elif env == Environment.PRODUCTION:
            return self._get_production_config()
        else:
            return self._get_development_config()
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Get development environment defaults."""
        return {
            "environment": "development",
            "debug": True,
            "testing": False,
            "host": "127.0.0.1",
            "port": 8000,
            "base_url": "http://localhost:8000",
            "database": {
                "engine": "sqlite",
                "sqlite_file": "autoerp_dev.db",
                "echo_queries": True
            },
            "cache": {
                "backend": "memory",
                "enabled": True
            },
            "security": {
                "secret_key": "dev-secret-key-change-in-production",
                "password_min_length": 6,
                "session_timeout_minutes": 120
            },
            "logging": {
                "level": "DEBUG",
                "console_enabled": True,
                "file_enabled": True,
                "file_path": "logs/autoerp_dev.log"
            },
            "email": {
                "smtp_host": "localhost",
                "smtp_port": 1025,  # MailHog default
                "queue_enabled": False
            }
        }
    
    def _get_testing_config(self) -> Dict[str, Any]:
        """Get testing environment defaults."""
        return {
            "environment": "testing",
            "debug": False,
            "testing": True,
            "host": "127.0.0.1",
            "port": 8001,
            "base_url": "http://localhost:8001",
            "database": {
                "engine": "sqlite",
                "sqlite_file": ":memory:",
                "echo_queries": False
            },
            "cache": {
                "backend": "memory",
                "enabled": False
            },
            "security": {
                "secret_key": "test-secret-key",
                "password_min_length": 4,
                "max_login_attempts": 10
            },
            "logging": {
                "level": "WARNING",
                "console_enabled": False,
                "file_enabled": False
            },
            "email": {
                "queue_enabled": False
            }
        }
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Get staging environment defaults."""
        return {
            "environment": "staging",
            "debug": False,
            "testing": False,
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 2,
            "database": {
                "engine": "postgresql",
                "host": "staging-db.internal",
                "database": "autoerp_staging"
            },
            "cache": {
                "backend": "redis",
                "redis_host": "staging-redis.internal"
            },
            "security": {
                "session_cookie_secure": True,
                "csrf_enabled": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "/var/log/autoerp/staging.log"
            }
        }
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Get production environment defaults."""
        return {
            "environment": "production",
            "debug": False,
            "testing": False,
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "database": {
                "engine": "postgresql",
                "pool_size": 20,
                "echo_queries": False
            },
            "cache": {
                "backend": "redis",
                "enabled": True
            },
            "security": {
                "session_cookie_secure": True,
                "csrf_enabled": True,
                "force_https": True,
                "rate_limit_enabled": True
            },
            "logging": {
                "level": "WARNING",
                "file_path": "/var/log/autoerp/production.log",
                "remote_enabled": True
            }
        }
    
    def _load_env_file(self, env_file: str) -> None:
        """Load environment variables from .env file."""
        if env_file and Path(env_file).exists():
            load_dotenv(env_file, override=True)
    
    def _find_config_file(self, config_file: Optional[str] = None) -> Optional[Path]:
        """Find configuration file in standard locations."""
        if config_file:
            path = Path(config_file)
            if path.exists():
                return path
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Search for config files in standard locations
        for search_path in DEFAULT_CONFIG_PATHS:
            for pattern in CONFIG_FILE_PATTERNS:
                config_path = search_path / pattern
                if config_path.exists():
                    return config_path
        
        return None
    
    def _load_config_file(self, config_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML or JSON file."""
        config_path = self._find_config_file(config_file)
        
        if not config_path:
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ('.yml', '.yaml'):
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {config_path}")
                    return None
        
        except Exception as e:
            self.logger.error(f"Error loading config file {config_path}: {e}")
            return None
    
    def _load_from_env_vars(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Map environment variables to config structure
        env_mappings = {
            # App settings
            "AUTOERP_ENV": ("environment",),
            "AUTOERP_DEBUG": ("debug", bool),
            "AUTOERP_HOST": ("host",),
            "AUTOERP_PORT": ("port", int),
            "AUTOERP_WORKERS": ("workers", int),
            "AUTOERP_BASE_URL": ("base_url",),
            
            # Database settings
            "DATABASE_ENGINE": ("database", "engine"),
            "DATABASE_HOST": ("database", "host"),
            "DATABASE_PORT": ("database", "port", int),
            "DATABASE_NAME": ("database", "database"),
            "DATABASE_USER": ("database", "username"),
            "DATABASE_PASSWORD": ("database", "password"),
            "DATABASE_SQLITE_FILE": ("database", "sqlite_file"),
            
            # Cache settings
            "CACHE_BACKEND": ("cache", "backend"),
            "CACHE_ENABLED": ("cache", "enabled", bool),
            "REDIS_HOST": ("cache", "redis_host"),
            "REDIS_PORT": ("cache", "redis_port", int),
            "REDIS_DB": ("cache", "redis_db", int),
            "REDIS_PASSWORD": ("cache", "redis_password"),
            
            # Security settings
            "SECRET_KEY": ("security", "secret_key"),
            "JWT_EXPIRATION_MINUTES": ("security", "jwt_expiration_minutes", int),
            "SESSION_TIMEOUT_MINUTES": ("security", "session_timeout_minutes", int),
            
            # Logging settings
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FILE": ("logging", "file_path"),
            
            # Email settings
            "SMTP_HOST": ("email", "smtp_host"),
            "SMTP_PORT": ("email", "smtp_port", int),
            "SMTP_USERNAME": ("email", "smtp_username"),
            "SMTP_PASSWORD": ("email", "smtp_password"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value based on type
                if len(config_path) > 2 and config_path[2] == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif len(config_path) > 2 and config_path[2] == int:
                    try:
                        value = int(value)
                    except ValueError:
                        self.logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                
                # Set nested configuration value
                self._set_nested_value(config, config_path[:-1], config_path[-1], value)
        
        return config
    
    def _load_from_cli_args(self, args: List[str]) -> Dict[str, Any]:
        """Load configuration from CLI arguments."""
        parser = argparse.ArgumentParser(description="AutoERP Configuration")
        
        # Add CLI arguments
        parser.add_argument("--env", choices=[e.value for e in Environment], help="Environment")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--host", help="Host address")
        parser.add_argument("--port", type=int, help="Port number")
        parser.add_argument("--workers", type=int, help="Number of workers")
        parser.add_argument("--config", help="Configuration file path")
        parser.add_argument("--db-engine", choices=[e.value for e in DatabaseEngine], help="Database engine")
        parser.add_argument("--db-host", help="Database host")
        parser.add_argument("--db-port", type=int, help="Database port")
        parser.add_argument("--db-name", help="Database name")
        parser.add_argument("--log-level", choices=[e.value for e in LogLevel], help="Log level")
        
        try:
            parsed_args = parser.parse_args(args)
            config = {}
            
            # Map CLI args to config structure
            if parsed_args.env:
                config["environment"] = parsed_args.env
            if parsed_args.debug:
                config["debug"] = parsed_args.debug
            if parsed_args.host:
                config["host"] = parsed_args.host
            if parsed_args.port:
                config["port"] = parsed_args.port
            if parsed_args.workers:
                config["workers"] = parsed_args.workers
            
            # Database settings
            db_config = {}
            if parsed_args.db_engine:
                db_config["engine"] = parsed_args.db_engine
            if parsed_args.db_host:
                db_config["host"] = parsed_args.db_host
            if parsed_args.db_port:
                db_config["port"] = parsed_args.db_port
            if parsed_args.db_name:
                db_config["database"] = parsed_args.db_name
            
            if db_config:
                config["database"] = db_config
            
            # Logging settings
            if parsed_args.log_level:
                config["logging"] = {"level": parsed_args.log_level}
            
            return config
        
        except SystemExit:
            # argparse calls sys.exit() on error or help
            return {}
    
    def _set_nested_value(self, config: Dict[str, Any], path: Tuple[str, ...], key: str, value: Any) -> None:
        """Set a nested configuration value."""
        current = config
        
        for part in path:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[key] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_app_settings(self, config_data: Dict[str, Any]) -> AppSettings:
        """Create AppSettings object from configuration dictionary."""
        try:
            # Convert string enums to enum instances
            if "environment" in config_data:
                config_data["environment"] = Environment(config_data["environment"])
            
            if "database" in config_data and "engine" in config_data["database"]:
                config_data["database"]["engine"] = DatabaseEngine(config_data["database"]["engine"])
            
            if "cache" in config_data and "backend" in config_data["cache"]:
                config_data["cache"]["backend"] = CacheBackend(config_data["cache"]["backend"])
            
            if "logging" in config_data and "level" in config_data["logging"]:
                config_data["logging"]["level"] = LogLevel(config_data["logging"]["level"])
            
            # Create nested objects
            database_data = config_data.pop("database", {})
            cache_data = config_data.pop("cache", {})
            security_data = config_data.pop("security", {})
            logging_data = config_data.pop("logging", {})
            email_data = config_data.pop("email", {})
            
            return AppSettings(
                database=DatabaseSettings(**database_data),
                cache=CacheSettings(**cache_data),
                security=SecuritySettings(**security_data),
                logging=LoggingSettings(**logging_data),
                email=EmailSettings(**email_data),
                **config_data
            )
        
        except Exception as e:
            self.logger.error(f"Error creating AppSettings: {e}")
            raise ValueError(f"Invalid configuration data: {e}")


# ==================== CONFIGURATION MANAGER ====================

class ConfigurationManager:
    """Main configuration manager for the AutoERP application."""
    
    _instance: Optional['ConfigurationManager'] = None
    _config: Optional[AppSettings] = None
    
    def __new__(cls) -> 'ConfigurationManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loader = ConfigurationLoader()
    
    def load_config(
        self,
        config_file: Optional[str] = None,
        env_file: Optional[str] = None,
        cli_args: Optional[List[str]] = None,
        validate: bool = True
    ) -> AppSettings:
        """Load and validate configuration."""
        try:
            self._config = self.loader.load_configuration(config_file, env_file, cli_args)
            
            if validate:
                errors = self._config.validate_configuration()
                if errors:
                    error_messages = [str(error) for error in errors]
                    raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))
            
            self.logger.info(f"Configuration loaded successfully for environment: {self._config.environment.value}")
            return self._config
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> AppSettings:
        """Get the current configuration."""
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def save_config(self, file_path: str, format: str = "yaml") -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise RuntimeError("No configuration to save")
        
        config_dict = asdict(self._config)
        
        # Convert enums to strings for serialization
        config_dict["environment"] = self._config.environment.value
        config_dict["database"]["engine"] = self._config.database.engine.value
        config_dict["cache"]["backend"] = self._config.cache.backend.value
        config_dict["security"] = asdict(self._config.security)
        config_dict["logging"]["level"] = self._config.logging.level.value
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() in ('yaml', 'yml'):
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration saved to: {file_path}")
    
    def reload_config(self) -> AppSettings:
        """Reload configuration from sources."""
        return self.load_config()
    
    def get_config_summary(self) -> str:
        """Get a summary of current configuration."""
        if self._config is None:
            return "No configuration loaded"
        
        summary = f"""
AutoERP Configuration Summary
============================
Environment: {self._config.environment.value}
Debug Mode: {self._config.debug}
Host: {self._config.host}:{self._config.port}
Base URL: {self._config.base_url}

Database: {self._config.database.engine.value}
Cache: {self._config.cache.backend.value} ({'enabled' if self._config.cache.enabled else 'disabled'})
Log Level: {self._config.logging.level.value}

Validation Status: {self._config.get_validation_summary()}
"""
        return summary.strip()


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    """
    Example usage of the configuration system.
    """
    
    # Setup logging for examples
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("AutoERP Configuration System Examples")
    print("=" * 40)
    
    # Example 1: Load default configuration
    print("\n1. Loading default development configuration:")
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_config(validate=False)
        print(f"   Environment: {config.environment.value}")
        print(f"   Database: {config.database.engine.value}")
        print(f"   Debug: {config.debug}")
        print(f"   Host: {config.host}:{config.port}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 2: Load configuration with validation
    print("\n2. Loading configuration with validation:")
    try:
        config = config_manager.load_config(validate=True)
        print("   ✅ Configuration is valid")
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
    
    # Example 3: Environment-specific configuration
    print("\n3. Testing different environments:")
    for env in Environment:
        try:
            os.environ["AUTOERP_ENV"] = env.value
            loader = ConfigurationLoader()
            config = loader.load_configuration()
            print(f"   {env.value}: Debug={config.debug}, Port={config.port}")
        except Exception as e:
            print(f"   {env.value}: Error - {e}")
    
    # Clean up environment
    os.environ.pop("AUTOERP_ENV", None)
    
    # Example 4: CLI arguments simulation
    print("\n4. Testing CLI argument parsing:")
    try:
        loader = ConfigurationLoader()
        config = loader.load_configuration(
            cli_args=["--env", "production", "--host", "0.0.0.0", "--port", "8080", "--debug"]
        )
        print(f"   Environment: {config.environment.value}")
        print(f"   Host: {config.host}:{config.port}")
        print(f"   Debug: {config.debug}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 5: Configuration validation
    print("\n5. Testing configuration validation:")
    try:
        # Create invalid configuration
        config = AppSettings(
            host="invalid-host",
            port=999999,  # Invalid port
            database=DatabaseSettings(
                engine=DatabaseEngine.POSTGRESQL,
                username=""  # Missing username
            )
        )
        
        errors = config.validate_configuration()
        print(f"   Found {len(errors)} validation errors:")
        for error in errors[:3]:  # Show first 3 errors
            print(f"     - {error}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 6: Save configuration
    print("\n6. Saving configuration to file:")
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_config(validate=False)
        
        # Save as YAML
        config_manager.save_config("example_config.yml", "yaml")
        print("   ✅ Configuration saved as YAML")
        
        # Save as JSON
        config_manager.save_config("example_config.json", "json")
        print("   ✅ Configuration saved as JSON")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 7: Configuration summary
    print("\n7. Configuration summary:")
    try:
        summary = config_manager.get_config_summary()
        print(summary)
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 8: Environment variables
    print("\n8. Testing environment variable override:")
    try:
        # Set some environment variables
        os.environ["AUTOERP_DEBUG"] = "true"
        os.environ["DATABASE_HOST"] = "production-db.example.com"
        os.environ["REDIS_HOST"] = "redis.example.com"
        
        loader = ConfigurationLoader()
        config = loader.load_configuration()
        
        print(f"   Debug (from env): {config.debug}")
        print(f"   DB Host (from env): {config.database.host}")
        print(f"   Redis Host (from env): {config.cache.redis_host}")
        
        # Clean up
        for key in ["AUTOERP_DEBUG", "DATABASE_HOST", "REDIS_HOST"]:
            os.environ.pop(key, None)
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 9: Create sample configuration files
    print("\n9. Creating sample configuration files:")
    
    # Sample YAML config
    sample_yaml = """
# AutoERP Configuration Example
environment: development
debug: true
app_name: "AutoERP Development"
host: 127.0.0.1
port: 8000

database:
  engine: sqlite
  sqlite_file: "autoerp_dev.db"
  pool_size: 5

cache:
  backend: memory
  enabled: true
  default_timeout: 300

security:
  secret_key: "your-secret-key-here"
  jwt_expiration_minutes: 1440
  password_min_length: 8

logging:
  level: DEBUG
  console_enabled: true
  file_enabled: true
  file_path: "logs/autoerp.log"

email:
  smtp_host: localhost
  smtp_port: 1025
  default_from_email: "noreply@autoerp.dev"
"""
    
    try:
        with open("sample_config.yml", "w") as f:
            f.write(sample_yaml)
        print("   ✅ Created sample_config.yml")
    except Exception as e:
        print(f"   Error creating YAML: {e}")
    
    # Sample .env file
    sample_env = """
# AutoERP Environment Variables
AUTOERP_ENV=development
AUTOERP_DEBUG=true
AUTOERP_HOST=127.0.0.1
AUTOERP_PORT=8000

# Database
DATABASE_ENGINE=sqlite
DATABASE_SQLITE_FILE=autoerp.db

# Cache
CACHE_BACKEND=memory
CACHE_ENABLED=true

# Security
SECRET_KEY=change-me-in-production

# Email
SMTP_HOST=localhost
SMTP_PORT=1025
"""
    
    try:
        with open("sample.env", "w") as f:
            f.write(sample_env)
        print("   ✅ Created sample.env")
    except Exception as e:
        print(f"   Error creating .env: {e}")
    
    print("\n" + "=" * 40)
    print("Configuration system examples completed!")
    print("Check the created files: example_config.yml, example_config.json, sample_config.yml, sample.env")


# ==================== CONFIGURATION TEMPLATES ====================

class ConfigurationTemplates:
    """Provides configuration templates for different deployment scenarios."""
    
    @staticmethod
    def get_docker_compose_template() -> str:
        """Get Docker Compose configuration template."""
        return """
# Docker Compose Configuration for AutoERP
version: '3.8'

services:
  autoerp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AUTOERP_ENV=production
      - DATABASE_ENGINE=postgresql
      - DATABASE_HOST=db
      - DATABASE_NAME=autoerp
      - DATABASE_USER=autoerp
      - DATABASE_PASSWORD=autoerp_password
      - REDIS_HOST=redis
      - SECRET_KEY=${SECRET_KEY:-change-me-in-production}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=autoerp
      - POSTGRES_USER=autoerp
      - POSTGRES_PASSWORD=autoerp_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""
    
    @staticmethod
    def get_kubernetes_template() -> str:
        """Get Kubernetes deployment template."""
        return """
# Kubernetes Deployment for AutoERP
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoerp-deployment
  labels:
    app: autoerp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: autoerp
  template:
    metadata:
      labels:
        app: autoerp
    spec:
      containers:
      - name: autoerp
        image: autoerp:latest
        ports:
        - containerPort: 8000
        env:
        - name: AUTOERP_ENV
          value: "production"
        - name: DATABASE_ENGINE
          value: "postgresql"
        - name: DATABASE_HOST
          valueFrom:
            secretKeyRef:
              name: autoerp-secrets
              key: database-host
        - name: DATABASE_NAME
          value: "autoerp"
        - name: DATABASE_USER
          valueFrom:
            secretKeyRef:
              name: autoerp-secrets
              key: database-user
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autoerp-secrets
              key: database-password
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: autoerp-secrets
              key: secret-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: autoerp-service
spec:
  selector:
    app: autoerp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""
    
    @staticmethod
    def get_nginx_template() -> str:
        """Get Nginx configuration template."""
        return """
# Nginx Configuration for AutoERP
upstream autoerp_backend {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name autoerp.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name autoerp.example.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/autoerp.crt;
    ssl_certificate_key /etc/ssl/private/autoerp.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Static files
    location /static/ {
        alias /var/www/autoerp/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Media files
    location /media/ {
        alias /var/www/autoerp/media/;
        expires 1M;
    }
    
    # Main application
    location / {
        proxy_pass http://autoerp_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check
    location /health {
        proxy_pass http://autoerp_backend/health;
        access_log off;
    }
}
"""
    
    @staticmethod
    def get_systemd_service_template() -> str:
        """Get systemd service template."""
        return """
# Systemd Service for AutoERP
# Save as: /etc/systemd/system/autoerp.service

[Unit]
Description=AutoERP Application
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=notify
User=autoerp
Group=autoerp
WorkingDirectory=/opt/autoerp
Environment=AUTOERP_ENV=production
EnvironmentFile=/etc/autoerp/environment
ExecStart=/opt/autoerp/venv/bin/python -m autoerp.main
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/autoerp/logs /opt/autoerp/uploads /tmp

[Install]
WantedBy=multi-user.target
"""
    
    @staticmethod
    def get_supervisor_template() -> str:
        """Get Supervisor configuration template."""
        return """
# Supervisor Configuration for AutoERP
# Save as: /etc/supervisor/conf.d/autoerp.conf

[program:autoerp]
command=/opt/autoerp/venv/bin/python -m autoerp.main
directory=/opt/autoerp
user=autoerp
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/autoerp/supervisor.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
environment=AUTOERP_ENV="production"

[program:autoerp-worker]
command=/opt/autoerp/venv/bin/python -m autoerp.worker
directory=/opt/autoerp
user=autoerp
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/autoerp/worker.log
numprocs=2
process_name=%(program_name)s_%(process_num)02d
"""


# ==================== CONFIGURATION MIGRATION ====================

class ConfigurationMigration:
    """Handles migration of configuration between versions."""
    
    VERSION_MIGRATIONS = {
        "1.0.0": "_migrate_from_legacy",
        "1.1.0": "_migrate_to_1_1_0",
        "2.0.0": "_migrate_to_2_0_0"
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def migrate_config(self, config_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate configuration from one version to another."""
        current_data = config_data.copy()
        current_version = from_version
        
        # Apply migrations in order
        for version in sorted(self.VERSION_MIGRATIONS.keys()):
            if self._version_greater_than(version, current_version) and self._version_less_equal(version, to_version):
                migration_method = getattr(self, self.VERSION_MIGRATIONS[version])
                current_data = migration_method(current_data)
                current_version = version
                self.logger.info(f"Migrated configuration to version {version}")
        
        return current_data
    
    def _migrate_from_legacy(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from legacy configuration format."""
        # Handle legacy database configuration
        if "db_url" in config_data:
            db_url = config_data.pop("db_url")
            parsed = urlparse(db_url)
            
            config_data["database"] = {
                "engine": parsed.scheme,
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 5432,
                "database": parsed.path.lstrip('/') if parsed.path else "autoerp",
                "username": parsed.username or "",
                "password": parsed.password or ""
            }
        
        # Handle legacy cache configuration
        if "cache_url" in config_data:
            cache_url = config_data.pop("cache_url")
            if cache_url.startswith("redis://"):
                parsed = urlparse(cache_url)
                config_data["cache"] = {
                    "backend": "redis",
                    "redis_host": parsed.hostname or "localhost",
                    "redis_port": parsed.port or 6379,
                    "redis_db": int(parsed.path.lstrip('/')) if parsed.path else 0
                }
        
        return config_data
    
    def _migrate_to_1_1_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to version 1.1.0."""
        # Add new security settings
        if "security" not in config_data:
            config_data["security"] = {}
        
        security = config_data["security"]
        
        # Add new 2FA settings
        if "enable_2fa" not in security:
            security["enable_2fa"] = False
            security["totp_issuer"] = "AutoERP"
            security["backup_codes_count"] = 10
        
        # Add rate limiting
        if "rate_limit_enabled" not in security:
            security["rate_limit_enabled"] = True
            security["rate_limit_requests_per_minute"] = 100
        
        return config_data
    
    def _migrate_to_2_0_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration to version 2.0.0."""
        # Restructure logging configuration
        if "log_level" in config_data:
            level = config_data.pop("log_level")
            config_data["logging"] = {"level": level}
        
        if "log_file" in config_data:
            file_path = config_data.pop("log_file")
            if "logging" not in config_data:
                config_data["logging"] = {}
            config_data["logging"]["file_path"] = file_path
        
        # Add new email queue settings
        if "email" not in config_data:
            config_data["email"] = {}
        
        email = config_data["email"]
        if "queue_enabled" not in email:
            email["queue_enabled"] = True
            email["queue_backend"] = "database"
            email["max_retry_attempts"] = 3
        
        return config_data
    
    def _version_greater_than(self, version1: str, version2: str) -> bool:
        """Compare version strings."""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        while len(v1_parts) < len(v2_parts):
            v1_parts.append(0)
        while len(v2_parts) < len(v1_parts):
            v2_parts.append(0)
        
        return v1_parts > v2_parts
    
    def _version_less_equal(self, version1: str, version2: str) -> bool:
        """Compare version strings."""
        return not self._version_greater_than(version1, version2)


# ==================== CONFIGURATION BACKUP AND RESTORE ====================

class ConfigurationBackup:
    """Handles backup and restore of configuration files."""
    
    def __init__(self, backup_dir: str = "config_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_backup(self, config: AppSettings, backup_name: Optional[str] = None) -> Path:
        """Create a backup of the current configuration."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"config_backup_{timestamp}"
        
        backup_file = self.backup_dir / f"{backup_name}.yml"
        
        # Convert config to dictionary
        config_dict = asdict(config)
        
        # Convert enums to strings
        config_dict["environment"] = config.environment.value
        config_dict["database"]["engine"] = config.database.engine.value
        config_dict["cache"]["backend"] = config.cache.backend.value
        config_dict["logging"]["level"] = config.logging.level.value
        
        # Add metadata
        backup_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": config.app_version,
                "environment": config.environment.value,
                "backup_name": backup_name
            },
            "configuration": config_dict
        }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            yaml.dump(backup_data, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration backup created: {backup_file}")
        return backup_file
    
    def restore_backup(self, backup_file: Union[str, Path]) -> AppSettings:
        """Restore configuration from backup."""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = yaml.safe_load(f)
        
        if "configuration" not in backup_data:
            raise ValueError("Invalid backup file format")
        
        config_data = backup_data["configuration"]
        
        # Handle version migration if needed
        backup_version = backup_data.get("metadata", {}).get("version", "1.0.0")
        current_version = "2.0.0"  # Current version
        
        if backup_version != current_version:
            migration = ConfigurationMigration()
            config_data = migration.migrate_config(config_data, backup_version, current_version)
        
        # Create AppSettings object
        loader = ConfigurationLoader()
        config = loader._create_app_settings(config_data)
        
        self.logger.info(f"Configuration restored from backup: {backup_file}")
        return config
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.yml"):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = yaml.safe_load(f)
                
                metadata = backup_data.get("metadata", {})
                backups.append({
                    "file": backup_file,
                    "name": metadata.get("backup_name", backup_file.stem),
                    "created_at": metadata.get("created_at"),
                    "version": metadata.get("version"),
                    "environment": metadata.get("environment"),
                    "size": backup_file.stat().st_size
                })
            
            except Exception as e:
                self.logger.warning(f"Error reading backup file {backup_file}: {e}")
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backup files, keeping only the specified number."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        deleted_count = 0
        for backup in backups[keep_count:]:
            try:
                backup["file"].unlink()
                deleted_count += 1
                self.logger.info(f"Deleted old backup: {backup['name']}")
            except Exception as e:
                self.logger.error(f"Error deleting backup {backup['name']}: {e}")
        
        return deleted_count


# ==================== CONFIGURATION WATCHER ====================

class ConfigurationWatcher:
    """Watches configuration files for changes and reloads automatically."""
    
    def __init__(self, config_manager: ConfigurationManager, watch_files: List[str]):
        self.config_manager = config_manager
        self.watch_files = [Path(f) for f in watch_files]
        self.logger = logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread = None
        self._file_timestamps = {}
    
    def start_watching(self) -> None:
        """Start watching configuration files."""
        if self._thread and self._thread.is_alive():
            self.logger.warning("Configuration watcher is already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        
        self.logger.info(f"Started watching {len(self.watch_files)} configuration files")
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        self.logger.info("Stopped configuration file watcher")
    
    def _watch_loop(self) -> None:
        """Main watching loop."""
        # Initialize file timestamps
        for file_path in self.watch_files:
            if file_path.exists():
                self._file_timestamps[file_path] = file_path.stat().st_mtime
        
        while not self._stop_event.wait(1.0):  # Check every second
            try:
                for file_path in self.watch_files:
                    if not file_path.exists():
                        continue
                    
                    current_mtime = file_path.stat().st_mtime
                    last_mtime = self._file_timestamps.get(file_path, 0)
                    
                    if current_mtime > last_mtime:
                        self._file_timestamps[file_path] = current_mtime
                        self._on_file_changed(file_path)
            
            except Exception as e:
                self.logger.error(f"Error in configuration watcher: {e}")
    
    def _on_file_changed(self, file_path: Path) -> None:
        """Handle configuration file change."""
        try:
            self.logger.info(f"Configuration file changed: {file_path}")
            
            # Reload configuration
            self.config_manager.reload_config()
            
            self.logger.info("Configuration reloaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")


# ==================== CONFIGURATION VALIDATION RULES ====================

class ConfigurationRules:
    """Advanced configuration validation rules."""
    
    @staticmethod
    def validate_production_readiness(config: AppSettings) -> List[ValidationError]:
        """Validate configuration for production deployment."""
        errors = []
        
        if config.environment == Environment.PRODUCTION:
            # Security checks
            if config.debug:
                errors.append(ValidationError("debug", True, "Debug mode must be disabled in production"))
            
            if config.security.secret_key in ("", "dev-secret-key-change-in-production", "test-secret-key"):
                errors.append(ValidationError("security.secret_key", config.security.secret_key, 
                                            "Must use a secure secret key in production"))
            
            if not config.security.force_https and config.base_url.startswith("https://"):
                errors.append(ValidationError("security.force_https", False,
                                            "HTTPS enforcement should be enabled for HTTPS sites"))
            
            if not config.session_cookie_secure and config.base_url.startswith("https://"):
                errors.append(ValidationError("session_cookie_secure", False,
                                            "Secure cookies should be enabled for HTTPS"))
            
            # Database checks
            if config.database.engine == DatabaseEngine.SQLITE:
                errors.append(ValidationError("database.engine", "sqlite",
                                            "SQLite is not recommended for production use"))
            
            if config.database.echo_queries:
                errors.append(ValidationError("database.echo_queries", True,
                                            "Query logging should be disabled in production"))
            
            # Logging checks
            if config.logging.level == LogLevel.DEBUG:
                errors.append(ValidationError("logging.level", "DEBUG",
                                            "Debug logging should be disabled in production"))
        
        return errors
    
    @staticmethod
    def validate_security_compliance(config: AppSettings) -> List[ValidationError]:
        """Validate configuration for security compliance."""
        errors = []
        
        security = config.security
        
        # Password policy
        if security.password_min_length < 8:
            errors.append(ValidationError("security.password_min_length", security.password_min_length,
                                        "Minimum password length should be at least 8 for security compliance"))
        
        # Session security
        if security.session_timeout_minutes > 480:  # 8 hours
            errors.append(ValidationError("security.session_timeout_minutes", security.session_timeout_minutes,
                                        "Session timeout should not exceed 8 hours"))
        
        # Login security
        if security.max_login_attempts > 10:
            errors.append(ValidationError("security.max_login_attempts", security.max_login_attempts,
                                        "Maximum login attempts should not exceed 10"))
        
        if security.lockout_duration_minutes < 15:
            errors.append(ValidationError("security.lockout_duration_minutes", security.lockout_duration_minutes,
                                        "Account lockout duration should be at least 15 minutes"))
        
        # Rate limiting
        if not security.rate_limit_enabled:
            errors.append(ValidationError("security.rate_limit_enabled", False,
                                        "Rate limiting should be enabled for security"))
        
        return errors
    
    @staticmethod
    def validate_performance_settings(config: AppSettings) -> List[ValidationError]:
        """Validate configuration for performance."""
        errors = []
        
        # Database connection pool
        db = config.database
        if db.pool_size < 5:
            errors.append(ValidationError("database.pool_size", db.pool_size,
                                        "Database pool size should be at least 5 for good performance"))
        
        if db.pool_size > 50:
            errors.append(ValidationError("database.pool_size", db.pool_size,
                                        "Database pool size should not exceed 50 to avoid resource exhaustion"))
        
        # Cache settings
        cache = config.cache
        if not cache.enabled and config.environment == Environment.PRODUCTION:
            errors.append(ValidationError("cache.enabled", False,
                                        "Caching should be enabled in production for performance"))
        
        if cache.default_timeout > 3600:  # 1 hour
            errors.append(ValidationError("cache.default_timeout", cache.default_timeout,
                                        "Cache timeout should not exceed 1 hour for data freshness"))
        
        return errors


# ==================== ENHANCED EXAMPLES ====================

if __name__ == "__main__":
    """Enhanced examples with additional features."""
    
    print("\nAutoERP Configuration System - Extended Examples")
    print("=" * 50)
    
    # Example 10: Configuration migration
    print("\n10. Testing configuration migration:")
    try:
        migration = ConfigurationMigration()
        
        # Simulate legacy config
        legacy_config = {
            "db_url": "postgresql://user:pass@localhost:5432/autoerp",
            "cache_url": "redis://localhost:6379/0",
            "log_level": "INFO"
        }
        
        migrated = migration.migrate_config(legacy_config, "1.0.0", "2.0.0")
        print("   ✅ Configuration migrated successfully")
        print(f"   Database engine: {migrated.get('database', {}).get('engine', 'N/A')}")
        print(f"   Cache backend: {migrated.get('cache', {}).get('backend', 'N/A')}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 11: Configuration backup and restore
    print("\n11. Testing configuration backup and restore:")
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_config(validate=False)
        
        backup = ConfigurationBackup()
        backup_file = backup.create_backup(config, "test_backup")
        print(f"   ✅ Backup created: {backup_file}")
        
        # List backups
        backups = backup.list_backups()
        print(f"   📋 Found {len(backups)} backups")
        
        # Restore backup
        restored_config = backup.restore_backup(backup_file)
        print("   ✅ Configuration restored from backup")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 12: Advanced validation
    print("\n12. Testing advanced validation rules:")
    try:
        # Create production config
        prod_config = AppSettings(
            environment=Environment.PRODUCTION,
            debug=True,  # Invalid for production
            security=SecuritySettings(
                secret_key="weak-key",  # Invalid for production
                password_min_length=4   # Too weak
            )
        )
        
        # Test production readiness
        prod_errors = ConfigurationRules.validate_production_readiness(prod_config)
        print(f"   Production readiness: {len(prod_errors)} issues found")
        
        # Test security compliance
        security_errors = ConfigurationRules.validate_security_compliance(prod_config)
        print(f"   Security compliance: {len(security_errors)} issues found")
        
        for error in (prod_errors + security_errors)[:3]:
            print(f"     - {error}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 13: Configuration templates
    print("\n13. Generating deployment templates:")
    try:
        templates = ConfigurationTemplates()
        
        # Generate Docker Compose
        with open("docker-compose.yml", "w") as f:
            f.write(templates.get_docker_compose_template())
        print("   ✅ Generated docker-compose.yml")
        
        # Generate Nginx config
        with open("nginx.conf", "w") as f:
            f.write(templates.get_nginx_template())
        print("   ✅ Generated nginx.conf")
        
        # Generate systemd service
        with open("autoerp.service", "w") as f:
            f.write(templates.get_systemd_service_template())
        print("   ✅ Generated autoerp.service")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Example 14: Environment-specific configuration files
    print("\n14. Creating environment-specific configurations:")
    
    environments = {
        "development": {
            "environment": "development",
            "debug": True,
            "database": {"engine": "sqlite", "sqlite_file": "dev.db"},
            "logging": {"level": "DEBUG"}
        },
        "testing": {
            "environment": "testing",
            "testing": True,
            "database": {"engine": "sqlite", "sqlite_file": ":memory:"},
            "logging": {"level": "WARNING"}
        },
        "production": {
            "environment": "production",
            "debug": False,
            "database": {"engine": "postgresql", "host": "prod-db"},
            "security": {"force_https": True},
            "logging": {"level": "ERROR"}
        }
    }
    
    for env_name, env_config in environments.items():
        try:
            filename = f"config.{env_name}.yml"
            with open(filename, "w") as f:
                yaml.dump(env_config, f, default_flow_style=False, indent=2)
            print(f"   ✅ Created {filename}")
        except Exception as e:
            print(f"   Error creating {filename}: {e}")
    
    # Example 15: Configuration schema export
    print("\n15. Exporting configuration schema:")
    try:
        # Generate JSON schema for configuration
        schema = {
            "type": "object",
            "properties": {
                "environment": {
                    "type": "string",
                    "enum": [e.value for e in Environment],
                    "description": "Deployment environment"
                },
                "debug": {
                    "type": "boolean",
                    "description": "Enable debug mode"
                },
                "host": {
                    "type": "string",
                    "description": "Server host address"
                },
                "port": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 65535,
                    "description": "Server port number"
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "engine": {
                            "type": "string",
                            "enum": [e.value for e in DatabaseEngine]
                        },
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "database": {"type": "string"},
                        "username": {"type": "string"},
                        "password": {"type": "string"}
                    },
                    "required": ["engine"]
                }
            },
            "required": ["environment"]
        }
        
        with open("config-schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        print("   ✅ Generated config-schema.json")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Extended configuration examples completed!")
    print("\nGenerated files:")
    print("- Configuration examples: example_config.yml, example_config.json")
    print("- Environment configs: config.development.yml, config.testing.yml, config.production.yml")
    print("- Environment file: sample.env")
    print("- Deployment templates: docker-compose.yml, nginx.conf, autoerp.service")
    print("- Schema: config-schema.json")
    print("\nNote: Remember to:")
    print("1. Change default secrets and passwords")
    print("2. Review security settings for production")
    print("3. Customize database and cache settings")
    print("4. Set up proper SSL certificates")
    print("5. Configure monitoring and logging")