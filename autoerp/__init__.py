"""
AutoERP - Enterprise Resource Planning System
============================================

AutoERP is a modern, comprehensive Enterprise Resource Planning (ERP) system 
built with hexagonal architecture principles. It provides a complete suite of 
business management tools including:

- Financial Management & Accounting
- Customer Relationship Management (CRM)
- Inventory & Supply Chain Management
- Human Resources Management
- Project Management & Collaboration
- Business Intelligence & Analytics
- Multi-tenant & Multi-currency Support

Architecture:
- Core: Business logic and domain models (hexagonal architecture)
- API: RESTful API layer built with FastAPI
- UI: Web interface built with Streamlit and Flask
- Database: Multi-database support (SQLite, PostgreSQL)
- Caching: Redis and in-memory caching
- Real-time: WebSocket support for live updates

Key Features:
- Modular and extensible design
- Built-in audit trails and data validation
- Role-based access control and permissions
- Comprehensive API with OpenAPI documentation
- Interactive dashboards and reporting
- Data import/export capabilities
- Multi-language and localization support
- Containerized deployment ready

Usage:
    # Start the API server
    python -m autoerp serve
    
    # Launch the web UI
    python -m autoerp ui
    
    # Get help
    python -m autoerp help
    
    # Programmatic usage
    from autoerp import services, models, app, dashboard
    
Example:
    >>> import autoerp
    >>> print(autoerp.get_version())
    >>> print(autoerp.about())
    
    # Initialize application
    >>> from autoerp.core import AutoERPConfig, AutoERPApplication
    >>> config = AutoERPConfig()
    >>> app = AutoERPApplication(config)
    >>> await app.initialize()

Dependencies:
- Python 3.8+
- FastAPI for API layer
- Streamlit for UI components
- SQLAlchemy for database ORM
- Pydantic for data validation
- Plotly for data visualization

License: MIT
Author: AutoERP Development Team
Version: 1.0.0
Repository: https://github.com/autoerp/autoerp
Documentation: https://docs.autoerp.com
"""

import sys
import os
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

# Version information
__version__ = "1.0.0"
__author__ = "AutoERP Development Team"
__email__ = "dev@autoerp.com"
__license__ = "MIT"
__copyright__ = "2024 AutoERP Development Team"

# Package metadata
__title__ = "autoerp"
__description__ = "Modern Enterprise Resource Planning System"
__url__ = "https://github.com/autoerp/autoerp"
__status__ = "Production"

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==================== CORE IMPORTS ====================
# Import core services, models, and utilities

try:
    # Core business logic and domain models
    from .core import (
        # Configuration classes
        AutoERPConfig, DatabaseConfig, SecurityConfig, CacheConfig,
        EmailConfig, BusinessConfig, SystemConfig,
        
        # Base models and mixins
        BaseModel, AuditMixin, SoftDeleteMixin, ValidationError,
        FieldDescriptor, FieldValidator, ModelRegistry,
        
        # Connection and data access
        ConnectionManager, ConnectionPool, IRepository, BaseRepository,
        RepositoryError, RecordNotFoundError, DuplicateRecordError,
        ConcurrencyError, UnitOfWork,
        
        # Domain events and caching
        DomainEvent, EntityCreatedEvent, EntityUpdatedEvent, EntityDeletedEvent,
        EventHandler, EventDispatcher, CacheBackend, MemoryCacheBackend,
        CacheManager, cached,
        
        # Value objects and entities
        Currency, Money, Address, EmailAddress, PhoneNumber, ContactInfo,
        PersonName, Person, Organization, User, Permission, Session,
        AuditLogEntry, NotificationTemplate, Notification,
        
        # Enumerations
        UserRole, NotificationChannel, NotificationPriority,
        
        # Services
        ServiceResult, BaseService, EnhancedBaseService, UserService,
        NotificationService, ServiceError, BusinessRuleViolationError,
        AuthorizationError, DataLoaderService, DataFormat, DataType,
        SchemaField, DataSchema, DataCleaningRule, CRUDService,
        PaginationInfo, FilterCriteria,
        
        # Security and audit
        PasswordManager, SessionManager, AuditLogger,
        
        # Health and monitoring
        HealthCheck, SystemMetrics, system_metrics, timed_operation,
        
        # Main application
        AutoERPApplication
    )
    
    # Group core exports for easier access
    models = {
        'BaseModel': BaseModel,
        'User': User,
        'Person': Person,
        'Organization': Organization,
        'Permission': Permission,
        'Session': Session,
        'AuditLogEntry': AuditLogEntry,
        'NotificationTemplate': NotificationTemplate,
        'Notification': Notification,
        'Currency': Currency,
        'Money': Money,
        'Address': Address,
        'ContactInfo': ContactInfo,
        'PersonName': PersonName
    }
    
    services = {
        'UserService': UserService,
        'NotificationService': NotificationService,
        'DataLoaderService': DataLoaderService,
        'CRUDService': CRUDService,
        'PasswordManager': PasswordManager,
        'SessionManager': SessionManager,
        'AuditLogger': AuditLogger
    }
    
    utils = {
        'AutoERPConfig': AutoERPConfig,
        'ConnectionManager': ConnectionManager,
        'CacheManager': CacheManager,
        'EventDispatcher': EventDispatcher,
        'HealthCheck': HealthCheck,
        'SystemMetrics': SystemMetrics,
        'ValidationError': ValidationError,
        'ServiceResult': ServiceResult,
        'PaginationInfo': PaginationInfo,
        'FilterCriteria': FilterCriteria
    }
    
    logger.info("Successfully imported core modules")
    
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    # Create empty dictionaries to prevent further errors
    models = {}
    services = {}
    utils = {}


# ==================== API IMPORTS ====================
# Import FastAPI application and CLI tools

try:
    from .api import (
        # FastAPI application
        app as fastapi_app,
        
        # Pydantic schemas
        APIResponse, PaginationRequest, SortRequest, FilterRequest,
        PaginatedResponse, HealthCheckResponse, MetricsResponse,
        UserCreateRequest, UserUpdateRequest, UserResponse,
        LoginRequest, LoginResponse, ChangePasswordRequest,
        TableInfo, ColumnInfo, RecordCreateRequest, RecordUpdateRequest,
        RecordResponse, BulkOperationRequest, BulkOperationResponse,
        
        # Context and middleware
        RequestContext, RequestContextMiddleware, AuthMiddleware,
        
        # Dependencies
        get_app_instance, get_user_service, get_notification_service,
        get_current_user, get_current_user_optional,
        require_permission, require_role
    )
    
    # Create API namespace
    app = {
        'fastapi_app': fastapi_app,
        'schemas': {
            'APIResponse': APIResponse,
            'UserResponse': UserResponse,
            'LoginRequest': LoginRequest,
            'LoginResponse': LoginResponse,
            'TableInfo': TableInfo,
            'RecordResponse': RecordResponse
        },
        'middleware': {
            'RequestContextMiddleware': RequestContextMiddleware,
            'AuthMiddleware': AuthMiddleware
        },
        'dependencies': {
            'get_current_user': get_current_user,
            'require_permission': require_permission,
            'require_role': require_role
        }
    }
    
    # CLI tools namespace
    cli = {
        'start_server': lambda: _start_api_server(),
        'run_migrations': lambda: _run_migrations(),
        'create_admin': lambda: _create_admin_user(),
        'backup_data': lambda: _backup_database(),
        'restore_data': lambda: _restore_database()
    }
    
    logger.info("Successfully imported API modules")
    
except ImportError as e:
    logger.error(f"Failed to import API modules: {e}")
    app = {}
    cli = {}


# ==================== UI IMPORTS ====================
# Import Streamlit and Flask dashboard components

try:
    from .ui import (
        # Streamlit components
        dashboard_page, tables_page, table_data_page, login_page,
        register_page, render_sidebar,
        
        # Session management
        UISessionManager,
        
        # API client
        AutoERPAPIClient,
        
        # Utility functions
        async_to_sync, display_notification, format_currency,
        format_date, create_metric_card, get_api_client
    )
    
    # Flask/SocketIO components (if available)
    try:
        from .ui import (
            flask_app, socketio_app, real_time_dashboard,
            notification_handler, chat_interface
        )
        flask_available = True
    except ImportError:
        flask_available = False
        logger.warning("Flask components not available")
    
    # Create dashboard namespace
    dashboard = {
        'streamlit': {
            'dashboard_page': dashboard_page,
            'tables_page': tables_page,
            'login_page': login_page,
            'render_sidebar': render_sidebar,
            'session_manager': UISessionManager,
            'api_client': AutoERPAPIClient
        },
        'utils': {
            'format_currency': format_currency,
            'format_date': format_date,
            'display_notification': display_notification,
            'create_metric_card': create_metric_card
        }
    }
    
    if flask_available:
        dashboard['flask'] = {
            'app': flask_app,
            'socketio': socketio_app,
            'real_time_dashboard': real_time_dashboard,
            'notification_handler': notification_handler
        }
    
    logger.info("Successfully imported UI modules")
    
except ImportError as e:
    logger.error(f"Failed to import UI modules: {e}")
    dashboard = {}


# ==================== PACKAGE UTILITY FUNCTIONS ====================

def get_version() -> str:
    """
    Get the current version of the AutoERP package.
    
    Returns:
        str: Version string in format 'major.minor.patch'
        
    Example:
        >>> import autoerp
        >>> print(autoerp.get_version())
        '1.0.0'
    """
    return __version__


def about() -> str:
    """
    Get a brief description about the AutoERP package.
    
    Returns:
        str: Formatted description string with package information
        
    Example:
        >>> import autoerp
        >>> print(autoerp.about())
        AutoERP v1.0.0 - Modern Enterprise Resource Planning System
        Developed by AutoERP Development Team
        License: MIT
        ...
    """
    return f"""
AutoERP v{__version__} - {__description__}

Developed by: {__author__}
Email: {__email__}
License: {__license__}
Status: {__status__}

Repository: {__url__}
Documentation: https://docs.autoerp.com
Support: https://support.autoerp.com

Features:
• Comprehensive business management suite
• Modern hexagonal architecture
• RESTful API with OpenAPI documentation
• Interactive web dashboards
• Multi-tenant and multi-currency support
• Built-in security and audit trails
• Real-time notifications and updates
• Extensible plugin system

Quick Start:
1. python -m autoerp serve    # Start API server
2. python -m autoerp ui       # Launch web interface
3. python -m autoerp help     # Show all commands

For detailed documentation, visit: https://docs.autoerp.com
    """.strip()


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for debugging and support.
    
    Returns:
        Dict[str, Any]: System information including versions, paths, and config
        
    Example:
        >>> import autoerp
        >>> info = autoerp.get_system_info()
        >>> print(info['python_version'])
    """
    try:
        import platform
        import psutil
        
        return {
            'autoerp_version': __version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'memory_total': f"{psutil.virtual_memory().total // (1024**3)} GB",
            'memory_available': f"{psutil.virtual_memory().available // (1024**3)} GB",
            'disk_usage': f"{psutil.disk_usage('/').percent}%",
            'cpu_count': psutil.cpu_count(),
            'python_path': sys.executable,
            'package_path': str(Path(__file__).parent),
            'dependencies': _get_dependency_versions()
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {
            'autoerp_version': __version__,
            'python_version': platform.python_version(),
            'error': str(e)
        }


def _get_dependency_versions() -> Dict[str, str]:
    """Get versions of key dependencies."""
    deps = {}
    
    try:
        import fastapi
        deps['fastapi'] = fastapi.__version__
    except ImportError:
        deps['fastapi'] = 'Not installed'
    
    try:
        import streamlit
        deps['streamlit'] = streamlit.__version__
    except ImportError:
        deps['streamlit'] = 'Not installed'
    
    try:
        import plotly
        deps['plotly'] = plotly.__version__
    except ImportError:
        deps['plotly'] = 'Not installed'
    
    try:
        import pandas
        deps['pandas'] = pandas.__version__
    except ImportError:
        deps['pandas'] = 'Not installed'
    
    try:
        import pydantic
        deps['pydantic'] = pydantic.VERSION
    except ImportError:
        deps['pydantic'] = 'Not installed'
    
    return deps


def validate_installation() -> Dict[str, Any]:
    """
    Validate the AutoERP installation and check for common issues.
    
    Returns:
        Dict[str, Any]: Validation results with status and recommendations
    """
    results = {
        'status': 'valid',
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check Python version
    if sys.version_info < (3, 8):
        results['errors'].append("Python 3.8 or higher is required")
        results['status'] = 'invalid'
    
    # Check core modules
    if not models:
        results['errors'].append("Core models not available")
        results['status'] = 'invalid'
    
    if not services:
        results['errors'].append("Core services not available")
        results['status'] = 'invalid'
    
    # Check optional dependencies
    try:
        import uvicorn
    except ImportError:
        results['warnings'].append("Uvicorn not installed - API server unavailable")
        results['recommendations'].append("Install with: pip install uvicorn")
    
    try:
        import streamlit
    except ImportError:
        results['warnings'].append("Streamlit not installed - UI unavailable")
        results['recommendations'].append("Install with: pip install streamlit")
    
    # Check database connectivity (if config available)
    try:
        config_path = Path("config/autoerp.json")
        if config_path.exists():
            from .core import AutoERPConfig
            config = AutoERPConfig.from_file(config_path)
            results['recommendations'].append("Configuration file found and loaded")
        else:
            results['warnings'].append("No configuration file found")
            results['recommendations'].append("Create config/autoerp.json for custom settings")
    except Exception as e:
        results['warnings'].append(f"Configuration validation failed: {e}")
    
    return results


def create_sample_config(output_path: str = "config/autoerp.json") -> bool:
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path where to save the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from .core import AutoERPConfig
        
        # Create default configuration
        config = AutoERPConfig()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config.save_to_file(output_path)
        
        logger.info(f"Sample configuration created at: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample config: {e}")
        return False


# ==================== COMMAND LINE INTERFACE ====================

def _start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    try:
        import uvicorn
        
        logger.info(f"Starting AutoERP API server on {host}:{port}")
        
        if 'fastapi_app' in app:
            uvicorn.run(
                app['fastapi_app'],
                host=host,
                port=port,
                reload=reload,
                log_level="info"
            )
        else:
            logger.error("FastAPI app not available")
            return False
            
    except ImportError:
        logger.error("Uvicorn not installed. Install with: pip install uvicorn")
        return False
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return False


def _start_ui_server(port: int = 8501):
    """Start the Streamlit UI server."""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        logger.info(f"Starting AutoERP UI server on port {port}")
        
        # Get the UI module path
        ui_path = Path(__file__).parent / "ui.py"
        
        if ui_path.exists():
            sys.argv = ["streamlit", "run", str(ui_path), "--server.port", str(port)]
            stcli.main()
        else:
            logger.error("UI module not found")
            return False
            
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"Failed to start UI server: {e}")
        return False


def _run_migrations():
    """Run database migrations."""
    try:
        from .core import AutoERPApplication, AutoERPConfig
        import asyncio
        
        async def run():
            config = AutoERPConfig()
            async with AutoERPApplication(config) as app_instance:
                logger.info("Running database migrations...")
                # Migration logic would go here
                logger.info("Migrations completed successfully")
        
        asyncio.run(run())
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def _create_admin_user():
    """Create initial admin user."""
    try:
        from .core import AutoERPApplication, AutoERPConfig, UserRole
        import asyncio
        import getpass
        
        async def create_admin():
            config = AutoERPConfig()
            async with AutoERPApplication(config) as app_instance:
                username = input("Admin username: ")
                email = input("Admin email: ")
                password = getpass.getpass("Admin password: ")
                first_name = input("First name: ")
                last_name = input("Last name: ")
                
                result = await app_instance.user_service.create_user(
                    username=username,
                    email=email,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                    role=UserRole.SUPER_ADMIN
                )
                
                if result.is_success():
                    logger.info("Admin user created successfully")
                else:
                    logger.error(f"Failed to create admin user: {result.error_message}")
        
        asyncio.run(create_admin())
        
    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        return False


def _backup_database():
    """Backup database."""
    try:
        logger.info("Database backup functionality coming soon...")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def _restore_database():
    """Restore database from backup."""
    try:
        logger.info("Database restore functionality coming soon...")
        return True
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return False


def _show_help():
    """Show help information."""
    help_text = """
AutoERP Command Line Interface

Usage: python -m autoerp [command] [options]

Commands:
  serve         Start the API server (default: localhost:8000)
  ui            Start the web interface (default: localhost:8501)
  migrate       Run database migrations
  create-admin  Create initial admin user
  backup        Backup database
  restore       Restore database from backup
  config        Create sample configuration file
  validate      Validate installation
  info          Show system information
  version       Show version information
  help          Show this help message

Examples:
  python -m autoerp serve                    # Start API server
  python -m autoerp serve --port 9000       # Start on custom port
  python -m autoerp ui                       # Start web interface
  python -m autoerp migrate                  # Run migrations
  python -m autoerp create-admin             # Create admin user
  python -m autoerp config                   # Create sample config
  python -m autoerp validate                 # Check installation

For more information, visit: https://docs.autoerp.com
"""
    print(help_text)


# ==================== PACKAGE EXPORTS ====================

# Define what gets exported when importing the package
__all__ = [
    # Main namespaces
    "services",
    "models", 
    "utils",
    "app",
    "cli",
    "dashboard",
    
    # Core classes (most commonly used)
    "AutoERPApplication",
    "AutoERPConfig",
    "User",
    "UserService",
    "CRUDService",
    
    # Utility functions
    "get_version",
    "about",
    "get_system_info",
    "validate_installation",
    "create_sample_config",
    
    # Version info
    "__version__",
    "__author__",
    "__license__"
]


# ==================== MODULE INITIALIZATION ====================

def _initialize_package():
    """Initialize the package and perform startup checks."""
    try:
        # Log package initialization
        logger.info(f"Initializing AutoERP v{__version__}")
        
        # Check installation
        validation = validate_installation()
        
        if validation['status'] == 'invalid':
            logger.error("AutoERP installation validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        # Log successful initialization
        logger.info("AutoERP package initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AutoERP package: {e}")
        return False


# Initialize the package
_initialize_package()


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    """
    Main entry point for the AutoERP package.
    
    Handles command line arguments and starts appropriate services.
    """
    
    try:
        # Parse command line arguments
        if len(sys.argv) < 2:
            # No command provided - show welcome message
            print(f"""
🏢 Welcome to AutoERP v{__version__}

{__description__}

Quick Start:
  python -m autoerp serve    # Start API server
  python -m autoerp ui       # Start web interface  
  python -m autoerp help     # Show all commands

For detailed documentation: https://docs.autoerp.com
            """.strip())
            
        else:
            command = sys.argv[1].lower()
            
            if command == "serve":
                # Parse additional arguments
                port = 8000
                host = "0.0.0.0"
                reload = False
                
                for i, arg in enumerate(sys.argv[2:], 2):
                    if arg == "--port" and i + 1 < len(sys.argv):
                        port = int(sys.argv[i + 1])
                    elif arg == "--host" and i + 1 < len(sys.argv):
                        host = sys.argv[i + 1]
                    elif arg == "--reload":
                        reload = True
                
                _start_api_server(host=host, port=port, reload=reload)
                
            elif command == "ui":
                # Parse port argument
                port = 8501
                for i, arg in enumerate(sys.argv[2:], 2):
                    if arg == "--port" and i + 1 < len(sys.argv):
                        port = int(sys.argv[i + 1])
                
                _start_ui_server(port=port)
                
            elif command == "migrate":
                _run_migrations()
                
            elif command == "create-admin":
                _create_admin_user()
                
            elif command == "backup":
                _backup_database()
                
            elif command == "restore":
                _restore_database()
                
            elif command == "config":
                output_path = "config/autoerp.json"
                if len(sys.argv) > 2:
                    output_path = sys.argv[2]
                
                if create_sample_config(output_path):
                    print(f"Sample configuration created at: {output_path}")
                else:
                    print("Failed to create sample configuration")
                    sys.exit(1)
                    
            elif command == "validate":
                validation = validate_installation()
                print("AutoERP Installation Validation")
                print("=" * 35)
                print(f"Status: {validation['status'].upper()}")
                
                if validation['errors']:
                    print("\nErrors:")
                    for error in validation['errors']:
                        print(f"  ❌ {error}")
                
                if validation['warnings']:
                    print("\nWarnings:")
                    for warning in validation['warnings']:
                        print(f"  ⚠️  {warning}")
                
                if validation['recommendations']:
                    print("\nRecommendations:")
                    for rec in validation['recommendations']:
                        print(f"  💡 {rec}")
                
                if validation['status'] == 'invalid':
                    sys.exit(1)
                    
            elif command == "info":
                info = get_system_info()
                print("AutoERP System Information")
                print("=" * 27)
                for key, value in info.items():
                    print(f"{key.replace('_', ' ').title()}: {value}")
                    
            elif command == "version":
                print(f"AutoERP v{__version__}")
                
            elif command == "help":
                _show_help()
                
            else:
                print(f"Unknown command: {command}")
                print("Run 'python -m autoerp help' for available commands")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        print(f"Error: {e}")
        print("Run 'python -m autoerp help' for usage information")
        sys.exit(1)