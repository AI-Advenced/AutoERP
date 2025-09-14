"""
AutoERP Database Module
=======================

This module provides database connectivity, session management, and migration support
for the AutoERP system. It supports SQLite, PostgreSQL, and MySQL databases with
proper connection pooling and thread safety.

Author: AutoERP Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import logging
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Dict, Any, Optional, List, Union, Generator, AsyncGenerator, Type
from pathlib import Path
from datetime import datetime, timezone
import asyncio
import warnings

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Engine, MetaData, Table, Column, Integer, String, DateTime, 
    Boolean, Text, ForeignKey, Index, event, pool, exc, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    sessionmaker, Session, scoped_session, relationship, backref
)
from sqlalchemy.pool import (
    QueuePool, StaticPool, NullPool, AssertionPool, SingletonThreadPool
)
from sqlalchemy.sql import text
from sqlalchemy.engine.events import PoolEvents
from sqlalchemy.exc import (
    SQLAlchemyError, DisconnectionError, TimeoutError, 
    IntegrityError, DataError, OperationalError
)

# Async SQLAlchemy (optional)
try:
    from sqlalchemy.ext.asyncio import (
        create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
    )
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    AsyncEngine = None
    AsyncSession = None

# Alembic imports
try:
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig
    from alembic.runtime.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory
    ALEMBIC_AVAILABLE = True
except ImportError:
    ALEMBIC_AVAILABLE = False
    warnings.warn("Alembic not available. Database migrations disabled.")

# Local imports
from .config import DatabaseSettings, LogLevel, Environment


# ==================== LOGGING SETUP ====================

logger = logging.getLogger(__name__)

# SQLAlchemy logging configuration
def setup_database_logging(log_level: LogLevel = LogLevel.INFO, echo_queries: bool = False):
    """
    Configure database logging settings.
    
    Args:
        log_level: Logging level for database operations
        echo_queries: Whether to echo SQL queries to console
    """
    # Set SQLAlchemy logger levels
    logging.getLogger('sqlalchemy.engine').setLevel(log_level.value)
    logging.getLogger('sqlalchemy.pool').setLevel(log_level.value)
    
    if echo_queries:
        logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


# ==================== BASE MODEL DEFINITIONS ====================

Base = declarative_base()
metadata = Base.metadata


class DatabaseModel(Base):
    """
    Base model class for all database entities.
    Provides common columns and functionality.
    """
    __abstract__ = True
    
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True, nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"


# ==================== CONNECTION STATISTICS ====================

class ConnectionStats:
    """
    Tracks database connection statistics and performance metrics.
    """
    
    def __init__(self):
        self.reset_stats()
        self._lock = threading.Lock()
    
    def reset_stats(self):
        """Reset all statistics to zero."""
        with getattr(self, '_lock', threading.Lock()):
            self.total_connections = 0
            self.active_connections = 0
            self.peak_connections = 0
            self.failed_connections = 0
            self.total_queries = 0
            self.slow_queries = 0
            self.query_times = []
            self.connection_times = []
            self.errors = []
            self.start_time = datetime.now(timezone.utc)
    
    def record_connection_created(self):
        """Record a new connection creation."""
        with self._lock:
            self.total_connections += 1
            self.active_connections += 1
            self.peak_connections = max(self.peak_connections, self.active_connections)
    
    def record_connection_closed(self):
        """Record a connection closure."""
        with self._lock:
            self.active_connections = max(0, self.active_connections - 1)
    
    def record_connection_failed(self, error: Exception):
        """Record a failed connection attempt."""
        with self._lock:
            self.failed_connections += 1
            self.errors.append({
                'timestamp': datetime.now(timezone.utc),
                'type': 'connection_failed',
                'error': str(error)
            })
    
    def record_query_executed(self, duration: float, query: str = ""):
        """Record a query execution."""
        with self._lock:
            self.total_queries += 1
            self.query_times.append(duration)
            
            # Keep only last 1000 query times for memory efficiency
            if len(self.query_times) > 1000:
                self.query_times = self.query_times[-1000:]
            
            # Record slow queries (> 1 second)
            if duration > 1.0:
                self.slow_queries += 1
                self.errors.append({
                    'timestamp': datetime.now(timezone.utc),
                    'type': 'slow_query',
                    'duration': duration,
                    'query': query[:200] + '...' if len(query) > 200 else query
                })
    
    def record_connection_time(self, duration: float):
        """Record connection establishment time."""
        with self._lock:
            self.connection_times.append(duration)
            
            # Keep only last 100 connection times
            if len(self.connection_times) > 100:
                self.connection_times = self.connection_times[-100:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            avg_query_time = (
                sum(self.query_times) / len(self.query_times) 
                if self.query_times else 0
            )
            
            avg_connection_time = (
                sum(self.connection_times) / len(self.connection_times) 
                if self.connection_times else 0
            )
            
            queries_per_second = self.total_queries / uptime if uptime > 0 else 0
            
            return {
                'uptime_seconds': uptime,
                'total_connections': self.total_connections,
                'active_connections': self.active_connections,
                'peak_connections': self.peak_connections,
                'failed_connections': self.failed_connections,
                'total_queries': self.total_queries,
                'slow_queries': self.slow_queries,
                'queries_per_second': queries_per_second,
                'avg_query_time_ms': avg_query_time * 1000,
                'avg_connection_time_ms': avg_connection_time * 1000,
                'error_count': len(self.errors),
                'recent_errors': self.errors[-10:] if self.errors else []
            }


# Global connection statistics instance
connection_stats = ConnectionStats()


# ==================== DATABASE ENGINE FACTORY ====================

class DatabaseEngineFactory:
    """
    Factory class for creating SQLAlchemy database engines with proper configuration.
    """
    
    @staticmethod
    def create_engine(
        settings: DatabaseSettings,
        echo: bool = False,
        **kwargs
    ) -> Engine:
        """
        Create a SQLAlchemy engine based on database settings.
        
        Args:
            settings: Database configuration settings
            echo: Whether to echo SQL statements
            **kwargs: Additional engine configuration options
            
        Returns:
            Configured SQLAlchemy Engine instance
            
        Raises:
            ValueError: If database engine type is not supported
            SQLAlchemyError: If engine creation fails
        """
        try:
            connection_url = DatabaseEngineFactory._build_connection_url(settings)
            engine_kwargs = DatabaseEngineFactory._build_engine_kwargs(settings, echo, **kwargs)
            
            logger.info(f"Creating {settings.engine.value} database engine")
            logger.debug(f"Connection URL: {DatabaseEngineFactory._mask_password(connection_url)}")
            
            engine = create_engine(connection_url, **engine_kwargs)
            
            # Set up event listeners
            DatabaseEngineFactory._setup_event_listeners(engine, settings)
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            connection_stats.record_connection_failed(e)
            raise
    
    @staticmethod
    def create_async_engine(
        settings: DatabaseSettings,
        echo: bool = False,
        **kwargs
    ) -> Optional[AsyncEngine]:
        """
        Create an async SQLAlchemy engine (if async support is available).
        
        Args:
            settings: Database configuration settings
            echo: Whether to echo SQL statements
            **kwargs: Additional engine configuration options
            
        Returns:
            Configured async SQLAlchemy Engine instance or None if not supported
        """
        if not ASYNC_SUPPORT:
            logger.warning("Async SQLAlchemy support not available")
            return None
        
        try:
            connection_url = DatabaseEngineFactory._build_async_connection_url(settings)
            engine_kwargs = DatabaseEngineFactory._build_engine_kwargs(settings, echo, **kwargs)
            
            # Remove sync-only parameters for async engine
            engine_kwargs.pop('strategy', None)
            
            logger.info(f"Creating async {settings.engine.value} database engine")
            
            engine = create_async_engine(connection_url, **engine_kwargs)
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create async database engine: {e}")
            return None
    
    @staticmethod
    def _build_connection_url(settings: DatabaseSettings) -> str:
        """Build database connection URL from settings."""
        if settings.engine.value == "sqlite":
            # Handle SQLite file path
            db_path = settings.sqlite_file
            if db_path == ":memory:":
                return "sqlite:///:memory:"
            else:
                # Ensure absolute path for SQLite
                if not os.path.isabs(db_path):
                    db_path = os.path.abspath(db_path)
                return f"sqlite:///{db_path}"
        
        elif settings.engine.value == "postgresql":
            # Build PostgreSQL connection URL
            url = f"postgresql://{settings.username}"
            if settings.password:
                url += f":{settings.password}"
            url += f"@{settings.host}:{settings.port}/{settings.database}"
            
            # Add SSL parameters
            if settings.ssl_mode and settings.ssl_mode != "disable":
                url += f"?sslmode={settings.ssl_mode}"
            
            return url
        
        elif settings.engine.value == "mysql":
            # Build MySQL connection URL
            url = f"mysql+pymysql://{settings.username}"
            if settings.password:
                url += f":{settings.password}"
            url += f"@{settings.host}:{settings.port}/{settings.database}"
            
            # Add charset
            if settings.charset:
                url += f"?charset={settings.charset}"
            
            return url
        
        else:
            raise ValueError(f"Unsupported database engine: {settings.engine.value}")
    
    @staticmethod
    def _build_async_connection_url(settings: DatabaseSettings) -> str:
        """Build async database connection URL from settings."""
        base_url = DatabaseEngineFactory._build_connection_url(settings)
        
        # Convert to async drivers
        if settings.engine.value == "postgresql":
            return base_url.replace("postgresql://", "postgresql+asyncpg://")
        elif settings.engine.value == "mysql":
            return base_url.replace("mysql+pymysql://", "mysql+aiomysql://")
        elif settings.engine.value == "sqlite":
            return base_url.replace("sqlite://", "sqlite+aiosqlite://")
        
        return base_url
    
    @staticmethod
    def _build_engine_kwargs(
        settings: DatabaseSettings,
        echo: bool = False,
        **extra_kwargs
    ) -> Dict[str, Any]:
        """Build engine configuration arguments."""
        kwargs = {
            'echo': echo or settings.echo_queries,
            'pool_pre_ping': settings.pool_pre_ping,
            'connect_args': {}
        }
        
        # Configure connection pooling
        if settings.engine.value == "sqlite":
            # SQLite-specific configuration
            kwargs['poolclass'] = StaticPool
            kwargs['connect_args'].update({
                'check_same_thread': False,  # Allow multi-threading
                'timeout': settings.connect_timeout,
                'isolation_level': None  # Autocommit mode
            })
        else:
            # PostgreSQL/MySQL configuration
            kwargs.update({
                'pool_size': settings.pool_size,
                'max_overflow': settings.max_overflow,
                'pool_timeout': settings.pool_timeout,
                'pool_recycle': settings.pool_recycle,
                'poolclass': QueuePool
            })
            
            kwargs['connect_args'].update({
                'connect_timeout': settings.connect_timeout,
                'server_side_cursors': True if settings.engine.value == "postgresql" else False
            })
        
        # Add SSL configuration for PostgreSQL
        if settings.engine.value == "postgresql" and settings.ssl_cert:
            kwargs['connect_args'].update({
                'sslmode': settings.ssl_mode,
                'sslcert': settings.ssl_cert,
                'sslkey': settings.ssl_key,
                'sslrootcert': settings.ssl_ca
            })
        
        # Merge extra kwargs
        kwargs.update(extra_kwargs)
        
        return kwargs
    
    @staticmethod
    def _setup_event_listeners(engine: Engine, settings: DatabaseSettings):
        """Set up SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            connection_stats.record_connection_created()
            logger.debug(f"New database connection created: {id(dbapi_connection)}")
        
        @event.listens_for(engine, "close")
        def receive_close(dbapi_connection, connection_record):
            """Handle connection closures."""
            connection_stats.record_connection_closed()
            logger.debug(f"Database connection closed: {id(dbapi_connection)}")
        
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time."""
            context._query_start_time = time.time()
        
        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query execution time."""
            if hasattr(context, '_query_start_time'):
                duration = time.time() - context._query_start_time
                connection_stats.record_query_executed(duration, statement)
                
                if duration > settings.slow_query_threshold and settings.log_slow_queries:
                    logger.warning(f"Slow query detected ({duration:.3f}s): {statement[:100]}...")
        
        @event.listens_for(engine, "handle_error")
        def receive_handle_error(exception_context):
            """Handle database errors."""
            logger.error(f"Database error: {exception_context.original_exception}")
            connection_stats.record_connection_failed(exception_context.original_exception)
    
    @staticmethod
    def _mask_password(connection_url: str) -> str:
        """Mask password in connection URL for logging."""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', connection_url)


# ==================== SESSION MANAGEMENT ====================

class SessionManager:
    """
    Manages SQLAlchemy database sessions with proper lifecycle management.
    """
    
    def __init__(self, engine: Engine):
        """
        Initialize the session manager.
        
        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self._session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        self._scoped_session = scoped_session(self._session_factory)
        self._async_session_factory = None
        
        # Initialize async session factory if async engine is available
        if ASYNC_SUPPORT and hasattr(engine, 'sync_engine'):
            # This is an async engine
            self._async_session_factory = async_sessionmaker(
                bind=engine, expire_on_commit=False
            )
        
        logger.info("Database session manager initialized")
    
    def create_session(self) -> Session:
        """
        Create a new database session.
        
        Returns:
            New SQLAlchemy Session instance
        """
        return self._session_factory()
    
    def get_scoped_session(self) -> Session:
        """
        Get a scoped session (thread-local).
        
        Returns:
            Scoped SQLAlchemy Session instance
        """
        return self._scoped_session()
    
    def remove_scoped_session(self):
        """Remove the current scoped session."""
        self._scoped_session.remove()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope around a series of operations.
        
        Yields:
            Database session with automatic transaction management
            
        Example:
            with session_manager.session_scope() as session:
                user = User(name="John Doe")
                session.add(user)
                # Transaction is automatically committed or rolled back
        """
        session = self.create_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session rolled back due to error: {e}")
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def async_session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide an async transactional scope around a series of operations.
        
        Yields:
            Async database session with automatic transaction management
        """
        if not self._async_session_factory:
            raise RuntimeError("Async session not available")
        
        async_session = self._async_session_factory()
        try:
            yield async_session
            await async_session.commit()
        except Exception as e:
            await async_session.rollback()
            logger.error(f"Async session rolled back due to error: {e}")
            raise
        finally:
            await async_session.close()
    
    def health_check(self) -> bool:
        """
        Perform a basic health check on the database connection.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def async_health_check(self) -> bool:
        """
        Perform an async health check on the database connection.
        
        Returns:
            True if database is accessible, False otherwise
        """
        if not self._async_session_factory:
            return False
        
        try:
            async with self.async_session_scope() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False


# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """
    Main database manager that coordinates engine creation, session management,
    and database operations.
    """
    
    def __init__(self, settings: DatabaseSettings):
        """
        Initialize the database manager.
        
        Args:
            settings: Database configuration settings
        """
        self.settings = settings
        self.engine: Optional[Engine] = None
        self.async_engine: Optional[AsyncEngine] = None
        self.session_manager: Optional[SessionManager] = None
        self._initialized = False
        
        # Set up logging
        setup_database_logging(
            log_level=LogLevel.DEBUG if settings.echo_queries else LogLevel.INFO,
            echo_queries=settings.echo_queries
        )
    
    def initialize(self) -> None:
        """
        Initialize the database connection and session manager.
        
        Raises:
            DatabaseError: If initialization fails
        """
        try:
            if self._initialized:
                logger.warning("Database manager already initialized")
                return
            
            # Create database engine
            self.engine = DatabaseEngineFactory.create_engine(
                self.settings,
                echo=self.settings.echo_queries
            )
            
            # Create async engine if supported
            if ASYNC_SUPPORT:
                self.async_engine = DatabaseEngineFactory.create_async_engine(
                    self.settings,
                    echo=self.settings.echo_queries
                )
            
            # Create session manager
            self.session_manager = SessionManager(self.engine)
            
            # Verify connection
            if not self.session_manager.health_check():
                raise RuntimeError("Initial database health check failed")
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if self.session_manager:
                self.session_manager.remove_scoped_session()
            
            if self.engine:
                self.engine.dispose()
                logger.info("Database engine disposed")
            
            if self.async_engine:
                # Note: async engine disposal should be done in async context
                logger.info("Async database engine marked for disposal")
            
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            New SQLAlchemy Session instance
            
        Raises:
            RuntimeError: If database manager is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        return self.session_manager.create_session()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """
        Get a database session with context management.
        
        Yields:
            Database session with automatic cleanup
        """
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        with self.session_manager.session_scope() as session:
            yield session
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get current database connection statistics.
        
        Returns:
            Dictionary containing connection statistics and performance metrics
        """
        stats = connection_stats.get_stats()
        
        # Add engine-specific information
        if self.engine:
            pool = self.engine.pool
            stats.update({
                'pool_size': getattr(pool, 'size', lambda: 0)(),
                'checked_in': getattr(pool, 'checkedin', lambda: 0)(),
                'checked_out': getattr(pool, 'checkedout', lambda: 0)(),
                'overflow': getattr(pool, 'overflow', lambda: 0)(),
                'invalid': getattr(pool, 'invalid', lambda: 0)()
            })
        
        # Add database-specific information
        stats.update({
            'database_engine': self.settings.engine.value,
            'database_name': self.settings.database,
            'pool_timeout': self.settings.pool_timeout,
            'initialized': self._initialized
        })
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# ==================== DATABASE INITIALIZATION ====================

def init_database(settings: DatabaseSettings, create_tables: bool = True) -> DatabaseManager:
    """
    Initialize database with proper configuration and optionally create tables.
    
    Args:
        settings: Database configuration settings
        create_tables: Whether to create database tables
        
    Returns:
        Initialized DatabaseManager instance
        
    Raises:
        DatabaseError: If initialization fails
    """
    try:
        logger.info("Initializing database...")
        
        # Create database manager
        db_manager = DatabaseManager(settings)
        db_manager.initialize()
        
        # Create database directory for SQLite
        if settings.engine.value == "sqlite" and settings.sqlite_file != ":memory:":
            db_path = Path(settings.sqlite_file)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables if requested
        if create_tables:
            create_database_tables(db_manager.engine)
        
        logger.info("Database initialization completed successfully")
        return db_manager
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise DatabaseError(f"Failed to initialize database: {e}")


def create_database_tables(engine: Engine) -> None:
    """
    Create all database tables defined in the metadata.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Raises:
        DatabaseError: If table creation fails
    """
    try:
        logger.info("Creating database tables...")
        
        # Create all tables
        metadata.create_all(engine)
        
        # Verify tables were created
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        logger.info(f"Created {len(table_names)} database tables: {', '.join(table_names)}")
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise DatabaseError(f"Table creation failed: {e}")


def drop_database_tables(engine: Engine) -> None:
    """
    Drop all database tables defined in the metadata.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Raises:
        DatabaseError: If table dropping fails
    """
    try:
        logger.warning("Dropping all database tables...")
        
        metadata.drop_all(engine)
        
        logger.warning("All database tables dropped")
        
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise DatabaseError(f"Table dropping failed: {e}")


# ==================== DATABASE MIGRATIONS ====================

class MigrationManager:
    """
    Manages database schema migrations using Alembic.
    """
    
    def __init__(self, db_manager: DatabaseManager, alembic_cfg_path: str = "alembic.ini"):
        """
        Initialize the migration manager.
        
        Args:
            db_manager: Database manager instance
            alembic_cfg_path: Path to Alembic configuration file
        """
        self.db_manager = db_manager
        self.alembic_cfg_path = alembic_cfg_path
        self.alembic_cfg = None
        
        if ALEMBIC_AVAILABLE:
            try:
                self.alembic_cfg = AlembicConfig(alembic_cfg_path)
                # Set SQLAlchemy URL in Alembic config
                connection_url = DatabaseEngineFactory._build_connection_url(db_manager.settings)
                self.alembic_cfg.set_main_option("sqlalchemy.url", connection_url)
            except Exception as e:
                logger.error(f"Failed to load Alembic configuration: {e}")
                self.alembic_cfg = None
    
    def run_migrations(self, revision: str = "head") -> bool:
        """
        Run database migrations to the specified revision.
        
        Args:
            revision: Target revision (default: "head" for latest)
            
        Returns:
            True if migrations ran successfully, False otherwise
        """
        if not ALEMBIC_AVAILABLE or not self.alembic_cfg:
            logger.error("Alembic not available or configuration not loaded")
            return False
        
        try:
            logger.info(f"Running database migrations to revision: {revision}")
            
            with self.db_manager.get_session_context() as session:
                connection = session.connection()
                
                # Run migrations
                alembic_command.upgrade(self.alembic_cfg, revision)
            
            logger.info("Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def create_migration(self, message: str, autogenerate: bool = True) -> bool:
        """
        Create a new migration revision.
        
        Args:
            message: Migration message/description
            autogenerate: Whether to auto-generate migration from model changes
            
        Returns:
            True if migration was created successfully, False otherwise
        """
        if not ALEMBIC_AVAILABLE or not self.alembic_cfg:
            logger.error("Alembic not available or configuration not loaded")
            return False
        
        try:
            logger.info(f"Creating new migration: {message}")
            
            alembic_command.revision(
                self.alembic_cfg,
                message=message,
                autogenerate=autogenerate
            )
            
            logger.info("Migration created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            return False
    
    def get_current_revision(self) -> Optional[str]:
        """
        Get the current database revision.
        
        Returns:
            Current revision string or None if unable to determine
        """
        if not self.alembic_cfg:
            return None
        
        try:
            with self.db_manager.get_session_context() as session:
                connection = session.connection()
                
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
                
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """
        Get the migration history.
        
        Returns:
            List of migration information dictionaries
        """
        if not ALEMBIC_AVAILABLE or not self.alembic_cfg:
            return []
        
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            history = []
            
            for revision in script_dir.walk_revisions():
                history.append({
                    'revision': revision.revision,
                    'down_revision': revision.down_revision,
                    'message': revision.doc,
                    'branch_labels': revision.branch_labels,
                    'depends_on': revision.depends_on
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []


def run_migrations(db_manager: DatabaseManager, alembic_cfg_path: str = "alembic.ini") -> bool:
    """
    Run database migrations using Alembic.
    
    Args:
        db_manager: Database manager instance
        alembic_cfg_path: Path to Alembic configuration file
        
    Returns:
        True if migrations ran successfully, False otherwise
    """
    migration_manager = MigrationManager(db_manager, alembic_cfg_path)
    return migration_manager.run_migrations()


# ==================== CUSTOM EXCEPTIONS ====================

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass


class TransactionError(DatabaseError):
    """Raised when database transaction fails."""
    pass


# ==================== UTILITY FUNCTIONS ====================

def check_database_connectivity(settings: DatabaseSettings) -> bool:
    """
    Check if database is accessible with the given settings.
    
    Args:
        settings: Database configuration settings
        
    Returns:
        True if database is accessible, False otherwise
    """
    try:
        # Create a temporary engine for testing
        engine = DatabaseEngineFactory.create_engine(settings)
        
        # Test connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        return False


def get_database_info(engine: Engine) -> Dict[str, Any]:
    """
    Get information about the database.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        Dictionary containing database information
    """
    try:
        inspector = inspect(engine)
        
        return {
            'engine_name': engine.name,
            'driver': engine.driver,
            'server_version': getattr(engine.dialect, 'server_version_info', 'Unknown'),
            'table_count': len(inspector.get_table_names()),
            'table_names': inspector.get_table_names(),
            'schema_names': inspector.get_schema_names() if hasattr(inspector, 'get_schema_names') else [],
            'connection_url': DatabaseEngineFactory._mask_password(str(engine.url))
        }
        
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {'error': str(e)}


def backup_database(engine: Engine, backup_path: str) -> bool:
    """
    Create a database backup (SQLite only for now).
    
    Args:
        engine: SQLAlchemy engine instance
        backup_path: Path where backup should be saved
        
    Returns:
        True if backup was successful, False otherwise
    """
    if engine.name != 'sqlite':
        logger.error("Database backup currently only supported for SQLite")
        return False
    
    try:
        import sqlite3
        import shutil
        
        # Get database file path from engine URL
        db_path = str(engine.url).replace('sqlite:///', '')
        
        if db_path == ':memory:':
            logger.error("Cannot backup in-memory SQLite database")
            return False
        
        # Create backup directory if it doesn't exist
        backup_dir = Path(backup_path).parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create backup
        shutil.copy2(db_path, backup_path)
        
        logger.info(f"Database backup created: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """
    Example usage of the database module.
    """
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example database settings
    from .config import DatabaseSettings, DatabaseEngine
    
    settings = DatabaseSettings(
        engine=DatabaseEngine.SQLITE,
        sqlite_file="example.db",
        pool_size=5,
        echo_queries=True
    )
    
    # Initialize database
    try:
        with init_database(settings) as db_manager:
            # Get connection stats
            stats = db_manager.get_connection_stats()
            print("Connection Stats:", stats)
            
            # Test session management
            with db_manager.get_session_context() as session:
                result = session.execute(text("SELECT 1 as test"))
                print("Query Result:", result.fetchone())
            
            # Check database info
            info = get_database_info(db_manager.engine)
            print("Database Info:", info)
            
            # Health check
            is_healthy = db_manager.session_manager.health_check()
            print("Database Health:", is_healthy)
        
        print("Database operations completed successfully")
        
    except Exception as e:
        print(f"Database operations failed: {e}")


# ==================== ADVANCED SESSION MANAGEMENT ====================

class SessionPool:
    """
    Advanced session pool for managing multiple database sessions with lifecycle tracking.
    """
    
    def __init__(self, session_factory: sessionmaker, max_sessions: int = 20):
        """
        Initialize session pool.
        
        Args:
            session_factory: SQLAlchemy session factory
            max_sessions: Maximum number of concurrent sessions
        """
        self.session_factory = session_factory
        self.max_sessions = max_sessions
        self.active_sessions: Dict[str, Session] = {}
        self.session_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def acquire_session(self, session_id: Optional[str] = None) -> Tuple[str, Session]:
        """
        Acquire a session from the pool.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Tuple of (session_id, session)
            
        Raises:
            RuntimeError: If maximum sessions exceeded
        """
        with self._lock:
            if len(self.active_sessions) >= self.max_sessions:
                raise RuntimeError(f"Maximum sessions ({self.max_sessions}) exceeded")
            
            if session_id is None:
                session_id = f"session_{int(time.time() * 1000000)}"
            
            if session_id in self.active_sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            session = self.session_factory()
            self.active_sessions[session_id] = session
            self.session_stats[session_id] = {
                'created_at': time.time(),
                'queries_executed': 0,
                'last_activity': time.time(),
                'errors': 0
            }
            
            logger.debug(f"Acquired session: {session_id}")
            return session_id, session
    
    def release_session(self, session_id: str) -> bool:
        """
        Release a session back to the pool.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was released successfully
        """
        with self._lock:
            if session_id not in self.active_sessions:
                logger.warning(f"Attempted to release unknown session: {session_id}")
                return False
            
            session = self.active_sessions[session_id]
            try:
                session.close()
                del self.active_sessions[session_id]
                del self.session_stats[session_id]
                logger.debug(f"Released session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error releasing session {session_id}: {e}")
                return False
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get an active session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session instance or None if not found
        """
        with self._lock:
            return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """
        Update session activity timestamp.
        
        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id in self.session_stats:
                self.session_stats[session_id]['last_activity'] = time.time()
                self.session_stats[session_id]['queries_executed'] += 1
    
    def cleanup_idle_sessions(self, idle_timeout: int = 1800) -> int:
        """
        Clean up idle sessions that haven't been active for the specified timeout.
        
        Args:
            idle_timeout: Idle timeout in seconds (default: 30 minutes)
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        idle_sessions = []
        
        with self._lock:
            for session_id, stats in self.session_stats.items():
                if current_time - stats['last_activity'] > idle_timeout:
                    idle_sessions.append(session_id)
        
        cleaned_count = 0
        for session_id in idle_sessions:
            if self.release_session(session_id):
                cleaned_count += 1
                logger.info(f"Cleaned up idle session: {session_id}")
        
        return cleaned_count
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get session pool statistics.
        
        Returns:
            Dictionary containing pool statistics
        """
        with self._lock:
            current_time = time.time()
            
            return {
                'active_sessions': len(self.active_sessions),
                'max_sessions': self.max_sessions,
                'session_ids': list(self.active_sessions.keys()),
                'total_queries': sum(stats['queries_executed'] for stats in self.session_stats.values()),
                'avg_session_age': (
                    sum(current_time - stats['created_at'] for stats in self.session_stats.values()) / 
                    len(self.session_stats) if self.session_stats else 0
                ),
                'oldest_session': (
                    current_time - min(stats['created_at'] for stats in self.session_stats.values())
                    if self.session_stats else 0
                )
            }


# ==================== TRANSACTION MANAGEMENT ====================

class TransactionManager:
    """
    Advanced transaction management with savepoints and nested transactions.
    """
    
    def __init__(self, session: Session):
        """
        Initialize transaction manager.
        
        Args:
            session: SQLAlchemy session instance
        """
        self.session = session
        self.transaction_stack: List[Dict[str, Any]] = []
        self.savepoint_counter = 0
    
    @contextmanager
    def transaction(self, savepoint_name: Optional[str] = None) -> Generator[None, None, None]:
        """
        Context manager for handling transactions with automatic rollback.
        
        Args:
            savepoint_name: Optional savepoint name for nested transactions
        """
        transaction_id = f"tx_{len(self.transaction_stack)}_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        # Determine if this is a nested transaction
        is_nested = len(self.transaction_stack) > 0
        
        if is_nested:
            # Create a savepoint for nested transaction
            if savepoint_name is None:
                self.savepoint_counter += 1
                savepoint_name = f"sp_{self.savepoint_counter}"
            
            savepoint = self.session.begin_nested()
            transaction_info = {
                'id': transaction_id,
                'type': 'savepoint',
                'savepoint_name': savepoint_name,
                'savepoint': savepoint,
                'start_time': start_time
            }
        else:
            # Start a new transaction
            transaction = self.session.begin()
            transaction_info = {
                'id': transaction_id,
                'type': 'transaction',
                'transaction': transaction,
                'start_time': start_time
            }
        
        self.transaction_stack.append(transaction_info)
        
        try:
            logger.debug(f"Started {transaction_info['type']}: {transaction_id}")
            yield
            
            # Commit the transaction/savepoint
            if is_nested:
                savepoint.commit()
            else:
                transaction.commit()
            
            duration = time.time() - start_time
            logger.debug(f"Committed {transaction_info['type']}: {transaction_id} ({duration:.3f}s)")
            
        except Exception as e:
            # Rollback the transaction/savepoint
            try:
                if is_nested:
                    savepoint.rollback()
                else:
                    transaction.rollback()
                
                duration = time.time() - start_time
                logger.warning(f"Rolled back {transaction_info['type']}: {transaction_id} ({duration:.3f}s) - {e}")
            except Exception as rollback_error:
                logger.error(f"Error during rollback of {transaction_id}: {rollback_error}")
            
            raise
        
        finally:
            self.transaction_stack.pop()
    
    def get_transaction_info(self) -> List[Dict[str, Any]]:
        """
        Get information about active transactions.
        
        Returns:
            List of active transaction information
        """
        current_time = time.time()
        return [
            {
                'id': tx['id'],
                'type': tx['type'],
                'duration': current_time - tx['start_time'],
                'savepoint_name': tx.get('savepoint_name')
            }
            for tx in self.transaction_stack
        ]


# ==================== QUERY PERFORMANCE MONITORING ====================

class QueryMonitor:
    """
    Monitors and analyzes database query performance.
    """
    
    def __init__(self, max_queries: int = 1000):
        """
        Initialize query monitor.
        
        Args:
            max_queries: Maximum number of queries to keep in history
        """
        self.max_queries = max_queries
        self.query_history: List[Dict[str, Any]] = []
        self.slow_queries: List[Dict[str, Any]] = []
        self.query_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def record_query(
        self, 
        query: str, 
        parameters: Optional[Dict] = None,
        duration: float = 0.0,
        error: Optional[Exception] = None
    ):
        """
        Record a query execution for monitoring.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            duration: Execution time in seconds
            error: Exception if query failed
        """
        with self._lock:
            # Normalize query for statistics
            normalized_query = self._normalize_query(query)
            
            query_record = {
                'timestamp': time.time(),
                'query': query[:500],  # Truncate long queries
                'normalized_query': normalized_query,
                'parameters': parameters,
                'duration': duration,
                'error': str(error) if error else None,
                'success': error is None
            }
            
            # Add to history
            self.query_history.append(query_record)
            if len(self.query_history) > self.max_queries:
                self.query_history.pop(0)
            
            # Track slow queries (>1 second)
            if duration > 1.0:
                self.slow_queries.append(query_record)
                if len(self.slow_queries) > 100:  # Keep only last 100 slow queries
                    self.slow_queries.pop(0)
            
            # Update statistics
            if normalized_query not in self.query_stats:
                self.query_stats[normalized_query] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'errors': 0,
                    'first_seen': time.time(),
                    'last_seen': time.time()
                }
            
            stats = self.query_stats[normalized_query]
            stats['count'] += 1
            stats['total_duration'] += duration
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['min_duration'] = min(stats['min_duration'], duration)
            stats['max_duration'] = max(stats['max_duration'], duration)
            stats['last_seen'] = time.time()
            
            if error:
                stats['errors'] += 1
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query by removing parameters and formatting.
        
        Args:
            query: Original SQL query
            
        Returns:
            Normalized query string
        """
        import re
        
        # Convert to uppercase and remove extra whitespace
        normalized = ' '.join(query.upper().split())
        
        # Replace parameter placeholders
        normalized = re.sub(r'\?', '?', normalized)
        normalized = re.sub(r'%\([^)]+\)s', '?', normalized)
        normalized = re.sub(r':\w+', '?', normalized)
        
        # Replace quoted strings and numbers with placeholders
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        return normalized
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report based on collected query data.
        
        Returns:
            Dictionary containing performance metrics and analysis
        """
        with self._lock:
            if not self.query_history:
                return {'status': 'No query data available'}
            
            # Calculate overall statistics
            total_queries = len(self.query_history)
            successful_queries = sum(1 for q in self.query_history if q['success'])
            failed_queries = total_queries - successful_queries
            
            durations = [q['duration'] for q in self.query_history if q['success']]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Find slowest queries
            slowest_queries = sorted(
                [q for q in self.query_history if q['success']],
                key=lambda x: x['duration'],
                reverse=True
            )[:10]
            
            # Most frequent queries
            frequent_queries = sorted(
                self.query_stats.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
            
            return {
                'summary': {
                    'total_queries': total_queries,
                    'successful_queries': successful_queries,
                    'failed_queries': failed_queries,
                    'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                    'avg_duration_ms': avg_duration * 1000,
                    'slow_queries_count': len(self.slow_queries)
                },
                'slowest_queries': [
                    {
                        'query': q['query'],
                        'duration_ms': q['duration'] * 1000,
                        'timestamp': q['timestamp']
                    }
                    for q in slowest_queries
                ],
                'most_frequent_queries': [
                    {
                        'query': query,
                        'count': stats['count'],
                        'avg_duration_ms': stats['avg_duration'] * 1000,
                        'total_duration_ms': stats['total_duration'] * 1000
                    }
                    for query, stats in frequent_queries
                ],
                'recent_errors': [
                    {
                        'query': q['query'],
                        'error': q['error'],
                        'timestamp': q['timestamp']
                    }
                    for q in self.query_history if not q['success']
                ][-10:]  # Last 10 errors
            }
    
    def reset_statistics(self):
        """Reset all collected statistics."""
        with self._lock:
            self.query_history.clear()
            self.slow_queries.clear()
            self.query_stats.clear()


# ==================== DATABASE SCHEMA MANAGEMENT ====================

class SchemaManager:
    """
    Manages database schema operations including introspection and validation.
    """
    
    def __init__(self, engine: Engine):
        """
        Initialize schema manager.
        
        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self.inspector = inspect(engine)
    
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing table information or None if table doesn't exist
        """
        try:
            if not self.inspector.has_table(table_name):
                return None
            
            columns = self.inspector.get_columns(table_name)
            primary_keys = self.inspector.get_pk_constraint(table_name)
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            indexes = self.inspector.get_indexes(table_name)
            unique_constraints = self.inspector.get_unique_constraints(table_name)
            
            return {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col.get('default'),
                        'autoincrement': col.get('autoincrement', False),
                        'primary_key': col['name'] in primary_keys.get('constrained_columns', [])
                    }
                    for col in columns
                ],
                'primary_keys': primary_keys.get('constrained_columns', []),
                'foreign_keys': [
                    {
                        'name': fk.get('name'),
                        'constrained_columns': fk['constrained_columns'],
                        'referred_table': fk['referred_table'],
                        'referred_columns': fk['referred_columns']
                    }
                    for fk in foreign_keys
                ],
                'indexes': [
                    {
                        'name': idx['name'],
                        'column_names': idx['column_names'],
                        'unique': idx['unique']
                    }
                    for idx in indexes
                ],
                'unique_constraints': unique_constraints
            }
            
        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return None
    
    def get_all_tables_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all tables in the database.
        
        Returns:
            Dictionary mapping table names to their information
        """
        tables_info = {}
        
        for table_name in self.inspector.get_table_names():
            table_info = self.get_table_info(table_name)
            if table_info:
                tables_info[table_name] = table_info
        
        return tables_info
    
    def validate_schema(self, expected_tables: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Validate the current database schema against expected schema.
        
        Args:
            expected_tables: Expected table definitions
            
        Returns:
            Validation report with any discrepancies found
        """
        validation_report = {
            'valid': True,
            'missing_tables': [],
            'extra_tables': [],
            'table_issues': {}
        }
        
        current_tables = set(self.inspector.get_table_names())
        expected_table_names = set(expected_tables.keys())
        
        # Check for missing and extra tables
        validation_report['missing_tables'] = list(expected_table_names - current_tables)
        validation_report['extra_tables'] = list(current_tables - expected_table_names)
        
        if validation_report['missing_tables'] or validation_report['extra_tables']:
            validation_report['valid'] = False
        
        # Validate existing tables
        for table_name in expected_table_names & current_tables:
            table_issues = self._validate_table_structure(table_name, expected_tables[table_name])
            if table_issues:
                validation_report['table_issues'][table_name] = table_issues
                validation_report['valid'] = False
        
        return validation_report
    
    def _validate_table_structure(self, table_name: str, expected_structure: Dict) -> List[str]:
        """
        Validate a single table's structure against expected structure.
        
        Args:
            table_name: Name of the table to validate
            expected_structure: Expected table structure
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        try:
            table_info = self.get_table_info(table_name)
            if not table_info:
                issues.append(f"Table {table_name} not found")
                return issues
            
            # Check columns
            current_columns = {col['name']: col for col in table_info['columns']}
            expected_columns = expected_structure.get('columns', {})
            
            for col_name, expected_col in expected_columns.items():
                if col_name not in current_columns:
                    issues.append(f"Missing column: {col_name}")
                else:
                    current_col = current_columns[col_name]
                    
                    # Check column type (simplified check)
                    if 'type' in expected_col:
                        expected_type = str(expected_col['type']).upper()
                        current_type = current_col['type'].upper()
                        if expected_type not in current_type and current_type not in expected_type:
                            issues.append(f"Column {col_name}: type mismatch (expected {expected_type}, got {current_type})")
                    
                    # Check nullable
                    if 'nullable' in expected_col:
                        if expected_col['nullable'] != current_col['nullable']:
                            issues.append(f"Column {col_name}: nullable mismatch")
            
            # Check for extra columns
            extra_columns = set(current_columns.keys()) - set(expected_columns.keys())
            for col_name in extra_columns:
                issues.append(f"Extra column: {col_name}")
            
        except Exception as e:
            issues.append(f"Error validating table structure: {e}")
        
        return issues
    
    def generate_ddl(self, table_name: str) -> Optional[str]:
        """
        Generate DDL (CREATE TABLE) statement for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DDL statement string or None if table doesn't exist
        """
        try:
            if not self.inspector.has_table(table_name):
                return None
            
            # This is a simplified DDL generation
            # In practice, you might want to use SQLAlchemy's DDL compilation
            table_info = self.get_table_info(table_name)
            if not table_info:
                return None
            
            ddl_parts = [f"CREATE TABLE {table_name} ("]
            
            column_definitions = []
            for col in table_info['columns']:
                col_def = f"  {col['name']} {col['type']}"
                
                if not col['nullable']:
                    col_def += " NOT NULL"
                
                if col['default'] is not None:
                    col_def += f" DEFAULT {col['default']}"
                
                if col['primary_key']:
                    col_def += " PRIMARY KEY"
                
                column_definitions.append(col_def)
            
            ddl_parts.append(",\n".join(column_definitions))
            ddl_parts.append(");")
            
            return "\n".join(ddl_parts)
            
        except Exception as e:
            logger.error(f"Error generating DDL for {table_name}: {e}")
            return None


# ==================== DATABASE MAINTENANCE ====================

class MaintenanceManager:
    """
    Handles database maintenance operations like optimization, cleanup, and health checks.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize maintenance manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.schema_manager = SchemaManager(db_manager.engine)
        self.query_monitor = QueryMonitor()
    
    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check.
        
        Returns:
            Dictionary containing health check results
        """
        health_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        try:
            # Basic connectivity check
            health_report['checks']['connectivity'] = {
                'status': 'healthy' if self.db_manager.session_manager.health_check() else 'unhealthy',
                'message': 'Database connectivity test'
            }
            
            # Connection pool status
            pool_stats = self.db_manager.get_connection_stats()
            pool_utilization = (pool_stats.get('checked_out', 0) / 
                              max(pool_stats.get('pool_size', 1), 1)) * 100
            
            health_report['checks']['connection_pool'] = {
                'status': 'healthy' if pool_utilization < 80 else 'warning' if pool_utilization < 95 else 'critical',
                'utilization_percent': pool_utilization,
                'active_connections': pool_stats.get('checked_out', 0),
                'pool_size': pool_stats.get('pool_size', 0)
            }
            
            # Table existence check
            expected_tables = ['users', 'sessions', 'audit_logs']  # Add your expected tables
            missing_tables = []
            
            for table in expected_tables:
                if not self.schema_manager.inspector.has_table(table):
                    missing_tables.append(table)
            
            health_report['checks']['schema'] = {
                'status': 'healthy' if not missing_tables else 'warning',
                'missing_tables': missing_tables,
                'total_tables': len(self.schema_manager.inspector.get_table_names())
            }
            
            # Performance check (query response time)
            start_time = time.time()
            with self.db_manager.get_session_context() as session:
                session.execute(text("SELECT 1"))
            response_time = (time.time() - start_time) * 1000
            
            health_report['checks']['performance'] = {
                'status': 'healthy' if response_time < 100 else 'warning' if response_time < 500 else 'critical',
                'response_time_ms': response_time
            }
            
            # Determine overall status
            statuses = [check['status'] for check in health_report['checks'].values()]
            if 'critical' in statuses:
                health_report['overall_status'] = 'critical'
            elif 'warning' in statuses:
                health_report['overall_status'] = 'warning'
            elif 'unhealthy' in statuses:
                health_report['overall_status'] = 'unhealthy'
            
        except Exception as e:
            health_report['overall_status'] = 'error'
            health_report['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_report
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Perform database optimization operations.
        
        Returns:
            Dictionary containing optimization results
        """
        optimization_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operations_performed': [],
            'errors': []
        }
        
        try:
            engine_name = self.db_manager.engine.name
            
            if engine_name == 'sqlite':
                self._optimize_sqlite(optimization_report)
            elif engine_name == 'postgresql':
                self._optimize_postgresql(optimization_report)
            elif engine_name == 'mysql':
                self._optimize_mysql(optimization_report)
            
        except Exception as e:
            optimization_report['errors'].append(f"Optimization failed: {e}")
            logger.error(f"Database optimization failed: {e}")
        
        return optimization_report
    
    def _optimize_sqlite(self, report: Dict[str, Any]):
        """Perform SQLite-specific optimization."""
        with self.db_manager.get_session_context() as session:
            # VACUUM to reclaim space and defragment
            try:
                session.execute(text("VACUUM"))
                report['operations_performed'].append("VACUUM completed")
            except Exception as e:
                report['errors'].append(f"VACUUM failed: {e}")
            
            # ANALYZE to update statistics
            try:
                session.execute(text("ANALYZE"))
                report['operations_performed'].append("ANALYZE completed")
            except Exception as e:
                report['errors'].append(f"ANALYZE failed: {e}")
            
            # Enable WAL mode for better concurrency
            try:
                result = session.execute(text("PRAGMA journal_mode=WAL"))
                mode = result.fetchone()[0]
                report['operations_performed'].append(f"Journal mode set to: {mode}")
            except Exception as e:
                report['errors'].append(f"WAL mode setup failed: {e}")
    
    def _optimize_postgresql(self, report: Dict[str, Any]):
        """Perform PostgreSQL-specific optimization."""
        with self.db_manager.get_session_context() as session:
            # Update table statistics
            try:
                session.execute(text("ANALYZE"))
                report['operations_performed'].append("ANALYZE completed")
            except Exception as e:
                report['errors'].append(f"ANALYZE failed: {e}")
            
            # Get table names for optimization
            try:
                tables = self.schema_manager.inspector.get_table_names()
                for table in tables:
                    try:
                        session.execute(text(f"VACUUM ANALYZE {table}"))
                        report['operations_performed'].append(f"VACUUM ANALYZE completed for table: {table}")
                    except Exception as e:
                        report['errors'].append(f"VACUUM ANALYZE failed for table {table}: {e}")
            except Exception as e:
                report['errors'].append(f"Table optimization failed: {e}")
    
    def _optimize_mysql(self, report: Dict[str, Any]):
        """Perform MySQL-specific optimization."""
        with self.db_manager.get_session_context() as session:
            try:
                tables = self.schema_manager.inspector.get_table_names()
                for table in tables:
                    try:
                        # Optimize table
                        session.execute(text(f"OPTIMIZE TABLE {table}"))
                        report['operations_performed'].append(f"OPTIMIZE TABLE completed for: {table}")
                        
                        # Analyze table
                        session.execute(text(f"ANALYZE TABLE {table}"))
                        report['operations_performed'].append(f"ANALYZE TABLE completed for: {table}")
                        
                    except Exception as e:
                        report['errors'].append(f"Table optimization failed for {table}: {e}")
            except Exception as e:
                report['errors'].append(f"MySQL optimization failed: {e}")
    
    def cleanup_old_data(self, cleanup_config: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Clean up old data based on configuration.
        
        Args:
            cleanup_config: Configuration for data cleanup
            
        Returns:
            Dictionary containing cleanup results
        """
        cleanup_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tables_processed': 0,
            'records_deleted': 0,
            'errors': []
        }
        
        try:
            with self.db_manager.get_session_context() as session:
                for table_name, config in cleanup_config.items():
                    try:
                        if not self.schema_manager.inspector.has_table(table_name):
                            cleanup_report['errors'].append(f"Table {table_name} not found")
                            continue
                        
                        # Build cleanup query based on configuration
                        date_column = config.get('date_column', 'created_at')
                        retention_days = config.get('retention_days', 90)
                        conditions = config.get('conditions', [])
                        
                        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
                        
                        # Basic cleanup query
                        query = f"DELETE FROM {table_name} WHERE {date_column} < :cutoff_date"
                        
                        # Add additional conditions
                        for condition in conditions:
                            query += f" AND {condition}"
                        
                        result = session.execute(text(query), {'cutoff_date': cutoff_date})
                        deleted_count = result.rowcount
                        
                        cleanup_report['tables_processed'] += 1
                        cleanup_report['records_deleted'] += deleted_count
                        
                        logger.info(f"Cleaned up {deleted_count} records from {table_name}")
                        
                    except Exception as e:
                        cleanup_report['errors'].append(f"Cleanup failed for table {table_name}: {e}")
                        logger.error(f"Cleanup failed for table {table_name}: {e}")
        
        except Exception as e:
            cleanup_report['errors'].append(f"Cleanup operation failed: {e}")
            logger.error(f"Data cleanup failed: {e}")
        
        return cleanup_report


# ==================== DATABASE BACKUP AND RESTORE ====================

class BackupManager:
    """
    Handles database backup and restore operations.
    """
    
    def __init__(self, db_manager: DatabaseManager, backup_dir: str = "backups"):
        """
        Initialize backup manager.
        
        Args:
            db_manager: Database manager instance
            backup_dir: Directory for storing backups
        """
        self.db_manager = db_manager
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a database backup.
        
        Args:
            backup_name: Optional backup name (auto-generated if not provided)
            
        Returns:
            Dictionary containing backup information
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_info = {
            'backup_name': backup_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'database_engine': self.db_manager.settings.engine.value,
            'status': 'failed',
            'file_path': None,
            'file_size': 0,
            'error': None
        }
        
        try:
            engine_name = self.db_manager.engine.name
            
            if engine_name == 'sqlite':
                backup_info = self._backup_sqlite(backup_name, backup_info)
            elif engine_name == 'postgresql':
                backup_info = self._backup_postgresql(backup_name, backup_info)
            elif engine_name == 'mysql':
                backup_info = self._backup_mysql(backup_name, backup_info)
            else:
                backup_info['error'] = f"Backup not supported for engine: {engine_name}"
                
        except Exception as e:
            backup_info['error'] = str(e)
            logger.error(f"Backup creation failed: {e}")
        
        return backup_info
    
    def _backup_sqlite(self, backup_name: str, backup_info: Dict) -> Dict:
        """Create SQLite database backup."""
        import shutil
        
        # Get source database file
        db_url = str(self.db_manager.engine.url)
        source_path = db_url.replace('sqlite:///', '')
        
        if source_path == ':memory:':
            backup_info['error'] = "Cannot backup in-memory database"
            return backup_info
        
        # Create backup file path
        backup_file = self.backup_dir / f"{backup_name}.db"
        
        # Copy database file
        shutil.copy2(source_path, backup_file)
        
        backup_info.update({
            'status': 'success',
            'file_path': str(backup_file),
            'file_size': backup_file.stat().st_size
        })
        
        logger.info(f"SQLite backup created: {backup_file}")
        return backup_info
    
    def _backup_postgresql(self, backup_name: str, backup_info: Dict) -> Dict:
        """Create PostgreSQL database backup using pg_dump."""
        import subprocess
        
        settings = self.db_manager.settings
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        # Build pg_dump command
        cmd = [
            'pg_dump',
            '-h', settings.host,
            '-p', str(settings.port),
            '-U', settings.username,
            '-d', settings.database,
            '-f', str(backup_file),
            '--no-password'  # Use .pgpass or environment variables for password
        ]
        
        # Set environment variables
        env = os.environ.copy()
        if settings.password:
            env['PGPASSWORD'] = settings.password
        
        # Execute pg_dump
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            backup_info.update({
                'status': 'success',
                'file_path': str(backup_file),
                'file_size': backup_file.stat().st_size
            })
            logger.info(f"PostgreSQL backup created: {backup_file}")
        else:
            backup_info['error'] = f"pg_dump failed: {result.stderr}"
        
        return backup_info
    
    def _backup_mysql(self, backup_name: str, backup_info: Dict) -> Dict:
        """Create MySQL database backup using mysqldump."""
        import subprocess
        
        settings = self.db_manager.settings
        backup_file = self.backup_dir / f"{backup_name}.sql"
        
        # Build mysqldump command
        cmd = [
            'mysqldump',
            f'--host={settings.host}',
            f'--port={settings.port}',
            f'--user={settings.username}',
            '--single-transaction',
            '--routines',
            '--triggers',
            settings.database
        ]
        
        if settings.password:
            cmd.append(f'--password={settings.password}')
        
        # Execute mysqldump
        with open(backup_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            backup_info.update({
                'status': 'success',
                'file_path': str(backup_file),
                'file_size': backup_file.stat().st_size
            })
            logger.info(f"MySQL backup created: {backup_file}")
        else:
            backup_info['error'] = f"mysqldump failed: {result.stderr}"
        
        return backup_info
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*"):
            if backup_file.is_file():
                stat = backup_file.stat()
                backups.append({
                    'name': backup_file.stem,
                    'file_path': str(backup_file),
                    'file_size': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a specific backup.
        
        Args:
            backup_name: Name of the backup to delete
            
        Returns:
            True if backup was deleted successfully
        """
        try:
            # Look for backup files with different extensions
            extensions = ['.db', '.sql', '.gz', '.tar.gz']
            
            for ext in extensions:
                backup_file = self.backup_dir / f"{backup_name}{ext}"
                if backup_file.exists():
                    backup_file.unlink()
                    logger.info(f"Deleted backup: {backup_file}")
                    return True
            
            logger.warning(f"Backup not found: {backup_name}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_name}: {e}")
            return False


# ==================== ENHANCED EXAMPLES AND TESTING ====================

if __name__ == "__main__":
    """
    Enhanced examples demonstrating advanced database features.
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("AutoERP Database Module - Advanced Examples")
    print("=" * 50)
    
    # Import configuration
    from .config import DatabaseSettings, DatabaseEngine, Environment
    
    # Example database settings
    settings = DatabaseSettings(
        engine=DatabaseEngine.SQLITE,
        sqlite_file="advanced_example.db",
        pool_size=5,
        echo_queries=False,
        slow_query_threshold=0.1  # 100ms
    )
    
    try:
        # Initialize database with advanced features
        with init_database(settings) as db_manager:
            print("\n1. Database Manager Initialization:")
            print(f"   Engine: {db_manager.engine.name}")
            print(f"   URL: {DatabaseEngineFactory._mask_password(str(db_manager.engine.url))}")
            
            # Test session pool
            print("\n2. Session Pool Testing:")
            session_pool = SessionPool(db_manager.session_manager._session_factory, max_sessions=3)
            
            session_ids = []
            for i in range(3):
                sid, session = session_pool.acquire_session()
                session_ids.append(sid)
                print(f"   Acquired session: {sid}")
            
            pool_stats = session_pool.get_pool_stats()
            print(f"   Pool stats: {pool_stats}")
            
            # Clean up sessions
            for sid in session_ids:
                session_pool.release_session(sid)
            
            # Test transaction management
            print("\n3. Transaction Management:")
            session = db_manager.get_session()
            tx_manager = TransactionManager(session)
            
            try:
                with tx_manager.transaction():
                    session.execute(text("CREATE TEMP TABLE test_tx (id INTEGER, name TEXT)"))
                    session.execute(text("INSERT INTO test_tx VALUES (1, 'test')"))
                    
                    with tx_manager.transaction("nested_savepoint"):
                        session.execute(text("INSERT INTO test_tx VALUES (2, 'nested')"))
                        # This will be committed
                    
                    tx_info = tx_manager.get_transaction_info()
                    print(f"   Active transactions: {len(tx_info)}")
                
                print("   ✅ Transactions completed successfully")
            finally:
                session.close()
            
            # Test query monitoring
            print("\n4. Query Performance Monitoring:")
            query_monitor = QueryMonitor()
            
            # Simulate some queries
            import random
            
            for i in range(10):
                duration = random.uniform(0.01, 0.5)
                query = f"SELECT * FROM test_table WHERE id = {i}"
                query_monitor.record_query(query, duration=duration)
            
            # Add a slow query
            query_monitor.record_query(
                "SELECT * FROM big_table JOIN another_table", 
                duration=1.5
            )
            
            performance_report = query_monitor.get_performance_report()
            print(f"   Total queries: {performance_report['summary']['total_queries']}")
            print(f"   Slow queries: {performance_report['summary']['slow_queries_count']}")
            print(f"   Average duration: {performance_report['summary']['avg_duration_ms']:.2f}ms")
            
            # Test schema management
            print("\n5. Schema Management:")
            schema_manager = SchemaManager(db_manager.engine)
            
            # Create a test table
            with db_manager.get_session_context() as session:
                session.execute(text("""
                    CREATE TABLE test_schema_table (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            
            table_info = schema_manager.get_table_info('test_schema_table')
            if table_info:
                print(f"   Table: {table_info['table_name']}")
                print(f"   Columns: {len(table_info['columns'])}")
                for col in table_info['columns']:
                    print(f"     - {col['name']}: {col['type']} ({'NOT NULL' if not col['nullable'] else 'NULL'})")
            
            # Generate DDL
            ddl = schema_manager.generate_ddl('test_schema_table')
            if ddl:
                print(f"\n   Generated DDL:\n{ddl}")
            
            # Test maintenance operations
            print("\n6. Database Maintenance:")
            maintenance = MaintenanceManager(db_manager)
            
            # Health check
            health_report = maintenance.perform_health_check()
            print(f"   Overall health: {health_report['overall_status']}")
            for check_name, check_result in health_report['checks'].items():
                print(f"   {check_name}: {check_result['status']}")
            
            # Database optimization
            optimization_report = maintenance.optimize_database()
            print(f"   Optimization operations: {len(optimization_report['operations_performed'])}")
            for operation in optimization_report['operations_performed']:
                print(f"     - {operation}")
            
            # Test backup operations
            print("\n7. Database Backup:")
            backup_manager = BackupManager(db_manager, "test_backups")
            
            backup_result = backup_manager.create_backup("test_backup")
            if backup_result['status'] == 'success':
                print(f"   ✅ Backup created: {backup_result['file_path']}")
                print(f"   File size: {backup_result['file_size']} bytes")
            else:
                print(f"   ❌ Backup failed: {backup_result.get('error', 'Unknown error')}")
            
            # List backups
            backups = backup_manager.list_backups()
            print(f"   Available backups: {len(backups)}")
            
            # Connection statistics
            print("\n8. Connection Statistics:")
            stats = db_manager.get_connection_stats()
            print(f"   Active connections: {stats.get('active_connections', 0)}")
            print(f"   Total queries: {stats.get('total_queries', 0)}")
            print(f"   Average query time: {stats.get('avg_query_time_ms', 0):.2f}ms")
            print(f"   Uptime: {stats.get('uptime_seconds', 0):.1f} seconds")
            
            print("\n✅ All advanced database features tested successfully!")
    
    except Exception as e:
        print(f"\n❌ Database operations failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Advanced database examples completed!")