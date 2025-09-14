# autoerp/plugins/__init__.py
"""
AutoERP Plugin System
=====================

A comprehensive plugin architecture for extending AutoERP functionality.
This module provides a flexible and secure plugin system that allows
developers to create custom extensions for the ERP system.

Features:
- Abstract plugin base class for standardized plugin development
- Plugin registry for managing loaded plugins
- Dynamic plugin loading and unloading
- Plugin dependency management
- Configuration validation
- Event-driven plugin communication
- Security sandboxing for untrusted plugins
- Plugin lifecycle management
- Hot-reloading support for development

Architecture:
- PluginBase: Abstract base class for all plugins
- PluginManager: Central plugin management system
- PluginRegistry: Registry for tracking loaded plugins
- PluginLoader: Dynamic loading and instantiation
- PluginConfig: Configuration management for plugins
- PluginEvents: Event system for plugin communication

Usage:
    # Load plugins from directory
    manager = PluginManager()
    manager.load_plugins_from_directory('plugins/')
    
    # Register a plugin manually
    manager.register_plugin(MyCustomPlugin())
    
    # Get loaded plugins
    plugins = manager.get_plugins_by_category('accounting')
    
    # Execute plugin hooks
    manager.execute_hook('before_invoice_create', invoice_data)

Security:
- Plugin isolation through sandboxing
- Permission-based access control
- Configuration validation
- Safe dynamic imports

Author: AutoERP Development Team
License: MIT
Version: 1.0.0
"""

import abc
import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
import traceback
import uuid
import weakref
from collections import defaultdict, OrderedDict
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Callable, Type, Set, Tuple,
    Protocol, runtime_checkable, Generic, TypeVar, ClassVar,
    get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field, asdict
from functools import wraps, lru_cache
import warnings

# Third-party imports (optional, with fallbacks)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn("PyYAML not available. YAML config support disabled.")

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn("jsonschema not available. Plugin config validation disabled.")

# Configure logging
logger = logging.getLogger(__name__)

# Plugin system constants
PLUGIN_FILE_EXTENSIONS = ['.py']
PLUGIN_CONFIG_FILES = ['plugin.yaml', 'plugin.yml', 'plugin.json']
DEFAULT_PLUGIN_TIMEOUT = 30  # seconds
MAX_PLUGIN_RECURSION_DEPTH = 10

# Type definitions
T = TypeVar('T')
PluginT = TypeVar('PluginT', bound='PluginBase')


# ==================== PLUGIN ENUMS AND CONSTANTS ====================

class PluginStatus(Enum):
    """Plugin status enumeration."""
    
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    ERROR = auto()
    UNLOADING = auto()


class PluginPriority(Enum):
    """Plugin execution priority levels."""
    
    CRITICAL = 1000
    HIGH = 750
    NORMAL = 500
    LOW = 250
    BACKGROUND = 100


class PluginCategory(Enum):
    """Standard plugin categories."""
    
    CORE = "core"
    ACCOUNTING = "accounting"
    INVENTORY = "inventory"
    CRM = "crm"
    HR = "hr"
    REPORTING = "reporting"
    INTEGRATION = "integration"
    UI = "ui"
    API = "api"
    WORKFLOW = "workflow"
    NOTIFICATION = "notification"
    SECURITY = "security"
    UTILITIES = "utilities"
    CUSTOM = "custom"


# ==================== PLUGIN EXCEPTIONS ====================

class PluginError(Exception):
    """Base exception for plugin-related errors."""
    
    def __init__(self, message: str, plugin_id: Optional[str] = None):
        super().__init__(message)
        self.plugin_id = plugin_id
        self.timestamp = datetime.now(timezone.utc)


class PluginLoadError(PluginError):
    """Raised when plugin loading fails."""
    pass


class PluginConfigError(PluginError):
    """Raised when plugin configuration is invalid."""
    pass


class PluginDependencyError(PluginError):
    """Raised when plugin dependencies cannot be satisfied."""
    pass


class PluginExecutionError(PluginError):
    """Raised when plugin execution fails."""
    pass


class PluginSecurityError(PluginError):
    """Raised when plugin violates security constraints."""
    pass


# ==================== PLUGIN CONFIGURATION ====================

@dataclass
class PluginDependency:
    """Plugin dependency specification."""
    
    plugin_id: str
    version: Optional[str] = None
    optional: bool = False
    
    def __str__(self) -> str:
        version_str = f"=={self.version}" if self.version else ""
        optional_str = " (optional)" if self.optional else ""
        return f"{self.plugin_id}{version_str}{optional_str}"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    
    id: str
    name: str
    version: str
    description: str
    author: str
    email: Optional[str] = None
    website: Optional[str] = None
    license: Optional[str] = None
    category: PluginCategory = PluginCategory.CUSTOM
    priority: PluginPriority = PluginPriority.NORMAL
    dependencies: List[PluginDependency] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.id:
            raise PluginConfigError("Plugin ID cannot be empty")
        
        if not self.name:
            raise PluginConfigError("Plugin name cannot be empty")
        
        if not self.version:
            raise PluginConfigError("Plugin version cannot be empty")
        
        # Ensure category is PluginCategory enum
        if isinstance(self.category, str):
            try:
                self.category = PluginCategory(self.category)
            except ValueError:
                self.category = PluginCategory.CUSTOM
        
        # Ensure priority is PluginPriority enum
        if isinstance(self.priority, str):
            try:
                self.priority = PluginPriority[self.priority.upper()]
            except KeyError:
                self.priority = PluginPriority.NORMAL
        elif isinstance(self.priority, int):
            # Find closest priority level
            closest = min(PluginPriority, key=lambda p: abs(p.value - self.priority))
            self.priority = closest


@dataclass
class PluginConfig:
    """Plugin configuration container."""
    
    enabled: bool = True
    auto_start: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = DEFAULT_PLUGIN_TIMEOUT
    sandbox: bool = True
    debug: bool = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(config)
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration against schema."""
        if not schema or not JSONSCHEMA_AVAILABLE:
            return True
        
        try:
            jsonschema.validate(self.config, schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Plugin config validation failed: {e}")
            return False


# ==================== PLUGIN EVENT SYSTEM ====================

@dataclass
class PluginEvent:
    """Plugin event data container."""
    
    event_type: str
    data: Dict[str, Any]
    source_plugin: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'source_plugin': self.source_plugin,
            'timestamp': self.timestamp.isoformat()
        }


class PluginEventBus:
    """Event bus for plugin communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_history: List[PluginEvent] = []
        self._max_history = 1000
    
    def subscribe(self, event_type: str, handler: Callable[[PluginEvent], None]) -> None:
        """Subscribe to plugin events."""
        with self._lock:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable[[PluginEvent], None]) -> None:
        """Unsubscribe from plugin events."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                    logger.debug(f"Unsubscribed from event type: {event_type}")
                except ValueError:
                    pass
    
    def emit(self, event: PluginEvent) -> None:
        """Emit plugin event to all subscribers."""
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            # Notify subscribers
            handlers = self._subscribers.get(event.event_type, [])
            
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    def get_event_history(self, event_type: Optional[str] = None) -> List[PluginEvent]:
        """Get event history, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self._event_history if e.event_type == event_type]
            return self._event_history.copy()


# ==================== PLUGIN BASE CLASS ====================

@runtime_checkable
class PluginInterface(Protocol):
    """Plugin interface protocol."""
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        ...
    
    async def initialize(self) -> None:
        """Initialize the plugin."""
        ...
    
    async def start(self) -> None:
        """Start the plugin."""
        ...
    
    async def stop(self) -> None:
        """Stop the plugin."""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...


class PluginBase(abc.ABC):
    """
    Abstract base class for all AutoERP plugins.
    
    This class provides the foundation for creating plugins that can
    extend AutoERP functionality. All plugins must inherit from this
    class and implement the required abstract methods.
    
    Example:
        class MyPlugin(PluginBase):
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    id="my_plugin",
                    name="My Custom Plugin",
                    version="1.0.0",
                    description="A sample plugin",
                    author="Developer Name"
                )
            
            async def initialize(self):
                # Plugin initialization logic
                pass
            
            async def start(self):
                # Plugin startup logic
                pass
    """
    
    def __init__(self, config: Optional[PluginConfig] = None):
        """
        Initialize plugin with optional configuration.
        
        Args:
            config: Plugin configuration object
        """
        self._config = config or PluginConfig()
        self._status = PluginStatus.UNLOADED
        self._manager: Optional['PluginManager'] = None
        self._event_bus: Optional[PluginEventBus] = None
        self._logger = logging.getLogger(f"plugin.{self.get_metadata().id}")
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._startup_time: Optional[datetime] = None
        self._stats = {
            'initialization_time': 0.0,
            'start_time': 0.0,
            'hook_calls': defaultdict(int),
            'errors': 0
        }
    
    @abc.abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata information.
        
        Returns:
            PluginMetadata: Plugin metadata object
        """
        pass
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the plugin.
        
        This method is called once when the plugin is first loaded.
        Use this for one-time setup operations like registering hooks,
        setting up database connections, etc.
        
        Raises:
            PluginError: If initialization fails
        """
        pass
    
    async def start(self) -> None:
        """
        Start the plugin.
        
        This method is called when the plugin should become active.
        Override this method to implement plugin-specific startup logic.
        
        Raises:
            PluginError: If startup fails
        """
        self._startup_time = datetime.now(timezone.utc)
        self._status = PluginStatus.ACTIVE
        self._logger.info(f"Plugin {self.get_metadata().name} started")
    
    async def stop(self) -> None:
        """
        Stop the plugin.
        
        This method is called when the plugin should become inactive.
        Override this method to implement plugin-specific shutdown logic.
        
        Raises:
            PluginError: If shutdown fails
        """
        self._status = PluginStatus.INACTIVE
        self._startup_time = None
        self._logger.info(f"Plugin {self.get_metadata().name} stopped")
    
    async def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        
        This method is called when the plugin is being unloaded.
        Use this to release resources, close connections, etc.
        
        Raises:
            PluginError: If cleanup fails
        """
        self._status = PluginStatus.UNLOADED
        self._hooks.clear()
        self._logger.info(f"Plugin {self.get_metadata().name} cleaned up")
    
    def register_hook(self, hook_name: str, handler: Callable) -> None:
        """
        Register a hook handler.
        
        Args:
            hook_name: Name of the hook to register for
            handler: Function to call when hook is executed
        """
        self._hooks[hook_name].append(handler)
        self._logger.debug(f"Registered hook: {hook_name}")
    
    def unregister_hook(self, hook_name: str, handler: Callable) -> None:
        """
        Unregister a hook handler.
        
        Args:
            hook_name: Name of the hook to unregister from
            handler: Function to remove
        """
        if hook_name in self._hooks:
            try:
                self._hooks[hook_name].remove(handler)
                self._logger.debug(f"Unregistered hook: {hook_name}")
            except ValueError:
                pass
    
    async def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Execute all handlers for a specific hook.
        
        Args:
            hook_name: Name of the hook to execute
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            List[Any]: Results from all hook handlers
        """
        results = []
        self._stats['hook_calls'][hook_name] += 1
        
        for handler in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                self._stats['errors'] += 1
                self._logger.error(f"Hook handler error in {hook_name}: {e}")
        
        return results
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit a plugin event.
        
        Args:
            event_type: Type of event to emit
            data: Event data dictionary
        """
        if self._event_bus:
            event = PluginEvent(
                event_type=event_type,
                data=data,
                source_plugin=self.get_metadata().id
            )
            self._event_bus.emit(event)
    
    def subscribe_to_event(self, event_type: str, handler: Callable[[PluginEvent], None]) -> None:
        """
        Subscribe to plugin events.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event occurs
        """
        if self._event_bus:
            self._event_bus.subscribe(event_type, handler)
    
    @property
    def status(self) -> PluginStatus:
        """Get current plugin status."""
        return self._status
    
    @property
    def config(self) -> PluginConfig:
        """Get plugin configuration."""
        return self._config
    
    @property
    def logger(self) -> logging.Logger:
        """Get plugin logger."""
        return self._logger
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get plugin statistics."""
        stats = self._stats.copy()
        stats['status'] = self._status.name
        stats['uptime'] = (
            (datetime.now(timezone.utc) - self._startup_time).total_seconds()
            if self._startup_time else 0
        )
        return stats
    
    def _set_manager(self, manager: 'PluginManager') -> None:
        """Set plugin manager reference (internal use)."""
        self._manager = manager
        self._event_bus = manager.event_bus
    
    def __str__(self) -> str:
        """String representation of plugin."""
        metadata = self.get_metadata()
        return f"{metadata.name} v{metadata.version} ({metadata.id})"
    
    def __repr__(self) -> str:
        """Detailed string representation of plugin."""
        return f"<{self.__class__.__name__}: {self}>"


# ==================== PLUGIN REGISTRY ====================

class PluginRegistry:
    """
    Registry for managing loaded plugins.
    
    Maintains a centralized registry of all loaded plugins with
    fast lookup capabilities by ID, category, and other attributes.
    """
    
    def __init__(self):
        """Initialize empty plugin registry."""
        self._plugins: Dict[str, PluginBase] = {}
        self._plugins_by_category: Dict[PluginCategory, List[PluginBase]] = defaultdict(list)
        self._plugins_by_priority: Dict[PluginPriority, List[PluginBase]] = defaultdict(list)
        self._plugin_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def register(self, plugin: PluginBase) -> None:
        """
        Register a plugin in the registry.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            PluginError: If plugin ID already exists
        """
        metadata = plugin.get_metadata()
        
        with self._lock:
            if metadata.id in self._plugins:
                raise PluginError(f"Plugin {metadata.id} already registered")
            
            self._plugins[metadata.id] = plugin
            self._plugins_by_category[metadata.category].append(plugin)
            self._plugins_by_priority[metadata.priority].append(plugin)
            
            # Register dependencies
            for dep in metadata.dependencies:
                self._plugin_dependencies[metadata.id].add(dep.plugin_id)
                self._reverse_dependencies[dep.plugin_id].add(metadata.id)
            
            logger.info(f"Registered plugin: {metadata.id}")
    
    def unregister(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_id: ID of plugin to unregister
            
        Returns:
            Optional[PluginBase]: Unregistered plugin or None if not found
        """
        with self._lock:
            plugin = self._plugins.pop(plugin_id, None)
            
            if plugin:
                metadata = plugin.get_metadata()
                
                # Remove from category and priority lists
                try:
                    self._plugins_by_category[metadata.category].remove(plugin)
                except ValueError:
                    pass
                
                try:
                    self._plugins_by_priority[metadata.priority].remove(plugin)
                except ValueError:
                    pass
                
                # Clean up dependencies
                for dep_id in self._plugin_dependencies[plugin_id]:
                    self._reverse_dependencies[dep_id].discard(plugin_id)
                
                del self._plugin_dependencies[plugin_id]
                
                for dependent_id in self._reverse_dependencies[plugin_id]:
                    self._plugin_dependencies[dependent_id].discard(plugin_id)
                
                del self._reverse_dependencies[plugin_id]
                
                logger.info(f"Unregistered plugin: {plugin_id}")
            
            return plugin
    
    def get(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Get plugin by ID.
        
        Args:
            plugin_id: ID of plugin to retrieve
            
        Returns:
            Optional[PluginBase]: Plugin instance or None if not found
        """
        with self._lock:
            return self._plugins.get(plugin_id)
    
    def get_all(self) -> Dict[str, PluginBase]:
        """
        Get all registered plugins.
        
        Returns:
            Dict[str, PluginBase]: Dictionary of all plugins keyed by ID
        """
        with self._lock:
            return self._plugins.copy()
    
    def get_by_category(self, category: PluginCategory) -> List[PluginBase]:
        """
        Get plugins by category.
        
        Args:
            category: Plugin category to filter by
            
        Returns:
            List[PluginBase]: List of plugins in the category
        """
        with self._lock:
            return self._plugins_by_category[category].copy()
    
    def get_by_priority(self, priority: PluginPriority) -> List[PluginBase]:
        """
        Get plugins by priority.
        
        Args:
            priority: Plugin priority to filter by
            
        Returns:
            List[PluginBase]: List of plugins with the priority
        """
        with self._lock:
            return self._plugins_by_priority[priority].copy()
    
    def get_by_status(self, status: PluginStatus) -> List[PluginBase]:
        """
        Get plugins by status.
        
        Args:
            status: Plugin status to filter by
            
        Returns:
            List[PluginBase]: List of plugins with the status
        """
        with self._lock:
            return [p for p in self._plugins.values() if p.status == status]
    
    def get_dependencies(self, plugin_id: str) -> Set[str]:
        """
        Get direct dependencies of a plugin.
        
        Args:
            plugin_id: ID of plugin to get dependencies for
            
        Returns:
            Set[str]: Set of dependency plugin IDs
        """
        with self._lock:
            return self._plugin_dependencies[plugin_id].copy()
    
    def get_dependents(self, plugin_id: str) -> Set[str]:
        """
        Get plugins that depend on a given plugin.
        
        Args:
            plugin_id: ID of plugin to get dependents for
            
        Returns:
            Set[str]: Set of dependent plugin IDs
        """
        with self._lock:
            return self._reverse_dependencies[plugin_id].copy()
    
    def get_load_order(self) -> List[str]:
        """
        Get plugin IDs in dependency resolution order.
        
        Returns:
            List[str]: Plugin IDs in load order
            
        Raises:
            PluginDependencyError: If circular dependencies detected
        """
        with self._lock:
            # Topological sort for dependency resolution
            visited = set()
            temp_visited = set()
            order = []
            
            def visit(plugin_id: str):
                if plugin_id in temp_visited:
                    raise PluginDependencyError(f"Circular dependency detected involving {plugin_id}")
                
                if plugin_id not in visited:
                    temp_visited.add(plugin_id)
                    
                    # Visit dependencies first
                    for dep_id in self._plugin_dependencies[plugin_id]:
                        if dep_id in self._plugins:  # Only if dependency is loaded
                            visit(dep_id)
                    
                    temp_visited.remove(plugin_id)
                    visited.add(plugin_id)
                    order.append(plugin_id)
            
            # Visit all plugins
            for plugin_id in self._plugins.keys():
                visit(plugin_id)
            
            return order
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate all plugin dependencies.
        
        Returns:
            Dict[str, List[str]]: Dictionary of plugin IDs to missing dependencies
        """
        missing_deps = {}
        
        with self._lock:
            for plugin_id, plugin in self._plugins.items():
                missing = []
                metadata = plugin.get_metadata()
                
                for dep in metadata.dependencies:
                    if dep.plugin_id not in self._plugins and not dep.optional:
                        missing.append(dep.plugin_id)
                
                if missing:
                    missing_deps[plugin_id] = missing
        
        return missing_deps
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dict[str, Any]: Statistics about registered plugins
        """
        with self._lock:
            stats = {
                'total_plugins': len(self._plugins),
                'plugins_by_category': {
                    cat.value: len(plugins) 
                    for cat, plugins in self._plugins_by_category.items()
                    if plugins
                },
                'plugins_by_priority': {
                    pri.name: len(plugins)
                    for pri, plugins in self._plugins_by_priority.items()
                    if plugins
                },
                'plugins_by_status': {}
            }
            
            # Count by status
            status_counts = defaultdict(int)
            for plugin in self._plugins.values():
                status_counts[plugin.status.name] += 1
            
            stats['plugins_by_status'] = dict(status_counts)
            
            return stats
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        with self._lock:
            self._plugins.clear()
            self._plugins_by_category.clear()
            self._plugins_by_priority.clear()
            self._plugin_dependencies.clear()
            self._reverse_dependencies.clear()
            logger.info("Cleared plugin registry")


# ==================== PLUGIN LOADER ====================

class PluginLoader:
    """
    Dynamic plugin loader for loading plugins from files and directories.
    
    Supports loading plugins from:
    - Python modules (.py files)
    - Plugin directories with metadata
    - Zip archives containing plugins
    """
    
    def __init__(self, sandbox: bool = True):
        """
        Initialize plugin loader.
        
        Args:
            sandbox: Whether to enable security sandboxing
        """
        self.sandbox = sandbox
        self._loaded_modules: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def load_from_file(self, file_path: Path, config: Optional[PluginConfig] = None) -> List[PluginBase]:
        """
        Load plugins from a Python file.
        
        Args:
            file_path: Path to Python file containing plugin classes
            config: Optional plugin configuration
            
        Returns:
            List[PluginBase]: List of loaded plugin instances
            
        Raises:
            PluginLoadError: If loading fails
        """
        if not file_path.exists():
            raise PluginLoadError(f"Plugin file not found: {file_path}")
        
        if file_path.suffix not in PLUGIN_FILE_EXTENSIONS:
            raise PluginLoadError(f"Unsupported file type: {file_path.suffix}")
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"plugin_{file_path.stem}",
                file_path
            )
            
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not create module spec for {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            with self._lock:
                self._loaded_modules[str(file_path)] = module
            
            # Execute module in sandbox if enabled
            if self.sandbox:
                self._execute_sandboxed(spec.loader.exec_module, module)
            else:
                spec.loader.exec_module(module)
            
            # Find plugin classes in module
            plugins = self._extract_plugins_from_module(module, config)
            
            logger.info(f"Loaded {len(plugins)} plugins from {file_path}")
            return plugins
            
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin from {file_path}: {e}")
    
    def load_from_directory(self, directory: Path, recursive: bool = True) -> List[PluginBase]:
        """
        Load plugins from a directory.
        
        Args:
            directory: Directory containing plugin files
            recursive: Whether to search subdirectories
            
        Returns:
            List[PluginBase]: List of loaded plugin instances
            
        Raises:
            PluginLoadError: If loading fails
        """
        if not directory.exists() or not directory.is_dir():
            raise PluginLoadError(f"Plugin directory not found: {directory}")
        
        plugins = []
        
        # Look for plugin configuration files
        config_file = self._find_config_file(directory)
        config = self._load_plugin_config(config_file) if config_file else None
        
        # Find Python files
        pattern = "**/*.py" if recursive else "*.py"
        
        for file_path in directory.glob(pattern):
            if file_path.name.startswith('__'):
                continue  # Skip __init__.py and __pycache__
            
            try:
                file_plugins = self.load_from_file(file_path, config)
                plugins.extend(file_plugins)
            except PluginLoadError as e:
                logger.warning(f"Failed to load plugin from {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(plugins)} plugins from {directory}")
        return plugins
    
    def _extract_plugins_from_module(self, module: Any, config: Optional[PluginConfig]) -> List[PluginBase]:
        """
        Extract plugin classes from loaded module.
        
        Args:
            module: Loaded Python module
            config: Optional plugin configuration
            
        Returns:
            List[PluginBase]: List of plugin instances
        """
        plugins = []
        
        for name in dir(module):
            obj = getattr(module, name)
            
            # Check if it's a plugin class
            if (inspect.isclass(obj) and
                issubclass(obj, PluginBase) and
                obj != PluginBase and
                not inspect.isabstract(obj)):
                
                try:
                    # Instantiate plugin
                    plugin_instance = obj(config)
                    plugins.append(plugin_instance)
                    logger.debug(f"Created plugin instance: {name}")
                    
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {name}: {e}")
        
        return plugins
    
    def _find_config_file(self, directory: Path) -> Optional[Path]:
        """
        Find plugin configuration file in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            Optional[Path]: Path to config file or None
        """
        for config_name in PLUGIN_CONFIG_FILES:
            config_path = directory / config_name
            if config_path.exists():
                return config_path
        
        return None
    
    def _load_plugin_config(self, config_file: Path) -> Optional[PluginConfig]:
        """
        Load plugin configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Optional[PluginConfig]: Loaded configuration or None
        """
        try:
            if config_file.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                with open(config_file, 'r') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_file}")
                return None
            
            return PluginConfig(**data)
            
        except Exception as e:
            logger.error(f"Failed to load plugin config from {config_file}: {e}")
            return None
    
    def _execute_sandboxed(self, exec_func: Callable, *args) -> None:
        """
        Execute code in a sandboxed environment.
        
        Args:
            exec_func: Function to execute
            *args: Arguments to pass to function
        """
        # Basic sandboxing - restrict dangerous operations
        original_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else vars(__builtins__).copy()
        
        # Remove dangerous functions
        restricted_builtins = original_builtins.copy()
        for dangerous in ['exec', 'eval', 'compile', '__import__']:
            restricted_builtins.pop(dangerous, None)
        
        try:
            # Temporarily replace builtins
            if isinstance(__builtins__, dict):
                __builtins__.clear()
                __builtins__.update(restricted_builtins)
            else:
                for key, value in restricted_builtins.items():
                    setattr(__builtins__, key, value)
            
            exec_func(*args)
            
        finally:
            # Restore original builtins
            if isinstance(__builtins__, dict):
                __builtins__.clear()
                __builtins__.update(original_builtins)
            else:
                for key in list(vars(__builtins__).keys()):
                    delattr(__builtins__, key)
                for key, value in original_builtins.items():
                    setattr(__builtins__, key, value)
    
    def unload_module(self, file_path: Path) -> None:
        """
        Unload a previously loaded module.
        
        Args:
            file_path: Path to the module file
        """
        with self._lock:
            module = self._loaded_modules.pop(str(file_path), None)
            if module and hasattr(module, '__name__'):
                sys.modules.pop(module.__name__, None)
                logger.debug(f"Unloaded module: {file_path}")


CONTINUE PLUGINS

Copier
Recherche approfondie

CONTINUE PLUGINS

# ==================== PLUGIN MANAGER ====================

class PluginManager:
    """
    Central plugin management system for AutoERP.
    
    The PluginManager is responsible for:
    - Loading and unloading plugins
    - Managing plugin lifecycle
    - Coordinating plugin communication
    - Enforcing security policies
    - Managing plugin dependencies
    - Providing plugin APIs
    
    Example:
        manager = PluginManager()
        manager.load_plugins_from_directory('plugins/')
        manager.start_all_plugins()
        
        # Execute plugin hooks
        results = await manager.execute_hook('before_invoice_create', invoice_data)
        
        # Get plugins by category
        accounting_plugins = manager.get_plugins_by_category(PluginCategory.ACCOUNTING)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = PluginRegistry()
        self.loader = PluginLoader(sandbox=self.config.get('sandbox', True))
        self.event_bus = PluginEventBus()
        
        self._hook_registry: Dict[str, List[Tuple[PluginBase, Callable]]] = defaultdict(list)
        self._global_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._plugin_contexts: Dict[str, Dict[str, Any]] = {}
        self._manager_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Performance monitoring
        self._performance_stats = {
            'plugin_load_times': {},
            'hook_execution_times': defaultdict(list),
            'total_plugins_loaded': 0,
            'total_hooks_executed': 0
        }
        
        logger.info("Plugin manager initialized")
    
    def register_plugin(self, plugin: PluginBase) -> None:
        """
        Register a plugin instance with the manager.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            PluginError: If plugin registration fails
        """
        with self._manager_lock:
            try:
                metadata = plugin.get_metadata()
                
                # Validate plugin metadata
                self._validate_plugin_metadata(metadata)
                
                # Check dependencies
                self._check_plugin_dependencies(metadata)
                
                # Register with registry
                self.registry.register(plugin)
                
                # Set manager reference
                plugin._set_manager(self)
                
                # Initialize plugin context
                self._plugin_contexts[metadata.id] = {
                    'registered_at': datetime.now(timezone.utc),
                    'initialization_attempts': 0,
                    'last_error': None
                }
                
                logger.info(f"Registered plugin: {metadata.id}")
                
                # Emit event
                self.event_bus.emit(PluginEvent(
                    event_type='plugin_registered',
                    data={'plugin_id': metadata.id, 'plugin_name': metadata.name}
                ))
                
            except Exception as e:
                raise PluginError(f"Failed to register plugin: {e}", 
                                getattr(plugin.get_metadata(), 'id', 'unknown'))
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin from the manager.
        
        Args:
            plugin_id: ID of plugin to unregister
            
        Returns:
            bool: True if successfully unregistered, False otherwise
        """
        with self._manager_lock:
            try:
                plugin = self.registry.get(plugin_id)
                if not plugin:
                    return False
                
                # Stop plugin if running
                if plugin.status == PluginStatus.ACTIVE:
                    asyncio.create_task(self.stop_plugin(plugin_id))
                
                # Cleanup plugin
                asyncio.create_task(plugin.cleanup())
                
                # Remove from hook registry
                self._remove_plugin_hooks(plugin)
                
                # Unregister from registry
                self.registry.unregister(plugin_id)
                
                # Clean up context
                self._plugin_contexts.pop(plugin_id, None)
                
                logger.info(f"Unregistered plugin: {plugin_id}")
                
                # Emit event
                self.event_bus.emit(PluginEvent(
                    event_type='plugin_unregistered',
                    data={'plugin_id': plugin_id}
                ))
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister plugin {plugin_id}: {e}")
                return False
    
    async def initialize_plugin(self, plugin_id: str) -> bool:
        """
        Initialize a specific plugin.
        
        Args:
            plugin_id: ID of plugin to initialize
            
        Returns:
            bool: True if successful, False otherwise
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        if plugin.status != PluginStatus.UNLOADED:
            logger.warning(f"Plugin {plugin_id} already initialized")
            return True
        
        try:
            plugin._status = PluginStatus.LOADING
            
            # Track initialization attempts
            context = self._plugin_contexts.get(plugin_id, {})
            context['initialization_attempts'] = context.get('initialization_attempts', 0) + 1
            
            start_time = datetime.now()
            
            # Initialize plugin
            await plugin.initialize()
            
            # Record performance stats
            init_time = (datetime.now() - start_time).total_seconds()
            self._performance_stats['plugin_load_times'][plugin_id] = init_time
            
            plugin._status = PluginStatus.LOADED
            
            # Update context
            context['last_initialized'] = datetime.now(timezone.utc)
            context['last_error'] = None
            
            logger.info(f"Initialized plugin: {plugin_id} (took {init_time:.3f}s)")
            
            # Emit event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_initialized',
                data={
                    'plugin_id': plugin_id,
                    'initialization_time': init_time
                }
            ))
            
            return True
            
        except Exception as e:
            plugin._status = PluginStatus.ERROR
            context = self._plugin_contexts.get(plugin_id, {})
            context['last_error'] = str(e)
            
            logger.error(f"Failed to initialize plugin {plugin_id}: {e}")
            
            # Emit error event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_initialization_failed',
                data={
                    'plugin_id': plugin_id,
                    'error': str(e)
                }
            ))
            
            return False
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """
        Start a specific plugin.
        
        Args:
            plugin_id: ID of plugin to start
            
        Returns:
            bool: True if successful, False otherwise
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        if plugin.status == PluginStatus.ACTIVE:
            logger.warning(f"Plugin {plugin_id} already running")
            return True
        
        if plugin.status != PluginStatus.LOADED:
            # Initialize first
            if not await self.initialize_plugin(plugin_id):
                return False
        
        try:
            start_time = datetime.now()
            
            # Start plugin
            await plugin.start()
            
            start_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Started plugin: {plugin_id} (took {start_duration:.3f}s)")
            
            # Emit event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_started',
                data={
                    'plugin_id': plugin_id,
                    'start_time': start_duration
                }
            ))
            
            return True
            
        except Exception as e:
            plugin._status = PluginStatus.ERROR
            context = self._plugin_contexts.get(plugin_id, {})
            context['last_error'] = str(e)
            
            logger.error(f"Failed to start plugin {plugin_id}: {e}")
            
            # Emit error event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_start_failed',
                data={
                    'plugin_id': plugin_id,
                    'error': str(e)
                }
            ))
            
            return False
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """
        Stop a specific plugin.
        
        Args:
            plugin_id: ID of plugin to stop
            
        Returns:
            bool: True if successful, False otherwise
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        if plugin.status != PluginStatus.ACTIVE:
            logger.warning(f"Plugin {plugin_id} not running")
            return True
        
        try:
            # Check for dependents that are still running
            dependents = self.registry.get_dependents(plugin_id)
            active_dependents = []
            
            for dependent_id in dependents:
                dependent = self.registry.get(dependent_id)
                if dependent and dependent.status == PluginStatus.ACTIVE:
                    active_dependents.append(dependent_id)
            
            if active_dependents:
                logger.warning(f"Cannot stop plugin {plugin_id}: active dependents {active_dependents}")
                return False
            
            # Stop plugin
            await plugin.stop()
            
            logger.info(f"Stopped plugin: {plugin_id}")
            
            # Emit event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_stopped',
                data={'plugin_id': plugin_id}
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop plugin {plugin_id}: {e}")
            
            # Emit error event
            self.event_bus.emit(PluginEvent(
                event_type='plugin_stop_failed',
                data={
                    'plugin_id': plugin_id,
                    'error': str(e)
                }
            ))
            
            return False
    
    def load_plugins_from_directory(self, directory: Union[str, Path], recursive: bool = True) -> int:
        """
        Load plugins from a directory.
        
        Args:
            directory: Directory path containing plugins
            recursive: Whether to search subdirectories
            
        Returns:
            int: Number of plugins successfully loaded
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Plugin directory not found: {directory}")
            return 0
        
        try:
            plugins = self.loader.load_from_directory(directory, recursive)
            
            loaded_count = 0
            for plugin in plugins:
                try:
                    self.register_plugin(plugin)
                    loaded_count += 1
                except PluginError as e:
                    logger.error(f"Failed to register plugin: {e}")
            
            logger.info(f"Loaded {loaded_count} plugins from {directory}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load plugins from {directory}: {e}")
            return 0
    
    def load_plugin_from_file(self, file_path: Union[str, Path], config: Optional[PluginConfig] = None) -> int:
        """
        Load plugins from a specific file.
        
        Args:
            file_path: Path to plugin file
            config: Optional plugin configuration
            
        Returns:
            int: Number of plugins successfully loaded
        """
        file_path = Path(file_path)
        
        try:
            plugins = self.loader.load_from_file(file_path, config)
            
            loaded_count = 0
            for plugin in plugins:
                try:
                    self.register_plugin(plugin)
                    loaded_count += 1
                except PluginError as e:
                    logger.error(f"Failed to register plugin: {e}")
            
            logger.info(f"Loaded {loaded_count} plugins from {file_path}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load plugins from {file_path}: {e}")
            return 0
    
    async def initialize_all_plugins(self) -> Dict[str, bool]:
        """
        Initialize all registered plugins.
        
        Returns:
            Dict[str, bool]: Results of initialization for each plugin
        """
        results = {}
        
        # Get plugins in dependency order
        try:
            plugin_order = self.registry.get_load_order()
        except PluginDependencyError as e:
            logger.error(f"Cannot initialize plugins due to dependency error: {e}")
            return results
        
        for plugin_id in plugin_order:
            results[plugin_id] = await self.initialize_plugin(plugin_id)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Initialized {successful}/{len(results)} plugins")
        
        return results
    
    async def start_all_plugins(self) -> Dict[str, bool]:
        """
        Start all loaded plugins.
        
        Returns:
            Dict[str, bool]: Results of starting for each plugin
        """
        results = {}
        
        # Get plugins in dependency order
        try:
            plugin_order = self.registry.get_load_order()
        except PluginDependencyError as e:
            logger.error(f"Cannot start plugins due to dependency error: {e}")
            return results
        
        for plugin_id in plugin_order:
            plugin = self.registry.get(plugin_id)
            if plugin and plugin.config.auto_start:
                results[plugin_id] = await self.start_plugin(plugin_id)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Started {successful}/{len(results)} plugins")
        
        return results
    
    async def stop_all_plugins(self) -> Dict[str, bool]:
        """
        Stop all running plugins.
        
        Returns:
            Dict[str, bool]: Results of stopping for each plugin
        """
        results = {}
        
        # Stop in reverse dependency order
        try:
            plugin_order = list(reversed(self.registry.get_load_order()))
        except PluginDependencyError:
            # If there are dependency issues, stop all active plugins
            plugin_order = [p for p in self.registry.get_all().keys()]
        
        for plugin_id in plugin_order:
            plugin = self.registry.get(plugin_id)
            if plugin and plugin.status == PluginStatus.ACTIVE:
                results[plugin_id] = await self.stop_plugin(plugin_id)
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Stopped {successful}/{len(results)} plugins")
        
        return results
    
    def register_global_hook(self, hook_name: str, handler: Callable) -> None:
        """
        Register a global hook handler.
        
        Args:
            hook_name: Name of hook to register for
            handler: Handler function to call
        """
        with self._manager_lock:
            self._global_hooks[hook_name].append(handler)
            logger.debug(f"Registered global hook: {hook_name}")
    
    def unregister_global_hook(self, hook_name: str, handler: Callable) -> None:
        """
        Unregister a global hook handler.
        
        Args:
            hook_name: Name of hook to unregister from
            handler: Handler function to remove
        """
        with self._manager_lock:
            try:
                self._global_hooks[hook_name].remove(handler)
                logger.debug(f"Unregistered global hook: {hook_name}")
            except ValueError:
                pass
    
    async def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Execute all handlers for a specific hook.
        
        Args:
            hook_name: Name of hook to execute
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            List[Any]: Results from all hook handlers
        """
        start_time = datetime.now()
        results = []
        
        with self._manager_lock:
            # Execute global hooks first
            for handler in self._global_hooks.get(hook_name, []):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(*args, **kwargs)
                    else:
                        result = handler(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Global hook handler error in {hook_name}: {e}")
            
            # Execute plugin hooks in priority order
            plugin_handlers = []
            
            for plugin_id, plugin in self.registry.get_all().items():
                if plugin.status == PluginStatus.ACTIVE:
                    plugin_results = await plugin.execute_hook(hook_name, *args, **kwargs)
                    results.extend(plugin_results)
        
        # Record performance stats
        execution_time = (datetime.now() - start_time).total_seconds()
        self._performance_stats['hook_execution_times'][hook_name].append(execution_time)
        self._performance_stats['total_hooks_executed'] += 1
        
        logger.debug(f"Executed hook {hook_name} with {len(results)} results (took {execution_time:.3f}s)")
        
        return results
    
    def get_plugins_by_category(self, category: PluginCategory) -> List[PluginBase]:
        """
        Get plugins by category.
        
        Args:
            category: Plugin category to filter by
            
        Returns:
            List[PluginBase]: List of plugins in category
        """
        return self.registry.get_by_category(category)
    
    def get_plugins_by_status(self, status: PluginStatus) -> List[PluginBase]:
        """
        Get plugins by status.
        
        Args:
            status: Plugin status to filter by
            
        Returns:
            List[PluginBase]: List of plugins with status
        """
        return self.registry.get_by_status(status)
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """
        Get a specific plugin by ID.
        
        Args:
            plugin_id: ID of plugin to retrieve
            
        Returns:
            Optional[PluginBase]: Plugin instance or None
        """
        return self.registry.get(plugin_id)
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered plugins.
        
        Returns:
            Dict[str, Dict[str, Any]]: Plugin information keyed by ID
        """
        plugin_info = {}
        
        for plugin_id, plugin in self.registry.get_all().items():
            metadata = plugin.get_metadata()
            context = self._plugin_contexts.get(plugin_id, {})
            
            plugin_info[plugin_id] = {
                'name': metadata.name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'category': metadata.category.value,
                'priority': metadata.priority.name,
                'status': plugin.status.name,
                'dependencies': [str(dep) for dep in metadata.dependencies],
                'registered_at': context.get('registered_at'),
                'initialization_attempts': context.get('initialization_attempts', 0),
                'last_error': context.get('last_error'),
                'stats': plugin.stats
            }
        
        return plugin_info
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive plugin system statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        registry_stats = self.registry.get_statistics()
        
        stats = {
            'registry': registry_stats,
            'performance': self._performance_stats.copy(),
            'event_bus': {
                'total_events': len(self.event_bus.get_event_history()),
                'subscribers': len(self.event_bus._subscribers)
            },
            'manager': {
                'total_hooks_registered': len(self._global_hooks),
                'plugin_contexts': len(self._plugin_contexts)
            }
        }
        
        # Calculate average hook execution times
        avg_hook_times = {}
        for hook_name, times in self._performance_stats['hook_execution_times'].items():
            if times:
                avg_hook_times[hook_name] = {
                    'count': len(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        stats['performance']['average_hook_times'] = avg_hook_times
        
        return stats
    
    def validate_plugin_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate all plugin dependencies.
        
        Returns:
            Dict[str, List[str]]: Missing dependencies per plugin
        """
        return self.registry.validate_dependencies()
    
    async def reload_plugin(self, plugin_id: str) -> bool:
        """
        Reload a plugin (stop, unregister, reload, register, start).
        
        Args:
            plugin_id: ID of plugin to reload
            
        Returns:
            bool: True if successful, False otherwise
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            logger.error(f"Plugin not found: {plugin_id}")
            return False
        
        try:
            # Store original config and metadata
            config = plugin.config
            was_active = plugin.status == PluginStatus.ACTIVE
            
            # Stop and unregister
            if was_active:
                await self.stop_plugin(plugin_id)
            
            self.unregister_plugin(plugin_id)
            
            # TODO: Implement actual module reloading
            # This would require tracking the source file and reloading it
            
            logger.info(f"Reloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_id}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """
        Shutdown the plugin manager and all plugins.
        """
        logger.info("Shutting down plugin manager...")
        
        self._shutdown_event.set()
        
        # Stop all plugins
        await self.stop_all_plugins()
        
        # Unregister all plugins
        with self._manager_lock:
            plugin_ids = list(self.registry.get_all().keys())
            
            for plugin_id in plugin_ids:
                self.unregister_plugin(plugin_id)
        
        # Clear registries
        self.registry.clear()
        self._hook_registry.clear()
        self._global_hooks.clear()
        self._plugin_contexts.clear()
        
        logger.info("Plugin manager shutdown complete")
    
    def _validate_plugin_metadata(self, metadata: PluginMetadata) -> None:
        """
        Validate plugin metadata.
        
        Args:
            metadata: Plugin metadata to validate
            
        Raises:
            PluginConfigError: If metadata is invalid
        """
        if not metadata.id:
            raise PluginConfigError("Plugin ID is required")
        
        if not metadata.name:
            raise PluginConfigError("Plugin name is required")
        
        if not metadata.version:
            raise PluginConfigError("Plugin version is required")
        
        # Validate ID format
        if not metadata.id.replace('_', '').replace('-', '').isalnum():
            raise PluginConfigError("Plugin ID must be alphanumeric with underscores/hyphens")
        
        # Validate version format (basic)
        version_parts = metadata.version.split('.')
        if len(version_parts) < 2:
            raise PluginConfigError("Plugin version must be in format 'major.minor' or 'major.minor.patch'")
    
    def _check_plugin_dependencies(self, metadata: PluginMetadata) -> None:
        """
        Check if plugin dependencies can be satisfied.
        
        Args:
            metadata: Plugin metadata with dependencies
            
        Raises:
            PluginDependencyError: If dependencies cannot be satisfied
        """
        missing_deps = []
        
        for dep in metadata.dependencies:
            if not dep.optional:
                dep_plugin = self.registry.get(dep.plugin_id)
                if not dep_plugin:
                    missing_deps.append(dep.plugin_id)
        
        if missing_deps:
            raise PluginDependencyError(f"Missing required dependencies: {missing_deps}")
    
    def _remove_plugin_hooks(self, plugin: PluginBase) -> None:
        """
        Remove all hooks registered by a plugin.
        
        Args:
            plugin: Plugin whose hooks to remove
        """
        with self._manager_lock:
            hooks_to_remove = []
            
            for hook_name, handlers in self._hook_registry.items():
                handlers_to_remove = []
                
                for handler_plugin, handler_func in handlers:
                    if handler_plugin == plugin:
                        handlers_to_remove.append((handler_plugin, handler_func))
                
                for handler_tuple in handlers_to_remove:
                    handlers.remove(handler_tuple)
                
                if not handlers:
                    hooks_to_remove.append(hook_name)
            
            for hook_name in hooks_to_remove:
                del self._hook_registry[hook_name]
    
    @contextmanager
    def plugin_context(self, plugin_id: str):
        """
        Context manager for plugin operations.
        
        Args:
            plugin_id: ID of plugin to create context for
        """
        plugin = self.registry.get(plugin_id)
        if not plugin:
            raise PluginError(f"Plugin not found: {plugin_id}")
        
        logger.debug(f"Entering plugin context: {plugin_id}")
        
        try:
            yield plugin
        except Exception as e:
            logger.error(f"Error in plugin context {plugin_id}: {e}")
            raise
        finally:
            logger.debug(f"Exiting plugin context: {plugin_id}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.create_task(self.shutdown())


# ==================== EXAMPLE PLUGIN IMPLEMENTATIONS ====================

class CorePlugin(PluginBase):
    """
    Example core plugin that provides essential ERP functionality.
    
    This plugin demonstrates how to create a basic plugin that integrates
    with the AutoERP core system.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get core plugin metadata."""
        return PluginMetadata(
            id="autoerp_core",
            name="AutoERP Core Plugin",
            version="1.0.0",
            description="Core functionality for AutoERP system",
            author="AutoERP Development Team",
            email="dev@autoerp.com",
            category=PluginCategory.CORE,
            priority=PluginPriority.CRITICAL,
            permissions=["core.read", "core.write"],
            config_schema={
                "type": "object",
                "properties": {
                    "enable_audit": {"type": "boolean", "default": True},
                    "max_records": {"type": "integer", "default": 10000}
                }
            },
            tags=["core", "essential", "erp"]
        )
    
    async def initialize(self) -> None:
        """Initialize core plugin."""
        self.logger.info("Initializing core plugin...")
        
        # Register core hooks
        self.register_hook("before_entity_save", self._audit_entity_save)
        self.register_hook("after_entity_save", self._log_entity_save)
        
        # Subscribe to events
        self.subscribe_to_event("user_created", self._handle_user_created)
        
        self.logger.info("Core plugin initialized")
    
    async def start(self) -> None:
        """Start core plugin."""
        await super().start()
        
        # Emit startup event
        self.emit_event("core_plugin_started", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": self.config.config
        })
    
    def _audit_entity_save(self, entity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Audit hook for entity saves."""
        if self.config.get("enable_audit", True):
            entity_data["audit_timestamp"] = datetime.now(timezone.utc).isoformat()
            entity_data["audit_user"] = "system"  # Would get from context
            self.logger.debug(f"Added audit data to entity: {entity_data.get('id', 'unknown')}")
        
        return entity_data
    
    def _log_entity_save(self, entity_data: Dict[str, Any]) -> None:
        """Logging hook for entity saves."""
        entity_type = entity_data.get("__type__", "Unknown")
        entity_id = entity_data.get("id", "unknown")
        self.logger.info(f"Saved {entity_type} entity: {entity_id}")
    
    def _handle_user_created(self, event: PluginEvent) -> None:
        """Handle user created events."""
        user_data = event.data
        self.logger.info(f"New user created: {user_data.get('username', 'unknown')}")


class AccountingPlugin(PluginBase):
    """
    Example accounting plugin that provides financial management features.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get accounting plugin metadata."""
        return PluginMetadata(
            id="autoerp_accounting",
            name="Accounting & Finance Plugin",
            version="1.2.0",
            description="Comprehensive accounting and financial management",
            author="AutoERP Accounting Team",
            email="accounting@autoerp.com",
            category=PluginCategory.ACCOUNTING,
            priority=PluginPriority.HIGH,
            dependencies=[
                PluginDependency("autoerp_core", "1.0.0"),
            ],
            permissions=["accounting.read", "accounting.write", "finance.manage"],
            config_schema={
                "type": "object",
                "properties": {
                    "default_currency": {"type": "string", "default": "USD"},
                    "fiscal_year_start": {"type": "string", "default": "01-01"},
                    "enable_multi_currency": {"type": "boolean", "default": False}
                }
            },
            tags=["accounting", "finance", "money"]
        )
    
    async def initialize(self) -> None:
        """Initialize accounting plugin."""
        self.logger.info("Initializing accounting plugin...")
        
        # Register accounting hooks
        self.register_hook("before_invoice_create", self._validate_invoice)
        self.register_hook("after_payment_received", self._update_accounts_receivable)
        
        # Set up default currency
        self.default_currency = self.config.get("default_currency", "USD")
        
        self.logger.info("Accounting plugin initialized")
    
    async def start(self) -> None:
        """Start accounting plugin."""
        await super().start()
        
        # Initialize chart of accounts
        await self._initialize_chart_of_accounts()
        
        self.emit_event("accounting_plugin_started", {
            "default_currency": self.default_currency,
            "multi_currency_enabled": self.config.get("enable_multi_currency", False)
        })
    
    def _validate_invoice(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate invoice before creation."""
        # Add validation logic
        if not invoice_data.get("amount") or invoice_data["amount"] <= 0:
            raise ValueError("Invoice amount must be positive")
        
        if not invoice_data.get("currency"):
            invoice_data["currency"] = self.default_currency
        
        self.logger.debug(f"Validated invoice: {invoice_data.get('id', 'new')}")
        return invoice_data
    
    def _update_accounts_receivable(self, payment_data: Dict[str, Any]) -> None:
        """Update accounts receivable after payment."""
        # Add accounting logic
        amount = payment_data.get("amount", 0)
        self.logger.info(f"Updated A/R for payment of {amount}")
    
    async def _initialize_chart_of_accounts(self) -> None:
        """Initialize default chart of accounts."""
        # Add chart of accounts setup
        self.logger.info("Initialized chart of accounts")


# ==================== PLUGIN UTILITY FUNCTIONS ====================

def load_plugins(plugin_dir: Union[str, Path] = "plugins") -> PluginManager:
    """
    Convenience function to load plugins from a directory.
    
    Args:
        plugin_dir: Directory containing plugins
        
    Returns:
        PluginManager: Configured plugin manager with loaded plugins
        
    Example:
        manager = load_plugins("my_plugins/")
        await manager.start_all_plugins()
    """
    manager = PluginManager()
    manager.load_plugins_from_directory(plugin_dir)
    return manager


def register_plugin(plugin_class: Type[PluginBase], manager: Optional[PluginManager] = None) -> PluginManager:
    """
    Register a plugin class with a manager.
    
    Args:
        plugin_class: Plugin class to register
        manager: Optional existing manager, creates new one if None
        
    Returns:
        PluginManager: Manager with registered plugin
        
    Example:
        manager = register_plugin(MyCustomPlugin)
        await manager.initialize_all_plugins()
    """
    if manager is None:
        manager = PluginManager()
    
    plugin_instance = plugin_class()
    manager.register_plugin(plugin_instance)
    
    return manager


def list_plugins(manager: PluginManager) -> None:
    """
    Print information about all loaded plugins.
    
    Args:
        manager: Plugin manager to list plugins from
        
    Example:
        manager = load_plugins()
        list_plugins(manager)
    """
    plugins = manager.list_plugins()
    
    print(f"\n{'='*60}")
    print(f"{'AutoERP Plugin System - Loaded Plugins':^60}")
    print(f"{'='*60}")
    print(f"Total Plugins: {len(plugins)}\n")
    
    for plugin_id, info in plugins.items():
        print(f"Plugin: {info['name']} (ID: {plugin_id})")
        print(f"  Version: {info['version']}")
        print(f"  Author: {info['author']}")
        print(f"  Category: {info['category']}")
        print(f"  Status: {info['status']}")
        print(f"  Description: {info['description']}")
        
        if info['dependencies']:
            print(f"  Dependencies: {', '.join(info['dependencies'])}")
        
        print()


# ==================== MODULE EXPORTS ====================

__all__ = [
    # Core classes
    "PluginBase",
    "PluginManager", 
    "PluginRegistry",
    "PluginLoader",
    
    # Configuration
    "PluginMetadata",
    "PluginConfig",
    "PluginDependency",
    
    # Enums
    "PluginStatus",
    "PluginPriority", 
    "PluginCategory",
    
    # Events
    "PluginEvent",
    "PluginEventBus",
    
    # Exceptions
    "PluginError",
    "PluginLoadError",
    "PluginConfigError",
    "PluginDependencyError",
    "PluginExecutionError",
    "PluginSecurityError",
    
    # Example plugins
    "CorePlugin",
    "AccountingPlugin",
    
    # Utility functions
    "load_plugins",
    "register_plugin", 
    "list_plugins"
]


# Initialize default plugin manager instance
default_manager: Optional[PluginManager] = None


def get_default_manager() -> PluginManager:
    """
    Get or create the default plugin manager instance.
    
    Returns:
        PluginManager: Default plugin manager
    """
    global default_manager
    if default_manager is None:
        default_manager = PluginManager()
    return default_manager


# Auto-discovery of plugins in standard locations
def auto_discover_plugins() -> None:
    """
    Automatically discover and load plugins from standard locations.
    """
    manager = get_default_manager()
    
    # Standard plugin directories
    plugin_dirs = [
        Path(__file__).parent.parent / "plugins",
        Path.cwd() / "plugins",
        Path.home() / ".autoerp" / "plugins"
    ]
    
    for plugin_dir in plugin_dirs:
        if plugin_dir.exists():
            try:
                count = manager.load_plugins_from_directory(plugin_dir)
                if count > 0:
                    logger.info(f"Auto-discovered {count} plugins from {plugin_dir}")
            except Exception as e:
                logger.warning(f"Failed to auto-discover plugins from {plugin_dir}: {e}")


# Run auto-discovery on module import
try:
    auto_discover_plugins()
except Exception as e:
    logger.warning(f"Plugin auto-discovery failed: {e}")


if __name__ == "__main__":
    # Example usage when run as script
    import asyncio
    
    async def main():
        print("AutoERP Plugin System Demo")
        print("=" * 40)
        
        # Create manager
        manager = PluginManager()
        
        # Register example plugins
        manager.register_plugin(CorePlugin())
        manager.register_plugin(AccountingPlugin())
        
        # Initialize and start plugins
        await manager.initialize_all_plugins()
        await manager.start_all_plugins()
        
        # List loaded plugins
        list_plugins(manager)
        
        # Execute some hooks
        results = await manager.execute_hook("before_entity_save", {
            "id": "test123",
            "__type__": "TestEntity",
            "data": "example"
        })
        
        print(f"Hook execution results: {results}")
        
        # Show statistics
        stats = manager.get_plugin_statistics()
        print(f"\nPlugin System Statistics:")
        print(f"Total plugins: {stats['registry']['total_plugins']}")
        print(f"Active plugins: {stats['registry']['plugins_by_status'].get('ACTIVE', 0)}")
        
        # Cleanup
        await manager.shutdown()
        print("\nShutdown complete")
    
    asyncio.run(main())