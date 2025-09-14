#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoERP Setup Configuration
==========================

Professional setup.py for the AutoERP Enterprise Resource Planning System.
This setup script provides comprehensive configuration for package distribution,
installation, and development workflows.

Features:
- Dynamic version detection from multiple sources
- Flexible dependency management with extras
- Development tools integration
- Plugin system support
- Cross-platform compatibility
- Professional package metadata

Author: AutoERP Development Team
License: MIT
Version: 1.0.0
"""

import os
import sys
import re
import ast
import subprocess
import codecs
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

# Core setup tools imports
from setuptools import setup, find_packages, Command
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand

# Distutils imports for custom commands
from distutils.command.clean import clean
from distutils.util import convert_path
from distutils import log

# Suppress setuptools warnings
warnings.filterwarnings('ignore', category=UserWarning, module='setuptools')

# ==================== PACKAGE METADATA ====================

# Basic package information
PACKAGE_NAME = "autoerp"
PACKAGE_DIR = Path(__file__).parent
SOURCE_DIR = PACKAGE_DIR / PACKAGE_NAME
README_FILE = PACKAGE_DIR / "README.md"
CHANGES_FILE = PACKAGE_DIR / "CHANGES.md"
REQUIREMENTS_DIR = PACKAGE_DIR / "requirements"

# Package URLs and contact information
PACKAGE_URL = "https://github.com/autoerp/autoerp"
DOCUMENTATION_URL = "https://docs.autoerp.com"
REPOSITORY_URL = "https://github.com/autoerp/autoerp.git"
BUG_TRACKER_URL = "https://github.com/autoerp/autoerp/issues"
CHANGELOG_URL = "https://github.com/autoerp/autoerp/blob/main/CHANGES.md"

# Author information
AUTHOR_NAME = "AutoERP Development Team"
AUTHOR_EMAIL = "dev@autoerp.com"
MAINTAINER_NAME = "AutoERP Development Team" 
MAINTAINER_EMAIL = "maintainer@autoerp.com"

# License information
LICENSE_NAME = "MIT"
LICENSE_FILE = PACKAGE_DIR / "LICENSE"

# Package description
SHORT_DESCRIPTION = "Modern Enterprise Resource Planning System with Hexagonal Architecture"

LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# ==================== UTILITY FUNCTIONS ====================

def read_file(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read content from a file safely with proper error handling.
    
    Args:
        file_path: Path to the file to read
        encoding: Character encoding to use (default: utf-8)
        
    Returns:
        str: File content or empty string if file not found
        
    Example:
        >>> content = read_file("README.md")
        >>> print(len(content))
    """
    file_path = Path(file_path)
    
    try:
        with codecs.open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
        log.info(f"Successfully read {file_path} ({len(content)} characters)")
        return content
    except (IOError, OSError, FileNotFoundError) as e:
        log.warn(f"Could not read {file_path}: {e}")
        return ""
    except UnicodeDecodeError as e:
        log.warn(f"Unicode decode error in {file_path}: {e}")
        try:
            # Fallback to latin-1 encoding
            with codecs.open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception:
            return ""


def load_requirements(filename: str, requirements_dir: Optional[Path] = None) -> List[str]:
    """
    Load requirements from a requirements file.
    
    Handles comments, empty lines, and git+https URLs properly.
    Supports both absolute and relative paths.
    
    Args:
        filename: Name of the requirements file (e.g., 'base.txt')
        requirements_dir: Directory containing requirements files
        
    Returns:
        List[str]: List of requirement specifications
        
    Example:
        >>> reqs = load_requirements('base.txt')
        >>> print(f"Found {len(reqs)} requirements")
    """
    if requirements_dir is None:
        requirements_dir = REQUIREMENTS_DIR
    
    req_file = requirements_dir / filename
    
    if not req_file.exists():
        log.warn(f"Requirements file not found: {req_file}")
        return []
    
    requirements = []
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle -r includes (recursive requirements files)
                if line.startswith('-r '):
                    include_file = line[3:].strip()
                    included_reqs = load_requirements(include_file, requirements_dir)
                    requirements.extend(included_reqs)
                    continue
                
                # Handle -e editable installs
                if line.startswith('-e '):
                    line = line[3:].strip()
                
                # Handle inline comments
                if ' #' in line:
                    line = line.split(' #')[0].strip()
                
                # Skip lines with environment markers we can't handle
                if ';' in line and ('python_version' in line or 'sys_platform' in line):
                    # For now, include all conditional requirements
                    line = line.split(';')[0].strip()
                
                if line:
                    requirements.append(line)
        
        log.info(f"Loaded {len(requirements)} requirements from {req_file}")
        return requirements
        
    except Exception as e:
        log.warn(f"Error loading requirements from {req_file}: {e}")
        return []


def get_version_from_init() -> Optional[str]:
    """
    Extract version from package __init__.py file using AST parsing.
    
    This method is safer than importing the module during setup.
    
    Returns:
        Optional[str]: Version string if found, None otherwise
        
    Example:
        >>> version = get_version_from_init()
        >>> print(f"Package version: {version}")
    """
    init_file = SOURCE_DIR / "__init__.py"
    
    if not init_file.exists():
        log.warn(f"__init__.py not found at {init_file}")
        return None
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__version__':
                        if isinstance(node.value, ast.Str):
                            version = node.value.s
                            log.info(f"Found version in __init__.py: {version}")
                            return version
                        elif isinstance(node.value, ast.Constant):
                            version = node.value.value
                            log.info(f"Found version in __init__.py: {version}")
                            return version
        
        log.warn("No __version__ found in __init__.py")
        return None
        
    except Exception as e:
        log.warn(f"Error parsing __init__.py for version: {e}")
        return None


def get_version_from_git() -> Optional[str]:
    """
    Get version from git tags.
    
    Looks for tags in the format 'v1.2.3' or '1.2.3' and returns
    the latest semantic version.
    
    Returns:
        Optional[str]: Git tag version if available, None otherwise
        
    Example:
        >>> git_version = get_version_from_git()
        >>> if git_version:
        ...     print(f"Git version: {git_version}")
    """
    try:
        # Get the latest git tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            cwd=PACKAGE_DIR,
            timeout=10
        )
        
        if result.returncode == 0:
            tag = result.stdout.strip()
            # Remove 'v' prefix if present
            version = tag.lstrip('v')
            
            # Validate version format (basic semantic versioning)
            if re.match(r'^\d+\.\d+\.\d+', version):
                log.info(f"Found git version: {version}")
                return version
            else:
                log.warn(f"Invalid version format in git tag: {tag}")
                return None
        else:
            log.warn("No git tags found or git not available")
            return None
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warn(f"Error getting version from git: {e}")
        return None


def get_dynamic_version() -> str:
    """
    Get package version from multiple sources with fallback strategy.
    
    Priority order:
    1. __init__.py __version__ attribute
    2. Git tags
    3. Environment variable AUTOERP_VERSION
    4. Fallback to default version
    
    Returns:
        str: Package version string
        
    Example:
        >>> version = get_dynamic_version()
        >>> print(f"Dynamic version: {version}")
    """
    # Try to get version from __init__.py first
    version = get_version_from_init()
    if version:
        return version
    
    # Try to get version from git tags
    version = get_version_from_git()
    if version:
        return version
    
    # Try environment variable
    version = os.environ.get('AUTOERP_VERSION')
    if version:
        log.info(f"Using version from environment: {version}")
        return version
    
    # Fallback version
    fallback_version = "1.0.0"
    log.warn(f"Using fallback version: {fallback_version}")
    return fallback_version


def get_long_description() -> str:
    """
    Generate long description by combining README and CHANGES files.
    
    Returns:
        str: Combined long description for PyPI
        
    Example:
        >>> desc = get_long_description()
        >>> print(f"Description length: {len(desc)}")
    """
    readme_content = read_file(README_FILE)
    changes_content = read_file(CHANGES_FILE)
    
    # Combine README and CHANGES with proper formatting
    long_desc_parts = []
    
    if readme_content:
        long_desc_parts.append(readme_content)
    
    if changes_content:
        long_desc_parts.append("\n\n## Changelog\n\n")
        long_desc_parts.append(changes_content)
    
    if not long_desc_parts:
        # Fallback description
        long_desc_parts.append(SHORT_DESCRIPTION)
        long_desc_parts.append("\n\nA comprehensive Enterprise Resource Planning system built with modern Python technologies.")
    
    return "".join(long_desc_parts)


# ==================== DEPENDENCY MANAGEMENT ====================

def get_core_requirements() -> List[str]:
    """
    Get core runtime requirements for the package.
    
    Returns:
        List[str]: List of core requirements
    """
    # Try to load from requirements file first
    requirements = load_requirements('base.txt')
    
    if not requirements:
        # Fallback to hardcoded core requirements
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "streamlit>=1.28.0",
            "pydantic>=2.4.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "pandas>=2.0.0",
            "plotly>=5.16.0",
            "numpy>=1.24.0",
            "python-multipart>=0.0.6",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "aiofiles>=23.2.1",
            "jinja2>=3.1.2",
            "openpyxl>=3.1.0",
            "celery>=5.3.0",
            "redis>=5.0.0",
            "psycopg2-binary>=2.9.7"
        ]
    
    return requirements


def get_development_requirements() -> List[str]:
    """
    Get development dependencies.
    
    Returns:
        List[str]: List of development requirements
    """
    dev_requirements = load_requirements('dev.txt')
    
    if not dev_requirements:
        dev_requirements = [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
            "tox>=4.11.0",
            "coverage>=7.3.0",
            "bandit>=1.7.5",
            "safety>=2.3.0"
        ]
    
    return dev_requirements


def get_test_requirements() -> List[str]:
    """
    Get testing dependencies.
    
    Returns:
        List[str]: List of test requirements
    """
    test_requirements = load_requirements('test.txt')
    
    if not test_requirements:
        test_requirements = [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "pytest-mock>=3.11.1",
            "factory-boy>=3.3.0",
            "faker>=19.6.0",
            "httpx>=0.24.0",
            "pytest-xdist>=3.3.1"
        ]
    
    return test_requirements


def get_documentation_requirements() -> List[str]:
    """
    Get documentation dependencies.
    
    Returns:
        List[str]: List of documentation requirements
    """
    docs_requirements = load_requirements('docs.txt')
    
    if not docs_requirements:
        docs_requirements = [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-openapi>=0.8.1",
            "myst-parser>=2.0.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "sphinx-copybutton>=0.5.2"
        ]
    
    return docs_requirements


# ==================== CUSTOM SETUP COMMANDS ====================

class CustomBuildPy(build_py):
    """Custom build command with additional processing."""
    
    description = "Build Python modules with custom processing"
    
    def run(self):
        """Run the build process with custom steps."""
        log.info("Running custom build process...")
        
        # Run standard build first
        super().run()
        
        # Add custom build steps here
        self._process_templates()
        self._generate_version_file()
        
        log.info("Custom build process completed")
    
    def _process_templates(self):
        """Process any template files."""
        log.info("Processing template files...")
        # Template processing logic would go here
    
    def _generate_version_file(self):
        """Generate version information file."""
        version = get_dynamic_version()
        version_file = Path(self.build_lib) / PACKAGE_NAME / '_version.py'
        
        version_content = f'''"""
Version information for {PACKAGE_NAME}.
This file is automatically generated during build.
"""

__version__ = "{version}"
__build_date__ = "{subprocess.getoutput('date')}"
__build_system__ = "setuptools"
'''
        
        try:
            version_file.parent.mkdir(parents=True, exist_ok=True)
            with open(version_file, 'w') as f:
                f.write(version_content)
            log.info(f"Generated version file: {version_file}")
        except Exception as e:
            log.warn(f"Could not generate version file: {e}")


class CustomInstall(install):
    """Custom install command with post-installation steps."""
    
    description = "Install package with custom post-install steps"
    
    def run(self):
        """Run installation with custom steps."""
        log.info("Running custom installation...")
        
        # Run standard installation
        super().run()
        
        # Post-installation steps
        self._post_install()
        
        log.info("Custom installation completed")
    
    def _post_install(self):
        """Execute post-installation tasks."""
        log.info("Running post-installation tasks...")
        
        # Print ASCII logo
        self._print_logo()
        
        # Create default configuration
        self._create_default_config()
    
    def _print_logo(self):
        """Print AutoERP ASCII logo."""
        logo = """
        
   █████╗ ██╗   ██╗████████╗ ██████╗ ███████╗██████╗ ██████╗ 
  ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗██╔════╝██╔══██╗██╔══██╗
  ███████║██║   ██║   ██║   ██║   ██║█████╗  ██████╔╝██████╔╝
  ██╔══██║██║   ██║   ██║   ██║   ██║██╔══╝  ██╔══██╗██╔═══╝ 
  ██║  ██║╚██████╔╝   ██║   ╚██████╔╝███████╗██║  ██║██║     
  ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝     
                                                              
        Enterprise Resource Planning System v{version}
        
        🎉 Installation completed successfully!
        
        Quick Start:
        • python -m autoerp serve    # Start API server
        • python -m autoerp ui       # Launch web interface
        • python -m autoerp help     # Show help
        
        Documentation: https://docs.autoerp.com
        Support: https://github.com/autoerp/autoerp/issues
        
        """.format(version=get_dynamic_version())
        
        print(logo)
    
    def _create_default_config(self):
        """Create default configuration directory."""
        try:
            config_dir = Path.home() / '.autoerp'
            config_dir.mkdir(exist_ok=True)
            
            config_file = config_dir / 'config.json'
            if not config_file.exists():
                default_config = {
                    "database": {
                        "engine": "sqlite",
                        "database": str(config_dir / "autoerp.db")
                    },
                    "security": {
                        "secret_key": "change-me-in-production"
                    }
                }
                
                with open(config_file, 'w') as f:
                    import json
                    json.dump(default_config, f, indent=2)
                
                log.info(f"Created default config at: {config_file}")
                
        except Exception as e:
            log.warn(f"Could not create default config: {e}")


class CustomDevelop(develop):
    """Custom develop command for development installations."""
    
    description = "Install package in development mode"
    
    def run(self):
        """Run development installation."""
        log.info("Installing in development mode...")
        super().run()
        self._setup_development_environment()
    
    def _setup_development_environment(self):
        """Setup development environment."""
        log.info("Setting up development environment...")
        
        # Install pre-commit hooks if available
        try:
            subprocess.run(['pre-commit', 'install'], check=True)
            log.info("Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warn("Could not install pre-commit hooks")


class CustomTest(TestCommand):
    """Custom test command with additional options."""
    
    description = "Run test suite with coverage"
    user_options = [
        ('pytest-args=', 'a', "Arguments to pass to pytest"),
        ('coverage', 'c', "Run with coverage reporting"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        TestCommand.initialize_options(self)
        self.pytest_args = []
        self.coverage = False
    
    def finalize_options(self):
        """Finalize command options."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    
    def run_tests(self):
        """Run the test suite."""
        import pytest
        
        args = ['tests/']
        
        if self.coverage:
            args.extend(['--cov=autoerp', '--cov-report=html', '--cov-report=term'])
        
        if self.pytest_args:
            args.extend(self.pytest_args.split())
        
        errno = pytest.main(args)
        sys.exit(errno)


class CustomClean(clean):
    """Enhanced clean command."""
    
    description = "Clean build files and directories"
    
    def run(self):
        """Run cleaning process."""
        super().run()
        
        # Additional directories to clean
        clean_dirs = [
            'build',
            'dist',
            '*.egg-info',
            '.pytest_cache',
            '.coverage',
            'htmlcov',
            '.tox',
            '__pycache__'
        ]
        
        for pattern in clean_dirs:
            self._remove_pattern(pattern)
    
    def _remove_pattern(self, pattern: str):
        """Remove files matching pattern."""
        import glob
        import shutil
        
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    log.info(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    log.info(f"Removed file: {path}")
            except OSError as e:
                log.warn(f"Could not remove {path}: {e}")


class BuildPackage(Command):
    """Command to build distribution packages."""
    
    description = "Build source and wheel distributions"
    user_options = []
    
    def initialize_options(self):
        """Initialize options."""
        pass
    
    def finalize_options(self):
        """Finalize options."""
        pass
    
    def run(self):
        """Build packages."""
        log.info("Building distribution packages...")
        
        # Clean previous builds
        self._clean_dist()
        
        # Build source distribution
        self.run_command('sdist')
        
        # Build wheel distribution
        try:
            self.run_command('bdist_wheel')
        except SystemExit:
            log.warn("Wheel build failed, continuing with source distribution only")
        
        log.info("Package building completed")
    
    def _clean_dist(self):
        """Clean dist directory."""
        import shutil
        
        dist_dir = PACKAGE_DIR / 'dist'
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
            log.info("Cleaned dist directory")


class PublishPackage(Command):
    """Command to publish package to PyPI."""
    
    description = "Publish package to PyPI"
    user_options = [
        ('test-pypi', 't', 'Upload to test PyPI instead'),
        ('username=', 'u', 'PyPI username'),
        ('password=', 'p', 'PyPI password'),
    ]
    
    def initialize_options(self):
        """Initialize options."""
        self.test_pypi = False
        self.username = None
        self.password = None
    
    def finalize_options(self):
        """Finalize options."""
        pass
    
    def run(self):
        """Publish package."""
        log.info("Publishing package...")
        
        # Build package first
        self.run_command('build_package')
        
        # Upload to PyPI
        try:
            import twine.commands.upload
            
            repository = 'testpypi' if self.test_pypi else 'pypi'
            
            upload_args = [
                '--repository', repository,
                'dist/*'
            ]
            
            if self.username:
                upload_args.extend(['--username', self.username])
            
            if self.password:
                upload_args.extend(['--password', self.password])
            
            twine.commands.upload.main(upload_args)
            log.info("Package published successfully")
            
        except ImportError:
            log.error("Twine not installed. Install with: pip install twine")
        except Exception as e:
            log.error(f"Publishing failed: {e}")


# ==================== SETUP CONFIGURATION ====================

def get_package_data() -> Dict[str, List[str]]:
    """
    Get package data files to include in distribution.
    
    Returns:
        Dict[str, List[str]]: Package data specification
    """
    return {
        PACKAGE_NAME: [
            '*.json',
            '*.yaml', 
            '*.yml',
            '*.ini',
            '*.conf',
            '*.cfg',
            'templates/*',
            'templates/**/*',
            'static/*',
            'static/**/*',
            'migrations/*',
            'migrations/**/*',
            'locale/*/LC_MESSAGES/*'
        ]
    }


def get_classifiers() -> List[str]:
    """
    Get comprehensive PyPI classifiers.
    
    Returns:
        List[str]: List of classifier strings
    """
    return [
        # Development Status
        'Development Status :: 4 - Beta',
        
        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Manufacturing',
        
        # Topic Classification
        'Topic :: Office/Business',
        'Topic :: Office/Business :: Financial',
        'Topic :: Office/Business :: Financial :: Accounting',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Database',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming Language
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        
        # Operating System
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        
        # Framework
        'Framework :: FastAPI',
        'Framework :: AsyncIO',
        
        # Environment
        'Environment :: Web Environment',
        'Environment :: Console',
        
        # Natural Language
        'Natural Language :: English',
        
        # Typing
        'Typing :: Typed'
    ]


def get_entry_points() -> Dict[str, Any]:
    """
    Get entry points for console scripts, GUI scripts, and plugins.
    
    Returns:
        Dict[str, Any]: Entry points specification
    """
    return {
        'console_scripts': [
            'autoerp = autoerp.__main__:main',
            'autoerp-server = autoerp.api:run_server',
            'autoerp-migrate = autoerp.cli:migrate_command',
            'autoerp-admin = autoerp.cli:admin_command',
        ],
        'gui_scripts': [
            'autoerp-ui = autoerp.ui:main_ui',
            'autoerp-dashboard = autoerp.ui:dashboard_main',
        ],
        'autoerp.plugins': [
            'core = autoerp.plugins.core:CorePlugin',
            'accounting = autoerp.plugins.accounting:AccountingPlugin',
            'inventory = autoerp.plugins.inventory:InventoryPlugin',
            'crm = autoerp.plugins.crm:CRMPlugin',
        ]
    }


def get_extras_require() -> Dict[str, List[str]]:
    """
    Get optional dependencies for different use cases.
    
    Returns:
        Dict[str, List[str]]: Extra requirements specification
    """
    return {
        'dev': get_development_requirements(),
        'test': get_test_requirements(),
        'docs': get_documentation_requirements(),
        'postgres': ['psycopg2-binary>=2.9.7'],
        'mysql': ['pymysql>=1.1.0'],
        'redis': ['redis>=5.0.0', 'hiredis>=2.2.0'],
        'async': ['aioredis>=2.0.0', 'asyncpg>=0.28.0'],
        'monitoring': ['prometheus-client>=0.17.0', 'sentry-sdk>=1.32.0'],
        'all': (
            get_development_requirements() + 
            get_test_requirements() + 
            get_documentation_requirements() +
            ['psycopg2-binary>=2.9.7', 'redis>=5.0.0', 'prometheus-client>=0.17.0']
        )
    }


# ==================== MAIN SETUP CALL ====================

def main():
    """Main setup function."""
    
    # Get dynamic values
    version = get_dynamic_version()
    long_description = get_long_description()
    core_requirements = get_core_requirements()
    
    setup(
        # Basic package information
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR_NAME,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER_NAME,
        maintainer_email=MAINTAINER_EMAIL,
        description=SHORT_DESCRIPTION,
        long_description=long_description,
        long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
        license=LICENSE_NAME,
        
        # URLs
        url=PACKAGE_URL,
        project_urls={
            'Documentation': DOCUMENTATION_URL,
            'Repository': REPOSITORY_URL,
            'Bug Tracker': BUG_TRACKER_URL,
            'Changelog': CHANGELOG_URL,
        },
        
        # Package discovery and data
        packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
        package_data=get_package_data(),
        include_package_data=True,
        zip_safe=False,
        
        # Requirements and compatibility
        python_requires='>=3.8',
        install_requires=core_requirements,
        extras_require=get_extras_require(),
        
        # Entry points
        entry_points=get_entry_points(),
        
        # Classification
        classifiers=get_classifiers(),
        keywords=['erp', 'enterprise', 'resource', 'planning', 'business', 'management', 'fastapi', 'streamlit'],
        
        # Custom commands
        cmdclass={
            'build_py': CustomBuildPy,
            'install': CustomInstall,
            'develop': CustomDevelop,
            'test': CustomTest,
            'clean': CustomClean,
            'build_package': BuildPackage,
            'publish': PublishPackage,
        },
    )


if __name__ == '__main__':
    main()