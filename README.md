Here’s the full English version of your README.md:

````markdown
# README.md

<div align="center">

# 🏢 AutoERP

**Modern Enterprise Resource Planning System**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Build Status](https://github.com/autoerp/autoerp/workflows/CI/badge.svg)](https://github.com/autoerp/autoerp/actions)
[![Coverage](https://codecov.io/gh/autoerp/autoerp/branch/main/graph/badge.svg)](https://codecov.io/gh/autoerp/autoerp)
[![Documentation](https://readthedocs.org/projects/autoerp/badge/?version=latest)](https://autoerp.readthedocs.io)

*A full-featured ERP system with hexagonal architecture, modern REST API, and intuitive user interface*

[🚀 Installation](#-installation) • [📖 Documentation](#-documentation) • [🎯 Features](#-features) • [🛠️ Development](#-development)

</div>

---
## 📑 Table of Contents
- [About](#-about)
- [Features](#-main-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Plugins](#-plugin-system)
- [Security](#-security)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 About

**AutoERP** is a modern enterprise resource planning (ERP) system designed with a hexagonal architecture for maximum flexibility and maintainability. It combines a robust REST API powered by FastAPI with a modern user interface using Streamlit.

### 🎯 Main Features

#### 💼 Business Modules
- **📊 Finance & Accounting** - Full financial management with advanced reporting
- **👥 CRM (Customer Relationship Management)** - Customer management with sales pipeline
- **📦 Inventory Management** - Real-time stock tracking with automatic alerts
- **🧑‍💼 Human Resources** - Employee management, payroll, and performance reviews
- **📋 Project Management** - Project planning and tracking with Gantt charts
- **📈 Business Intelligence** - Advanced dashboards and analytics

#### 🏗️ Architecture & Technologies
- **🔶 Hexagonal Architecture** - Clear separation of concerns
- **⚡ FastAPI** - Modern REST API with automatic documentation
- **🎨 Streamlit** - Responsive and interactive user interface
- **🗄️ Multi-Database Support** - SQLite, PostgreSQL, MySQL
- **🔄 Redis Cache** - Optimized performance with caching
- **🔌 Plugin System** - Extensible architecture
- **🔐 Security** - JWT authentication, role management, audit trails

## 🗺️ Roadmap

- [x] REST API with FastAPI
- [x] UI with Streamlit
- [x] Finance, CRM, Inventory modules
- [ ] Full PostgreSQL/MySQL support
- [ ] Advanced BI integration
- [ ] Mobile support (React Native)
- [ ] Multi-tenant SaaS deployable on Kubernetes

## 📜 License

Distributed under the **MIT** license. See [LICENSE](LICENSE) for more details.

## 🚀 Installation

### Quick Installation

```bash
pip install autoerp
````

Installation with Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/autoerp/autoerp.git
cd autoerp

# Install with Poetry
poetry install

# Activate virtual environment
poetry shell
```

Installation with Docker

```bash
# Build the image
docker build -t autoerp .

# Run the container
docker run -p 8000:8000 -p 8501:8501 autoerp
```

### 🎯 Quick Start

1. Initialize the Project

```bash
# Initialize the database and configuration
autoerp init

# Or with custom parameters
autoerp init --db-url postgresql://user:pass@localhost/autoerp --redis-url redis://localhost:6379
```

2. Start Services

**REST API (FastAPI)**

```bash
# Start the API on port 8000
autoerp-server

# Or with custom configuration
uvicorn autoerp.api:app --host 0.0.0.0 --port 8000 --reload
```

**User Interface (Streamlit)**

```bash
# Start the UI on port 8501
autoerp-ui

# Or directly with Streamlit
streamlit run autoerp/ui.py
```

**CLI (Command Line Interface)**

```bash
# Use the CLI
autoerp --help
```

## 🤝 Contributing

Contributions are welcome!
Please follow these steps:

1. Fork the project
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📊 Usage Examples

```bash
autoerp users create --email admin@example.com --password admin123 --role admin
autoerp finance create-invoice --customer-id 1 --amount 1000.00
autoerp inventory add-product --name "Laptop" --sku LAP001 --price 999.99
```

3. Access Interfaces

* API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
* User Interface: [http://localhost:8501](http://localhost:8501)
* API Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Programmatic Usage

```python
from autoerp import AutoERP, config
from autoerp.core import UserService, FinanceService, InventoryService

# Initialization
erp = AutoERP(config_path="config/production.yaml")

# Business services
user_service = UserService(erp.get_repository('users'))
finance_service = FinanceService(erp.get_repository('finance'))
inventory_service = InventoryService(erp.get_repository('inventory'))

# Create a user
user = user_service.create_user(
    email="john.doe@example.com",
    password="secure_password",
    full_name="John Doe",
    role="manager"
)

# Create an invoice
invoice = finance_service.create_invoice(
    customer_id=user.id,
    items=[
        {"product_id": 1, "quantity": 2, "price": 99.99},
        {"product_id": 2, "quantity": 1, "price": 149.99}
    ],
    tax_rate=0.20
)

# Inventory management
product = inventory_service.add_product(
    name="MacBook Pro",
    sku="MBP-2023-001", 
    category="Electronics",
    price=2499.99,
    stock_quantity=50
)

# Update stock
inventory_service.update_stock(product.id, quantity=-5, reason="Sale")
```

### Plugin Configuration

```python
from autoerp.plugins import PluginManager

# Load custom plugins
plugin_manager = PluginManager()
plugin_manager.load_plugin("custom_reporting")
plugin_manager.load_plugin("advanced_crm")

# Use a plugin
reporting_plugin = plugin_manager.get_plugin("custom_reporting")
report = reporting_plugin.generate_sales_report(start_date="2023-01-01", end_date="2023-12-31")
```

## 🛠️ Configuration

**Base Configuration (config.yaml)**

```yaml
database:
  url: "postgresql://user:password@localhost:5432/autoerp"
  echo: false
  pool_size: 20

redis:
  url: "redis://localhost:6379/0"
  decode_responses: true

security:
  secret_key: "your-super-secret-key-change-in-production"
  access_token_expire_minutes: 30
  algorithm: "HS256"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

features:
  enable_audit_log: true
  enable_cache: true
  enable_celery: false
```

**Environment Variables**

```bash
# Database
export AUTOERP_DB_URL="postgresql://user:pass@localhost/autoerp"
export AUTOERP_REDIS_URL="redis://localhost:6379"

# Security
export AUTOERP_SECRET_KEY="your-production-secret-key"
export AUTOERP_DEBUG=false

# Advanced configuration
export AUTOERP_LOG_LEVEL="INFO"
export AUTOERP_ENABLE_CORS=true
```

## 🏗️ Architecture

**Hexagonal Architecture Overview**

```
┌───────────────────────────┐
│       INTERFACE LAYER     │
├─────────────┬─────────────┤
│ FastAPI     │ Streamlit   │ CLI
└─────────────┴─────────────┘
          │
┌───────────────────────────┐
│     APPLICATION LAYER     │
├─────────────┬─────────────┤
│ UserService │ FinanceService │ InventoryService │ ...
└─────────────┴─────────────┘
          │
┌───────────────────────────┐
│       DOMAIN LAYER        │
├─────────────┬─────────────┤
│ User │ Invoice │ Product │ Customer │ ...
└─────────────┴─────────────┘
          │
┌───────────────────────────┐
│     INFRASTRUCTURE LAYER  │
├─────────────┬─────────────┤
│ PostgreSQL  │ Redis      │ File System
└─────────────┴─────────────┘
```

## 🔌 Plugin System

**Create a Custom Plugin**

```python
# plugins/custom_analytics.py
from autoerp.plugins.base import BasePlugin

class CustomAnalyticsPlugin(BasePlugin):
    name = "custom_analytics"
    version = "1.0.0"
    description = "Advanced analytics plugin"
    
    def initialize(self):
        """Plugin initialization"""
        self.setup_dashboard_widgets()
        self.register_api_endpoints()
    
    def setup_dashboard_widgets(self):
        """Add dashboard widgets"""
        pass
    
    def register_api_endpoints(self):
        """Register new API endpoints"""
        pass
    
    def generate_advanced_report(self, report_type: str):
        """Generate advanced reports"""
        return {"status": "success", "data": []}
```

## 📊 Dashboards

The Streamlit UI provides interactive dashboards for:

* 📈 Finance Dashboard – KPIs, revenue charts, trend analysis
* 👥 CRM Dashboard – Sales pipeline, client activities, conversion rates
* 📦 Inventory Dashboard – Stock levels, movements, alerts
* 🧑‍💼 HR Dashboard – Workforce, performance, leave tracking
* 📋 Project Dashboard – Progress, resources, deadlines

## 🔐 Security

* 🔑 JWT Authentication – Secure tokens with expiration
* 🛡️ Role Management – RBAC (Role-Based Access Control)
* 📝 Audit Trails – Full action traceability
* 🔒 Data Encryption – Protection of sensitive data
* 🚫 CSRF Protection – Safeguards against cross-site attacks
* ⚡ Rate Limiting – Protection against DoS attacks

```

If you want, I can also create a **fully polished GitHub-ready version with badges, TOC links, and proper English formatting**. It would look professional for a real repository. Do you want me to do that?
```
