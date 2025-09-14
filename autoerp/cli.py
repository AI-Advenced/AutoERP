# autoerp/cli.py
"""
🖥️ AutoERP - CLI Enhanced (Version 1.0.0)
═══════════════════════════════════════════════════════════════════════════════

CORRECTIONS ET AMÉLIORATIONS:
1. ✅ Fix: Commande create-admin ajoutée
2. ✅ Fix: init command accepte maintenant <project_name>
3. ✅ Amélioration: Gestion d'erreurs robuste
4. ✅ Amélioration: Logging structuré
5. ✅ Nouveau: Validation des inputs utilisateur
"""

import typer
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional
import getpass
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import asyncio

# Import du core
try:
    from .core import AutoERPCore, logger, config, AuthenticationService
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from autoerp.core import AutoERPCore, logger, config, AuthenticationService

# Rich console pour affichage coloré
console = Console()
app = typer.Typer(help="🏢 AutoERP - Enterprise Resource Planning CLI")

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 COMMANDE RUNSERVER (EXISTANTE - AMÉLIORÉE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("runserver")
def runserver(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Auto reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Worker processes")
):
    """🚀 Start FastAPI server"""
    console.print(Panel.fit(
        f"🚀 Starting AutoERP API Server\n"
        f"📍 Host: [bold green]{host}[/bold green]\n"
        f"🔌 Port: [bold blue]{port}[/bold blue]\n"
        f"🔄 Reload: [bold yellow]{reload}[/bold yellow]",
        title="🏢 AutoERP API",
        border_style="green"
    ))
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "autoerp.api:app",
            "--host", host,
            "--port", str(port)
        ]
        
        if reload:
            cmd.append("--reload")
        else:
            cmd.extend(["--workers", str(workers)])
        
        subprocess.run(cmd, check=True)
    
    except subprocess.CalledProcessError as e:
        console.print(f"❌ [red]Server startup failed:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Server stopped by user[/yellow]")

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 COMMANDE UI (NOUVELLE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("ui")
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port")
):
    """🎨 Start Streamlit UI"""
    console.print(Panel.fit(
        f"🎨 Starting AutoERP Streamlit UI\n"
        f"🌐 URL: [bold green]http://localhost:{port}[/bold green]\n"
        f"🔧 Use Ctrl+C to stop",
        title="🎨 AutoERP UI",
        border_style="blue"
    ))
    
    try:
        # Chemin vers le fichier ui.py
        ui_file = Path(__file__).parent / "ui.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(ui_file),
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ]
        
        subprocess.run(cmd, check=True)
    
    except subprocess.CalledProcessError as e:
        console.print(f"❌ [red]UI startup failed:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]UI stopped by user[/yellow]")

# ═══════════════════════════════════════════════════════════════════════════════
# 🗄️ COMMANDE MIGRATE (EXISTANTE - AMÉLIORÉE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("migrate")
def migrate():
    """🔧 Run database migrations"""
    console.print(Panel.fit(
        "🔧 Running database migrations...",
        title="🗄️ Database Migration",
        border_style="cyan"
    ))
    
    try:
        with console.status("[bold green]Migrating database..."):
            app_instance = AutoERPCore()
            app_instance.init_database()
        
        console.print("✅ [green]Database migrations completed successfully![/green]")
        logger.info("✅ Database migrations completed")
    
    except Exception as e:
        console.print(f"❌ [red]Migration failed:[/red] {e}")
        logger.error(f"❌ Migration error: {e}")
        raise typer.Exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 🌱 COMMANDE SEED (EXISTANTE - AMÉLIORÉE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("seed")
def seed(
    reset: bool = typer.Option(False, "--reset", help="Reset before seeding")
):
    """🌱 Seed database with sample data"""
    console.print(Panel.fit(
        f"🌱 Seeding database with sample data\n"
        f"🔄 Reset: [bold red]{reset}[/bold red]",
        title="🌿 Data Seeding",
        border_style="yellow"
    ))
    
    try:
        with console.status("[bold green]Seeding data..."):
            app_instance = AutoERPCore()
            app_instance.seed_sample_data(reset=reset)
        
        console.print("✅ [green]Sample data seeded successfully![/green]")
        logger.info("✅ Sample data seeded")
    
    except Exception as e:
        console.print(f"❌ [red]Seeding failed:[/red] {e}")
        logger.error(f"❌ Seeding error: {e}")
        raise typer.Exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 🆕 COMMANDE CREATE-ADMIN (NOUVELLE - CORRECTION PRINCIPALE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("create-admin")
def create_admin(
    username: Optional[str] = typer.Option(None, "--username", "-u", help="Admin username"),
    email: Optional[str] = typer.Option(None, "--email", "-e", help="Admin email"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Admin password"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode")
):
    """
    👤 Create admin user
    
    CORRECTION: Cette commande était complètement manquante
    Permet la création sécurisée d'un utilisateur administrateur
    """
    console.print(Panel.fit(
        "👤 Creating Administrator User",
        title="🔐 Admin Creation",
        border_style="magenta"
    ))
    
    try:
        app_instance = AutoERPCore()
        
        # Mode interactif par défaut
        if interactive and not all([username, email, password]):
            console.print("📝 [blue]Interactive admin creation[/blue]")
            
            # Collecte des informations
            if not username:
                username = Prompt.ask("👤 Admin username", default="admin")
            
            if not email:
                email = Prompt.ask("📧 Admin email", default="admin@autoerp.com")
            
            if not password:
                password = getpass.getpass("🔒 Admin password: ")
                if not password:
                    password = "admin123"
                    console.print("⚠️ [yellow]Using default password: admin123[/yellow]")
            
            # Confirmation
            console.print(f"\n📋 Creating admin user:")
            console.print(f"   Username: [bold]{username}[/bold]")
            console.print(f"   Email: [bold]{email}[/bold]")
            
            if not Confirm.ask("Continue?"):
                console.print("❌ [red]Admin creation cancelled[/red]")
                raise typer.Exit(0)
        
        # Validation des inputs
        if not username or not email or not password:
            console.print("❌ [red]Missing required parameters[/red]")
            console.print("💡 Use --help for usage information")
            raise typer.Exit(1)
        
        # Validation email basique
        if "@" not in email:
            console.print("❌ [red]Invalid email format[/red]")
            raise typer.Exit(1)
        
        # Validation mot de passe
        if len(password) < 6:
            console.print("❌ [red]Password must be at least 6 characters[/red]")
            raise typer.Exit(1)
        
        # Création de l'admin
        with console.status("[bold green]Creating admin user..."):
            admin_user = app_instance.create_admin_user(
                username=username,
                email=email, 
                password=password
            )
        
        console.print("✅ [green]Admin user created successfully![/green]")
        console.print(f"👤 Username: [bold]{admin_user.username}[/bold]")
        console.print(f"📧 Email: [bold]{admin_user.email}[/bold]")
        console.print(f"🔐 Role: [bold]{admin_user.role}[/bold]")
        
        logger.info(f"✅ Admin user created: {username}")
    
    except Exception as e:
        console.print(f"❌ [red]Admin creation failed:[/red] {e}")
        logger.error(f"❌ Admin creation error: {e}")
        raise typer.Exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 🆕 COMMANDE INIT AMÉLIORÉE (CORRECTION PRINCIPALE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("init")
def init(
    project_name: Optional[str] = typer.Argument(None, help="Project name"),
    db_url: Optional[str] = typer.Option(None, "--db-url", help="Database URL"),
    force: bool = typer.Option(False, "--force", help="Force initialization")
):
    """
    🎯 Initialize AutoERP project
    
    CORRECTION: Maintenant accepte <project_name> comme argument
    Exemple: autoerp init my_company
    """
    
    # Détermination du nom de projet
    if not project_name:
        project_name = Prompt.ask("📝 Project name", default="autoerp_project")
    
    console.print(Panel.fit(
        f"🎯 Initializing AutoERP Project\n"
        f"📁 Project: [bold cyan]{project_name}[/bold cyan]\n"
        f"🗄️ Database: [bold yellow]{db_url or 'SQLite default'}[/bold yellow]\n"
        f"⚡ Force: [bold red]{force}[/bold red]",
        title="🏗️ Project Initialization",
        border_style="green"
    ))
    
    try:
        # Création du dossier projet
        project_path = Path.cwd() / project_name
        
        if project_path.exists() and not force:
            console.print("⚠️ [yellow]Project directory already exists![/yellow]")
            if not Confirm.ask("Continue anyway?"):
                console.print("🛑 [red]Initialization cancelled[/red]")
                raise typer.Exit(0)
        
        # Créer le dossier
        project_path.mkdir(exist_ok=True)
        
        with console.status("[bold green]Initializing project..."):
            # Initialiser AutoERP core
            app_instance = AutoERPCore()
            
            # Créer la configuration
            config_path = project_path / "config.yaml"
            config_content = f"""
# AutoERP Configuration - {project_name}
project:
  name: "{project_name}"
  version: "1.0.0"
  created: "{app_instance.get_current_timestamp()}"

database:
  url: "{db_url or 'sqlite:///autoerp.db'}"
  echo: false

api:
  host: "0.0.0.0"
  port: 8000

ui:
  host: "0.0.0.0" 
  port: 8501

security:
  secret_key: "change-this-in-production"
  jwt_expiration_hours: 24
"""
            
            with open(config_path, 'w') as f:
                f.write(config_content.strip())
            
            # Initialiser la base de données
            app_instance.init_database()
            
            # Créer l'utilisateur admin par défaut
            try:
                app_instance.create_admin_user(
                    username="admin",
                    email="admin@autoerp.com",
                    password="admin123"
                )
            except Exception:
                pass  # Admin peut déjà exister
        
        console.print("✅ [green]Project initialized successfully![/green]")
        console.print(f"📁 Project path: [bold]{project_path}[/bold]")
        console.print(f"⚙️ Config file: [bold]{config_path}[/bold]")
        console.print("\n🚀 [blue]Next steps:[/blue]")
        console.print(f"   cd {project_name}")
        console.print("   autoerp runserver  # Start API")
        console.print("   autoerp ui         # Start UI")
        
        logger.info(f"✅ Project {project_name} initialized")
    
    except Exception as e:
        console.print(f"❌ [red]Initialization failed:[/red] {e}")
        logger.error(f"❌ Init error: {e}")
        raise typer.Exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 COMMANDE TEST (EXISTANTE - AMÉLIORÉE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("test")
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Generate coverage")
):
    """🧪 Run tests"""
    console.print(Panel.fit(
        f"🧪 Running AutoERP test suite\n"
        f"📊 Coverage: [bold green]{coverage}[/bold green]\n"
        f"🔍 Verbose: [bold blue]{verbose}[/bold blue]",
        title="🔬 Testing",
        border_style="blue"
    ))
    
    try:
        cmd = [sys.executable, "-m", "pytest", "tests/"]
        
        if coverage:
            cmd.extend(["--cov=autoerp", "--cov-report=term-missing"])
        
        if verbose:
            cmd.append("-v")
        
        subprocess.run(cmd, check=True)
        console.print("✅ [green]All tests passed![/green]")
    
    except subprocess.CalledProcessError:
        console.print("❌ [red]Some tests failed[/red]")
        raise typer.Exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# 📋 COMMANDE VERSION
# ═══════════════════════════════════════════════════════════════════════════════

@app.command("version")
def version():
    """📋 Show version information"""
    console.print(Panel.fit(
        f"🏢 [bold blue]AutoERP[/bold blue] v{config.version}\n"
        f"🐍 Python {sys.version.split()[0]}\n"
        f"📦 Status: [green]Active[/green]",
        title="📋 Version Info",
        border_style="blue"
    ))

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app()