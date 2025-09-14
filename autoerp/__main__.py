# autoerp/__main__.py
"""
🏢 AutoERP - Point d'entrée principal du package
═══════════════════════════════════════════════════════════════════════════════

Module principal permettant l'exécution du package via:
    python -m autoerp <command>

Ce module gère tous les points d'entrée CLI et coordonne les différents
services et fonctionnalités d'AutoERP.

Auteur: AutoERP Team
Version: 0.1.0
Licence: MIT
"""

import sys
import argparse
from typing import List, Optional

# Importation des fonctions depuis __init__.py
try:
    from . import (
        _start_api_server,
        _start_ui_server, 
        _run_migrations,
        _create_admin_user,
        _backup_database,
        _restore_database,
        create_sample_config,
        validate_installation,
        get_system_info,
        _show_help,
        __version__
    )
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("💡 Assurez-vous qu'AutoERP est correctement installé")
    sys.exit(1)


def _print_welcome_message() -> None:
    """
    Affiche le message de bienvenue avec les informations de base
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    🏢 AutoERP System                        ║")
    print("║            Système de Planification des Ressources         ║")
    print("║                      d'Entreprise                          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Version: {__version__:<20} Licence: MIT                ║")
    print("║                                                              ║")
    print("║  🚀 Utilisation:                                             ║")
    print("║     python -m autoerp <command>                             ║")
    print("║                                                              ║")
    print("║  📖 Commandes principales:                                   ║")
    print("║     serve      - Démarrer le serveur API                   ║")
    print("║     ui         - Lancer l'interface utilisateur            ║")
    print("║     migrate    - Appliquer les migrations BDD              ║")
    print("║     config     - Créer un fichier de configuration         ║")
    print("║     help       - Afficher l'aide complète                  ║")
    print("║                                                              ║")
    print("║  💡 Pour plus d'aide: python -m autoerp help                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")


def _print_error_message(command: str) -> None:
    """
    Affiche un message d'erreur pour une commande inconnue
    
    Args:
        command: La commande invalide entrée par l'utilisateur
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                        ❌ ERREUR                            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Commande inconnue: '{command}'                              ║")
    print("║                                                              ║")
    print("║  📖 Commandes disponibles:                                   ║")
    print("║     serve, ui, migrate, create-admin, backup, restore,      ║")
    print("║     config, validate, info, version, help                   ║")
    print("║                                                              ║")
    print("║  💡 Utilisez 'python -m autoerp help' pour plus d'infos     ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("")


def _execute_serve_command(args: List[str]) -> None:
    """
    Exécute la commande 'serve' pour démarrer le serveur API
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("🚀 Démarrage du serveur API AutoERP...")
    print("📡 L'API sera disponible sur: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🛑 Appuyez sur Ctrl+C pour arrêter le serveur")
    print("-" * 60)
    
    try:
        _start_api_server()
    except KeyboardInterrupt:
        print("\n🛑 Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage du serveur: {e}")
        sys.exit(1)


def _execute_ui_command(args: List[str]) -> None:
    """
    Exécute la commande 'ui' pour démarrer l'interface utilisateur
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("🎨 Démarrage de l'interface utilisateur AutoERP...")
    print("🌐 L'interface sera disponible sur: http://localhost:8080")
    print("🛑 Appuyez sur Ctrl+C pour arrêter l'interface")
    print("-" * 60)
    
    try:
        _start_ui_server()
    except KeyboardInterrupt:
        print("\n🛑 Interface arrêtée par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors du démarrage de l'interface: {e}")
        sys.exit(1)


def _execute_migrate_command(args: List[str]) -> None:
    """
    Exécute la commande 'migrate' pour appliquer les migrations
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("🔧 Application des migrations de base de données...")
    print("-" * 60)
    
    try:
        _run_migrations()
        print("✅ Migrations appliquées avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors des migrations: {e}")
        sys.exit(1)


def _execute_create_admin_command(args: List[str]) -> None:
    """
    Exécute la commande 'create-admin' pour créer un utilisateur administrateur
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("👤 Création d'un utilisateur administrateur...")
    print("-" * 60)
    
    try:
        _create_admin_user()
        print("✅ Utilisateur administrateur créé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de la création de l'administrateur: {e}")
        sys.exit(1)


def _execute_backup_command(args: List[str]) -> None:
    """
    Exécute la commande 'backup' pour sauvegarder la base de données
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("💾 Création d'une sauvegarde de la base de données...")
    print("-" * 60)
    
    try:
        backup_path = _backup_database()
        print(f"✅ Sauvegarde créée: {backup_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        sys.exit(1)


def _execute_restore_command(args: List[str]) -> None:
    """
    Exécute la commande 'restore' pour restaurer une sauvegarde
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("📥 Restauration de la base de données...")
    print("-" * 60)
    
    # Vérifier si un fichier de sauvegarde est spécifié
    if len(args) < 1:
        print("❌ Erreur: Veuillez spécifier le fichier de sauvegarde")
        print("💡 Utilisation: python -m autoerp restore <fichier_sauvegarde>")
        sys.exit(1)
    
    backup_file = args[0]
    try:
        _restore_database(backup_file)
        print("✅ Base de données restaurée avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors de la restauration: {e}")
        sys.exit(1)


def _execute_config_command(args: List[str]) -> None:
    """
    Exécute la commande 'config' pour créer un fichier de configuration
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("⚙️ Création du fichier de configuration par défaut...")
    print("-" * 60)
    
    try:
        config_path = create_sample_config()
        print(f"✅ Fichier de configuration créé: {config_path}")
        print("💡 Modifiez ce fichier selon vos besoins avant le premier démarrage")
    except Exception as e:
        print(f"❌ Erreur lors de la création de la configuration: {e}")
        sys.exit(1)


def _execute_validate_command(args: List[str]) -> None:
    """
    Exécute la commande 'validate' pour vérifier l'installation
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("🔍 Validation de l'installation AutoERP...")
    print("-" * 60)
    
    try:
        is_valid = validate_installation()
        if is_valid:
            print("✅ Installation validée avec succès!")
            print("🚀 AutoERP est prêt à être utilisé")
        else:
            print("❌ Des problèmes ont été détectés dans l'installation")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors de la validation: {e}")
        sys.exit(1)


def _execute_info_command(args: List[str]) -> None:
    """
    Exécute la commande 'info' pour afficher les informations système
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("ℹ️ Informations système AutoERP:")
    print("=" * 60)
    
    try:
        system_info = get_system_info()
        print(system_info)
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des informations: {e}")
        sys.exit(1)


def _execute_version_command(args: List[str]) -> None:
    """
    Exécute la commande 'version' pour afficher la version
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    📋 VERSION AutoERP                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Version: {__version__:<48}║")
    print("║  Licence: MIT                                                ║")
    print("║  Python:  >= 3.8                                            ║")
    print("║                                                              ║")
    print("║  🔗 Liens utiles:                                            ║")
    print("║     • Documentation: https://autoerp.readthedocs.io          ║")
    print("║     • Code source: https://github.com/autoerp/autoerp       ║")
    print("║     • Issues: https://github.com/autoerp/autoerp/issues     ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def _execute_help_command(args: List[str]) -> None:
    """
    Exécute la commande 'help' pour afficher l'aide complète
    
    Args:
        args: Arguments additionnels passés à la commande
    """
    _show_help()


def main() -> None:
    """
    Fonction principale - Point d'entrée du module
    
    Analyse les arguments de ligne de commande et exécute la commande appropriée.
    Gère les cas d'erreur et affiche les messages d'aide nécessaires.
    """
    # Dictionnaire de correspondance commandes -> fonctions
    commands = {
        'serve': _execute_serve_command,
        'ui': _execute_ui_command,
        'migrate': _execute_migrate_command,
        'create-admin': _execute_create_admin_command,
        'backup': _execute_backup_command,
        'restore': _execute_restore_command,
        'config': _execute_config_command,
        'validate': _execute_validate_command,
        'info': _execute_info_command,
        'version': _execute_version_command,
        'help': _execute_help_command
    }
    
    # Récupération des arguments (sans le nom du script)
    args = sys.argv[1:]
    
    # Cas 1: Aucun argument fourni → Message de bienvenue
    if not args:
        _print_welcome_message()
        return
    
    # Cas 2: Premier argument est la commande
    command = args[0].lower()
    remaining_args = args[1:]  # Arguments restants pour la commande
    
    # Cas 3: Commande reconnue → Exécution
    if command in commands:
        try:
            commands[command](remaining_args)
        except KeyboardInterrupt:
            print("\n🛑 Opération interrompue par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {e}")
            sys.exit(1)
    else:
        # Cas 4: Commande inconnue → Message d'erreur
        _print_error_message(command)
        sys.exit(1)


# Point d'entrée principal du module
if __name__ == "__main__":
    main()