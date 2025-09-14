#!/usr/bin/env python3
# create_admin.py
"""
👤 AutoERP - Admin User Creation Helper
═══════════════════════════════════════════════════════════════════════════════

Script autonome pour créer un utilisateur administrateur
Utile pour la configuration initiale ou la récupération d'accès

Usage:
    python create_admin.py
    python create_admin.py --username admin --email admin@company.com
"""

import sys
import os
import argparse
import getpass
from pathlib import Path

# Ajouter autoerp au path si nécessaire
sys.path.insert(0, str(Path(__file__).parent))

try:
    from autoerp.core import AutoERPCore, logger
except ImportError:
    print("❌ Erreur: Impossible d'importer AutoERP")
    print("💡 Assurez-vous d'être dans le bon répertoire ou que AutoERP est installé")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Créer un utilisateur administrateur AutoERP")
    parser.add_argument("--username", "-u", help="Nom d'utilisateur admin")
    parser.add_argument("--email", "-e", help="Email admin")  
    parser.add_argument("--password", "-p", help="Mot de passe admin")
    parser.add_argument("--full-name", "-n", help="Nom complet")
    parser.add_argument("--force", "-f", action="store_true", help="Forcer la création")
    
    args = parser.parse_args()
    
    print("🏢 AutoERP - Création d'Administrateur")
    print("=" * 40)
    
    try:
        # Initialiser AutoERP
        app = AutoERPCore()
        app.init_database()
        
        # Collecter les informations
        username = args.username or input("👤 Nom d'utilisateur admin [admin]: ") or "admin"
        email = args.email or input("📧 Email admin [admin@autoerp.com]: ") or "admin@autoerp.com"
        full_name = args.full_name or input("📝 Nom complet (optionnel): ") or None
        
        if args.password:
            password = args.password
        else:
            password = getpass.getpass("🔒 Mot de passe admin: ")
            if not password:
                password = "admin123"
                print("⚠️ Utilisation du mot de passe par défaut: admin123")
        
        # Validation basique
        if len(password) < 6:
            print("❌ Le mot de passe doit contenir au moins 6 caractères")
            sys.exit(1)
        
        if "@" not in email:
            print("❌ Format d'email invalide")
            sys.exit(1)
        
        # Confirmation
        print(f"\n📋 Création de l'utilisateur admin:")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Nom complet: {full_name or 'N/A'}")
        
        if not args.force:
            confirm = input("\n✅ Confirmer la création? [y/N]: ")
            if confirm.lower() != 'y':
                print("❌ Création annulée")
                sys.exit(0)
        
        # Créer l'administrateur
        admin_user = app.create_admin_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name
        )
        
        print("\n🎉 Administrateur créé avec succès!")
        print(f"👤 Username: {admin_user.username}")
        print(f"📧 Email: {admin_user.email}")
        print(f"🔐 Rôle: {admin_user.role}")
        print(f"📅 Créé le: {admin_user.created_at}")
        
        print("\n🚀 Prochaines étapes:")
        print("   1. Démarrez l'API: python -m autoerp runserver")
        print("   2. Démarrez l'UI: python -m autoerp ui")
        print(f"   3. Connectez-vous avec: {username}")
        
    except ValueError as e:
        print(f"❌ Erreur de validation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        logger.error(f"Admin creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()