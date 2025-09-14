# Démarrer le serveur
autoerp runserver --host 0.0.0.0 --port 8000

# Lancer les tests avec couverture
autoerp test --coverage --verbose

# Initialiser un projet
autoerp init --db-url postgresql://user:pass@localhost/autoerp

# Voir la version
autoerp version

# Makefile

# Variables
PYTHON := python
PIP := pip
POETRY := poetry
PACKAGE_NAME := autoerp
SRC_DIR := autoerp
TESTS_DIR := tests
DOCS_DIR := docs
VENV_DIR := .venv

# Couleurs pour l'affichage
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.DEFAULT_GOAL := help

# Aide par défaut
help: ## 📖 Afficher l'aide
	@echo "$(GREEN)🏢 AutoERP - Makefile Commands$(NC)"
	@echo "════════════════════════════════════════════"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 INSTALLATION ET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

install: ## 🔧 Installer le package en mode développement
	@echo "$(GREEN)🔧 Installation d'AutoERP en mode développement...$(NC)"
	$(POETRY) install --with dev,docs,testing
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✅ Installation terminée!$(NC)"

install-prod: ## 🏭 Installer pour la production (sans dépendances dev)
	@echo "$(GREEN)🏭 Installation pour la production...$(NC)"
	$(POETRY) install --only main
	@echo "$(GREEN)✅ Installation production terminée!$(NC)"

update: ## 🔄 Mettre à jour les dépendances
	@echo "$(GREEN)🔄 Mise à jour des dépendances...$(NC)"
	$(POETRY) update
	$(POETRY) show --outdated
	@echo "$(GREEN)✅ Dépendances mises à jour!$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 TESTS ET QUALITÉ DU CODE
# ═══════════════════════════════════════════════════════════════════════════════

test: ## 🧪 Lancer tous les tests
	@echo "$(GREEN)🧪 Exécution des tests...$(NC)"
	$(POETRY) run pytest $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✅ Tests terminés! Rapport: htmlcov/index.html$(NC)"

test-fast: ## ⚡ Tests rapides (sans couverture)
	@echo "$(GREEN)⚡ Tests rapides...$(NC)"
	$(POETRY) run pytest $(TESTS_DIR) -x --tb=short
	@echo "$(GREEN)✅ Tests rapides terminés!$(NC)"

test-unit: ## 🔬 Tests unitaires seulement
	@echo "$(GREEN)🔬 Tests unitaires...$(NC)"
	$(POETRY) run pytest $(TESTS_DIR) -m "unit" -v
	@echo "$(GREEN)✅ Tests unitaires terminés!$(NC)"

test-integration: ## 🔗 Tests d'intégration seulement
	@echo "$(GREEN)🔗 Tests d'intégration...$(NC)"
	$(POETRY) run pytest $(TESTS_DIR) -m "integration" -v
	@echo "$(GREEN)✅ Tests d'intégration terminés!$(NC)"

coverage: ## 📊 Générer le rapport de couverture détaillé
	@echo "$(GREEN)📊 Génération du rapport de couverture...$(NC)"
	$(POETRY) run pytest --cov=$(SRC_DIR) --cov-report=html --cov-report=xml --cov-report=term
	@echo "$(GREEN)✅ Rapport généré: htmlcov/index.html$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 FORMATAGE ET LINTING
# ═══════════════════════════════════════════════════════════════════════════════

format: ## 🎨 Formater le code avec black et isort
	@echo "$(GREEN)🎨 Formatage du code...$(NC)"
	$(POETRY) run black $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run isort $(SRC_DIR) $(TESTS_DIR)
	@echo "$(GREEN)✅ Code formaté!$(NC)"

lint: ## 🔍 Vérifier la qualité du code
	@echo "$(GREEN)🔍 Vérification de la qualité du code...$(NC)"
	$(POETRY) run flake8 $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run mypy $(SRC_DIR)
	$(POETRY) run bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	@echo "$(GREEN)✅ Vérifications terminées!$(NC)"

lint-fix: ## 🔧 Corriger automatiquement les problèmes de linting
	@echo "$(GREEN)🔧 Correction automatique...$(NC)"
	$(POETRY) run autopep8 --in-place --recursive $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run black $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run isort $(SRC_DIR) $(TESTS_DIR)
	@echo "$(GREEN)✅ Corrections appliquées!$(NC)"

security: ## 🔒 Vérifier la sécurité du code
	@echo "$(GREEN)🔒 Vérification de sécurité...$(NC)"
	$(POETRY) run bandit -r $(SRC_DIR)
	$(POETRY) run safety check
	@echo "$(GREEN)✅ Vérifications sécuritaires terminées!$(NC)"

pre-commit: ## 🪝 Lancer les hooks pre-commit sur tous les fichiers
	@echo "$(GREEN)🪝 Exécution des hooks pre-commit...$(NC)"
	$(POETRY) run pre-commit run --all-files
	@echo "$(GREEN)✅ Pre-commit terminé!$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 CONSTRUCTION ET DÉPLOIEMENT
# ═══════════════════════════════════════════════════════════════════════════════

build: ## 📦 Construire le package
	@echo "$(GREEN)📦 Construction du package...$(NC)"
	$(POETRY) build
	@echo "$(GREEN)✅ Package construit dans dist/$(NC)"

clean: ## 🧹 Nettoyer les fichiers générés
	@echo "$(GREEN)🧹 Nettoyage...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Nettoyage terminé!$(NC)"

publish-test: build ## 🧪 Publier sur TestPyPI
	@echo "$(GREEN)🧪 Publication sur TestPyPI...$(NC)"
	$(POETRY) config repositories.testpypi https://test.pypi.org/legacy/
	$(POETRY) publish -r testpypi
	@echo "$(GREEN)✅ Publié sur TestPyPI!$(NC)"

publish: build ## 🚀 Publier sur PyPI
	@echo "$(YELLOW)⚠️  Publication sur PyPI (production)!$(NC)"
	@read -p "Êtes-vous sûr? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(POETRY) publish; \
		echo "$(GREEN)✅ Publié sur PyPI!$(NC)"; \
	else \
		echo "$(RED)❌ Publication annulée$(NC)"; \
	fi

# ═══════════════════════════════════════════════════════════════════════════════
# 🐳 DOCKER
# ═══════════════════════════════════════════════════════════════════════════════

docker-build: ## 🐳 Construire l'image Docker
	@echo "$(GREEN)🐳 Construction de l'image Docker...$(NC)"
	docker build -t $(PACKAGE_NAME):latest .
	docker build -t $(PACKAGE_NAME):$(shell $(POETRY) version -s) .
	@echo "$(GREEN)✅ Image Docker construite!$(NC)"

docker-run: ## 🏃 Lancer le conteneur Docker
	@echo "$(GREEN)🏃 Lancement du conteneur Docker...$(NC)"
	docker run -it --rm -p 8000:8000 -p 8501:8501 $(PACKAGE_NAME):latest
	@echo "$(GREEN)✅ Conteneur arrêté!$(NC)"

docker-push: docker-build ## ☁️ Pousser l'image Docker vers le registry
	@echo "$(GREEN)☁️ Envoi vers le registry Docker...$(NC)"
	docker tag $(PACKAGE_NAME):latest $(DOCKER_REGISTRY)/$(PACKAGE_NAME):latest
	docker tag $(PACKAGE_NAME):latest $(DOCKER_REGISTRY)/$(PACKAGE_NAME):$(shell $(POETRY) version -s)
	docker push $(DOCKER_REGISTRY)/$(PACKAGE_NAME):latest
	docker push $(DOCKER_REGISTRY)/$(PACKAGE_NAME):$(shell $(POETRY) version -s)
	@echo "$(GREEN)✅ Images poussées!$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

docs: ## 📚 Générer la documentation
	@echo "$(GREEN)📚 Génération de la documentation...$(NC)"
	$(POETRY) run mkdocs build
	@echo "$(GREEN)✅ Documentation générée dans site/$(NC)"

docs-serve: ## 🌐 Servir la documentation en local
	@echo "$(GREEN)🌐 Serveur de documentation: http://localhost:8000$(NC)"
	$(POETRY) run mkdocs serve

docs-deploy: ## 🚀 Déployer la documentation
	@echo "$(GREEN)🚀 Déploiement de la documentation...$(NC)"
	$(POETRY) run mkdocs gh-deploy
	@echo "$(GREEN)✅ Documentation déployée!$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# 🗄️ BASE DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

db-init: ## 🗄️ Initialiser la base de données
	@echo "$(GREEN)🗄️ Initialisation de la base de données...$(NC)"
	$(POETRY) run autoerp init
	@echo "$(GREEN)✅ Base de données initialisée!$(NC)"

db-migrate: ## 🔄 Appliquer les migrations
	@echo "$(GREEN)🔄 Application des migrations...$(NC)"
	$(POETRY) run autoerp migrate
	@echo "$(GREEN)✅ Migrations appliquées!$(NC)"

db-seed: ## 🌱 Peupler la base avec des données de test
	@echo "$(GREEN)🌱 Peuplement de la base de données...$(NC)"
	$(POETRY) run autoerp seed
	@echo "$(GREEN)✅ Données de test ajoutées!$(NC)"

db-reset: ## ⚠️ Réinitialiser complètement la base de données
	@echo "$(YELLOW)⚠️  Réinitialisation de la base de données!$(NC)"
	@read -p "Êtes-vous sûr? Toutes les données seront perdues! [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(POETRY) run autoerp init --force; \
		$(POETRY) run autoerp seed --reset; \
		echo "$(GREEN)✅ Base réinitialisée!$(NC)"; \
	else \
		echo "$(RED)❌ Réinitialisation annulée$(NC)"; \
	fi

# ═══════════════════════════════════════════════════════════════════════════════
# 🏃 DÉVELOPPEMENT
# ═══════════════════════════════════════════════════════════════════════════════

dev: ## 🚀 Lancer l'environnement de développement complet
	@echo "$(GREEN)🚀 Démarrage de l'environnement de développement...$(NC)"
	@echo "$(YELLOW)📡 API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)🎨 UI: http://localhost:8501$(NC)"
	@echo "$(YELLOW)📖 Docs: http://localhost:8000/docs$(NC)"
	$(POETRY) run autoerp runserver --reload &
	sleep 3
	$(POETRY) run streamlit run $(SRC_DIR)/ui.py --server.port 8501 &
	@echo "$(GREEN)✅ Environnement démarré! Ctrl+C pour arrêter$(NC)"
	@trap 'kill %1 %2 2>/dev/null || true' EXIT; wait

server: ## 🖥️ Lancer seulement le serveur API
	@echo "$(GREEN)🖥️ Démarrage du serveur API...$(NC)"
	$(POETRY) run autoerp runserver --reload

ui: ## 🎨 Lancer seulement l'interface utilisateur
	@echo "$(GREEN)🎨 Démarrage de l'interface utilisateur...$(NC)"
	$(POETRY) run streamlit run $(SRC_DIR)/ui.py

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

check: lint test ## 🔍 Vérification complète (lint + tests)
	@echo "$(GREEN)✅ Vérification complète terminée!$(NC)"

check-all: format lint security test coverage ## 🎯 Vérification exhaustive
	@echo "$(GREEN)✅ Vérification exhaustive terminée!$(NC)"

setup: install db-init ## 🎯 Configuration initiale complète du projet
	@echo "$(GREEN)🎯 Configuration initiale terminée!$(NC)"
	@echo "$(YELLOW)💡 Prochaines étapes:$(NC)"
	@echo "  - make dev    # Lancer l'environnement de développement"
	@echo "  - make test   # Lancer les tests"
	@echo "  - make docs   # Générer la documentation"

requirements: ## 📋 Exporter les requirements.txt
	@echo "$(GREEN)📋 Export des requirements...$(NC)"
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --with dev --without-hashes
	@echo "$(GREEN)✅ Requirements exportés!$(NC)"

version: ## 📋 Afficher la version actuelle
	@echo "$(GREEN)📋 Version actuelle: $(shell $(POETRY) version -s)$(NC)"

bump-patch: ## 🔢 Incrémenter la version patch (0.1.0 -> 0.1.1)
	$(POETRY) version patch
	@echo "$(GREEN)✅ Version patch incrémentée: $(shell $(POETRY) version -s)$(NC)"

bump-minor: ## 🔢 Incrémenter la version minor (0.1.0 -> 0.2.0)
	$(POETRY) version minor
	@echo "$(GREEN)✅ Version minor incrémentée: $(shell $(POETRY) version -s)$(NC)"

bump-major: ## 🔢 Incrémenter la version major (0.1.0 -> 1.0.0)
	$(POETRY) version major
	@echo "$(GREEN)✅ Version major incrémentée: $(shell $(POETRY) version -s)$(NC)"

.PHONY: help install install-prod update test test-fast test-unit test-integration coverage format lint lint-fix security pre-commit build clean publish-test publish docker-build docker-run docker-push docs docs-serve docs-deploy db-init db-migrate db-seed db-reset dev server ui check check-all setup requirements version bump-patch bump-minor bump-major