# Projet Deep Learning Médical Multimodal — Pipeline DataOps & Fusion

Ce projet implémente un pipeline **DataOps automatisé, testé, versionné et reproductible** associant de l'imagerie thoracique (radiographies pulmonaires) et des données cliniques tabulaires hospitalières (EHR / analyses de laboratoire) pour la détection de pathologies pulmonaires (14 classes).

---

## 1. Arborescence du Projet

```text
projet/
├── configs/
│   └── params.yaml              # Configuration centralisée (externalisation seuils, chemins, pathologies)
├── data/                        # Données suivies par DVC (ignorées par Git)
│   ├── raw/                     # Données brutes téléchargées (Kaggle, PhysioNet)
│   ├── staging/                 # Images nettoyées, réencodées et dédupliquées
│   ├── interim/                 # Données de transformation intermédiaire
│   └── processed/               # Datasets prêts pour l'entraînement
├── flows/
│   └── pipeline.py              # Orchestration Prefect multi-sources
├── reports/
│   ├── data_card.md             # Fiche descriptive des données (Data Card)
│   ├── RAPPORT_TACHE_3.md       # Rapport académique complet Tâche 3 (Parties A & B)
│   ├── quality_report.html      # Rapport de qualité HTML automatique (Bonus)
│   ├── drift_report.json        # Rapport d'analyse de dérive des données (Bonus)
│   └── *.json                   # Audits d'ingestion horodatés par source
├── sql/
│   └── schema.sql               # Schéma relationnel PostgreSQL (10 tables)
├── src/
│   ├── alignment/               # Jointure X-Ray ↔ EHR (align.py)
│   ├── fusion/                  # Modules de fusion PyTorch (modules.py, losses.py, train.py)
│   ├── ingestion/               # Téléchargement robuste et reprise sur coupure (collect.py)
│   ├── preprocessing/           # Nettoyage, réencodage, assignation des splits (transform.py)
│   ├── validation/              # Schémas de données Pandera (validate.py)
│   ├── anonymize.py             # Pseudonymisation HMAC-SHA256 & sanitisation d'image
│   ├── config.py                # Chargeur de configuration unifié (YAML + .env)
│   ├── db.py                    # Chargement transactionnel PostgreSQL & post-checks
│   ├── drift.py                 # Détection de dérive statistique (KS / PSI / TVD)
│   ├── generate_html_report.py  # Génération du rapport qualité HTML
│   ├── mimic_demo.py            # Préparation MIMIC-IV Demo
│   ├── pneumonia.py             # Préparation Kaggle Pneumonia
│   ├── scheduler.py             # Planification de la collecte périodique (cron / daemon)
│   └── storage.py               # Stockage PostgreSQL BYTEA et lecture PIL
├── tests/
│   ├── conftest.py              # Fixtures pytest (données synthétiques)
│   ├── test_data_quality.py     # Tests de qualité (schéma, fuites, intégrité)
│   ├── test_fusion.py           # Tests unitaires de la fusion PyTorch
│   ├── test_pipeline.py         # Tests des étapes de transformation
│   └── test_schema.py           # Validation stricte Pandera
├── ui/
│   ├── __init__.py
│   └── app.py                   # Serveur FastAPI du DataOps Control Center
├── .github/
│   └── workflows/
│       └── ci.yml               # Intégration continue GitHub Actions
├── .dvc/                        # Configuration et métadonnées DVC
├── app.py                       # Point d'entrée pour lancer l'interface web (http://localhost:8000)
├── docker-compose.yml           # Service conteneurisé PostgreSQL 16
├── dvc.yaml                     # Déclaration du DAG de pipeline DVC
├── dvc.lock                     # Fichier d'état figé du pipeline DVC
├── Makefile                     # Commandes unifiées (pipeline, test, ui, lint, docker-up)
├── pytest.ini                   # Configuration pytest
├── requirements.txt             # Dépendances Python
├── run.py                       # CLI unifié d'ingestion
└── run_tests.py                 # Runner de test unifié (100% sans dépendance externe)
```

---

## 2. Installation & Démarrage Rapide

### Prérequis
- Python 3.12
- Docker Desktop (recommandé pour PostgreSQL conteneurisé)
- Git & DVC

### Installation
```bash
# 1. Cloner le dépôt
git clone <url_du_depot>
cd "Projet deep learning medimodal"

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Sur Windows :
.venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
copy .env.example .env
# Générer une clé secrète de pseudonymisation :
python -c "import secrets; print(secrets.token_hex(32))"
# Renseigner PSEUDO_SECRET dans .env
```

---

## 3. Comment un Tiers Récupère les Données (DVC)

Les données lourdes (radiographies et données brutes) sont versionnées via **DVC (Data Version Control)** et dissociées de Git :

```bash
# Récupérer les données depuis le stockage distant DVC :
dvc pull

# Vérifier l'état des données et des étapes :
dvc status
```

---

## 4. Exécution du Pipeline en une Seule Commande

Conformément aux exigences de la Tâche 3, une commande unique permet de reconstruire l'ensemble des données préparées :

```bash
# Via DVC (avec détection automatique du cache) :
dvc repro

# Ou via le Makefile :
make pipeline

# Visualiser le graphe des étapes (DAG) :
dvc dag
```

---

## 5. Interface Utilisateur DataOps Control Center

Une interface web interactive (FastAPI + Tailwind) est incluse pour monitorer et piloter le pipeline en temps réel :

```bash
# Lancer l'interface web :
python app.py
# ou
make ui
```
Puis ouvrez votre navigateur sur : **`http://localhost:8000`**

### Fonctionnalités de l'interface :
- **Pilotage 1-Clic :** Déclenchement de chaque source (`NIH`, `Pneumonia`, `MIMIC`, `Tout le pipeline`, `Post-checks`).
- **Console d'exécution en direct :** Streaming des logs d'ingestion en temps réel.
- **Métriques dynamiques :** Mise à jour automatique des volumes, taux de rejet et doublons à chaque exécution.
- **Détection de dérive (Data Drift) :** Suivi du PSI et de la dérive statistique entre splits.
- **Checklist Interactive :** Suivi point par point de toutes les exigences du sujet avec statut en temps réel.
- **Simulateur de Fusion Multimodale :** Test interactif des dimensions tensorielles et de l'attention croisée.

---

## 6. Tests Automatisés & Qualité des Données

```bash
# Exécuter l'ensemble de la suite de tests qualité :
python run_tests.py
# ou via pytest :
pytest tests/ -v
```

Les tests vérifient :
1. La conformité des schémas Pandera.
2. La détection et neutralisation des âges aberrants (> 110 ans).
3. **L'absence totale de fuite patient** entre train / validation / test (`Zero Data Leakage`).
4. La déduplication stricte des images par hash SHA-256 des pixels.
5. L'intégrité référentielle en base de données PostgreSQL.

---

## 7. Architecture de Fusion Multimodale (Partie B)

- **Approche retenue :** Fusion intermédiaire par **attention croisée bidirectionnelle avec unité à porte adaptative (GMU)**.
- **Encodeur Image :** ResNet-50 pré-entraîné extrayant 49 tokens spatiaux $7 \times 7$ projetés à $d=256$.
- **Encodeur EHR :** Transformer encodeur 2 couches pour les données tabulaires et biologiques ($d=256$).
- **Gestion des modalités manquantes :** Jeton absent appris (`learned absent token`) et `modality dropout` à l'entraînement.
- **Rapport complet :** Voir [`reports/RAPPORT_TACHE_3.md`](reports/RAPPORT_TACHE_3.md).
