# Rapport Tâche 3 : Automatisation DataOps du Pipeline & Étude de la Fusion de Données Multimodales

**Projet :** Deep Learning Multimodal Médical (Radiographies Thoraciques & Dossiers Électroniques de Santé)  
**Auteurs :** Équipe Medimodal  
**Date :** Novembre 2026  
**Livrables :** Dépôt Git versionné + Pipeline DVC reproductible + Interface Utilisateur DataOps + Rapport complet + Prototype PyTorch

---

## Sommaire

1. [Partie A — Automatisation du Pipeline selon une Approche DataOps](#1-partie-a--automatisation-du-pipeline-selon-une-approche-dataops)
   - [A.1 Respect des 7 Principes DataOps](#a1-respect-des-7-principes-dataops)
   - [A.2 Nouvelle Arborescence et Modularité du Projet](#a2-nouvelle-arborescence-et-modularité-du-projet)
   - [A.3 Externalisation de la Configuration (`configs/params.yaml`)](#a3-externalisation-de-la-configuration-configsparamsyaml)
   - [A.4 Versionnement des Données et Reproductibilité avec DVC](#a4-versionnement-des-données-et-reproductibilité-avec-dvc)
   - [A.5 Orchestration et Graphe Déclaratif des Étapes (`dvc.yaml`)](#a5-orchestration-et-graphe-déclaratif-des-étapes-dvcyaml)
   - [A.6 Contrôles de Qualité des Données et Tests Automatiques](#a6-contrôles-de-qualité-des-données-et-tests-automatiques)
   - [A.7 Intégration Continue (GitHub Actions CI)](#a7-intégration-continue-github-actions-ci)
   - [A.8 Traçabilité, Observabilité et Fiche de Données (Data Card)](#a8-traçabilité-observabilité-et-fiche-de-données-data-card)
   - [A.9 Interface Utilisateur DataOps Control Center](#a9-interface-utilisateur-dataops-control-center)
2. [Partie B — Étude des Approches de Fusion Multimodale](#2-partie-b--étude-des-approches-de-fusion-multimodale)
   - [B.1 Synthèse Bibliographique Exhaustive](#b1-synthèse-bibliographique-exhaustive)
   - [B.2 Tableau Comparatif des Mécanismes de Fusion](#b2-tableau-comparatif-des-mécanismes-de-fusion)
   - [B.3 Justification Argumentée du Choix pour Notre Projet](#b3-justification-argumentée-du-choix-pour-notre-projet)
   - [B.4 Architecture Précise du Modèle de Fusion](#b4-architecture-précise-du-modèle-de-fusion)
   - [B.5 Formulation Mathématique et Pseudo-Algorithme](#b5-formulation-mathématique-et-pseudo-algorithme)
   - [B.6 Fonction de Perte et Robustesse aux Données Manquantes](#b6-fonction-de-perte-et-robustesse-aux-données-manquantes)
   - [B.7 Baselines et Protocole d'Évaluation Expérimental](#b7-baselines-et-protocole-dévaluation-expérimental)
3. [Conclusion & Perspectives](#3-conclusion--perspectives)

---

# 1. Partie A — Automatisation du Pipeline selon une Approche DataOps

## A.1 Respect des 7 Principes DataOps

L'approche DataOps adapte les concepts du DevOps et de l'ingénierie logicielle aux flux de données hétérogènes. Notre pipeline médical garantit l'application stricte des sept principes fondamentaux :

| Principe DataOps | Implémentation concrète dans notre projet |
| :--- | :--- |
| **1. Automatisation** | Reconstitution intégrale des données préparées via une commande unique (`dvc repro` ou `make pipeline`). Aucune intervention manuelle entre le téléchargement brut et les tables relationnelles finales. |
| **2. Modularité** | Découpage en packages distincts : `src/ingestion/` (collecte sécurisée), `src/validation/` (schémas Pandera), `src/preprocessing/` (anonymisation HMAC, normalisation d'images), `src/alignment/` (appariement multi-modal), et `src/fusion/` (modules PyTorch). |
| **3. Reproductibilité** | Graine aléatoire figée (`random_seed: 42`), partitionnement déterministe par hachage cryptographique du pseudonyme patient (`SHA-256(pseudo_id) % 100`), dépendances figées dans `requirements.txt`. |
| **4. Versionnement** | Git assure la traçabilité intégrale du code, des schémas DDL et des fichiers de configuration. DVC versionne les gros volumes d'images et données tabulaires (`data/raw`, `data/staging`, `data/processed`). Aucun fichier lourd n'est committé dans Git. |
| **5. Qualité Continue** | Suite de tests automatisés (`tests/test_data_quality.py`, `tests/test_schema.py`) exécutée à chaque exécution du pipeline et sous CI. Arrêt strict si une violation critique survient. |
| **6. Observabilité** | Métriques enregistrées à chaque exécution dans la table PostgreSQL `ingestion_audit` (volumes, doublons écartés, taux d'outliers, durées) et rapports JSON horodatés dans `reports/`. |
| **7. Intégration Continue (CI)** | Workflow GitHub Actions (`.github/workflows/ci.yml`) configuré pour exécuter le linting (`flake8`), les tests unitaires et la validation de schéma sur échantillon à chaque `git push`. |

---

## A.2 Nouvelle Arborescence et Modularité du Projet

Le dépôt a été restructuré selon les standards industriels MLOps :

```text
projet/
├── .github/
│   └── workflows/
│       └── ci.yml               # Workflow d'intégration continue GitHub Actions
├── configs/
│   └── params.yaml              # Fichier central de configuration (externalisation)
├── data/                        # Suivi par DVC, ignoré par Git
│   ├── raw/                     # Données sources brutes téléchargées
│   ├── staging/                 # Images réencodées dédupliquées sans métadonnées
│   └── processed/               # Ensembles prêts pour l'entraînement
├── flows/
│   └── pipeline.py              # Orchestration Prefect multi-tâches
├── reports/
│   ├── data_card.md             # Fiche descriptive de données (Data Card)
│   ├── nih_c713fdff.json        # Rapport statistique NIH
│   ├── pneumonia_5e48fe20.json  # Rapport statistique Pneumonia
│   └── mimic_demo_9388a31d.json # Rapport statistique MIMIC-IV demo
├── sql/
│   └── schema.sql               # Schéma relationnel PostgreSQL (10 tables + index)
├── src/
│   ├── alignment/               # Appariement et jointure des modalités
│   │   ├── __init__.py
│   │   └── align.py             # Jointure X-Ray ↔ EHR & agrégation biologique
│   ├── fusion/                  # Réservé à la Partie B & tâches futures
│   │   ├── __init__.py
│   │   ├── modules.py           # Encodeurs PyTorch, Cross-Attention, Modality Dropout
│   │   └── losses.py            # MultimodalLoss (BCE + aux + InfoNCE)
│   ├── ingestion/               # Téléchargement et reprise sur coupure
│   ├── preprocessing/           # Nettoyage, réencodage, assignation des splits
│   ├── validation/              # Schémas de données Pandera
│   ├── anonymize.py             # Pseudonymisation HMAC-SHA256 & sanitisation d'image
│   ├── collect.py               # Connecteurs KaggleHub & PhysioNet avec Range HTTP
│   ├── config.py                # Chargeur unifié (params.yaml + .env)
│   ├── db.py                    # Chargement transactionnel PostgreSQL & post-checks
│   ├── mimic_demo.py            # Préparation spécifique MIMIC-IV EHR
│   ├── pneumonia.py             # Préparation spécifique Kaggle Pneumonia
│   ├── storage.py               # Stockage PostgreSQL BYTEA et lecture PIL
│   └── transform.py             # Préparation NIH ChestX-ray14
├── tests/
│   ├── conftest.py              # Fixtures pytest (jeux de données synthétiques)
│   ├── test_data_quality.py     # Tests de qualité des données (splits, outliers, leaks)
│   ├── test_fusion.py           # Tests unitaires du module de fusion multimodale
│   ├── test_pipeline.py         # Tests d'intégration des étapes de transformation
│   └── test_schema.py           # Validation stricte des schémas Pandera
├── ui/
│   ├── __init__.py
│   └── app.py                   # Interface Web DataOps Control Center (FastAPI)
├── app.py                       # Point d'entrée pour lancer l'interface web
├── docker-compose.yml           # Service conteneurisé PostgreSQL 16
├── dvc.yaml                     # Déclaration du graphe acyclique direct (DAG) du pipeline
├── Makefile                     # Commandes unifiées (pipeline, test, ui, lint)
├── pytest.ini                   # Configuration pytest (marqueurs unit/integration)
├── requirements.txt             # Dépendances Python figées
├── run.py                       # CLI unifié d'ingestion et de contrôle
└── run_tests.py                 # Runner de test compatible unittest et pytest
```

---

## A.3 Externalisation de la Configuration (`configs/params.yaml`)

Conformément à la règle DataOps « *aucun seuil ni chemin codé en dur* », toutes les constantes métier ont été déportées dans `configs/params.yaml` :
- **Seuils biologiques et physiologiques :** `max_age: 110` (permettant d'écarter automatiquement les anomalies de saisie, ex. âge 414 dans NIH).
- **Partitionnement :** `split_ratios: [70, 85]` définissant 70% pour le train, 15% pour la validation et 15% pour le test.
- **Pathologies :** Liste ordonnée des 14 affections pulmonaires suivies.
- **Sources et téléchargements :** Slugs Kaggle, URL PhysioNet MIMIC-IV, nombre de tentatives et timeouts avec reprise sur coupure (HTTP Range).
- **Hyperparamètres de fusion :** Dimension de projection $d=256$, têtes d'attention $H=8$, taux de dropout $0.1$, poids de perte auxiliaire $\lambda=0.3$.

Le module `src/config.py` intègre un parseur PyYAML robuste chargeant dynamiquement ces paramètres tout en préservant la compatibilité avec les variables d'environnement confidentielles du fichier `.env` (mots de passe PostgreSQL, secret HMAC).

---

## A.4 Versionnement des Données et Reproductibilité avec DVC

Le versionnement des données repose sur **DVC (Data Version Control)** :
1. Les données volumineuses brutes (`data/raw`) et les images réencodées (`data/staging`) sont exclues du suivi Git via `.gitignore`.
2. DVC calcule une empreinte md5 pour chaque répertoire et fichier volumineux et stocke les métadonnées dans des fichiers légers `.dvc`.
3. Le stockage distant (remote) peut être configuré vers un dossier partagé local, un bucket AWS S3, Google Cloud Storage ou un serveur MinIO :
   ```bash
   dvc remote add -d myremote s3://medimodal-data-bucket/dvcstore
   dvc push
   ```
4. **Récupération des données par un tiers :**
   ```bash
   git clone <url_du_depot>
   cd medimodal
   pip install -r requirements.txt
   dvc pull
   ```

---

## A.5 Orchestration et Graphe Déclaratif des Étapes (`dvc.yaml`)

Le pipeline est formalisé dans `dvc.yaml` sous la forme d'un DAG dont chaque nœud spécifie ses dépendances de code, de configuration, et ses sorties :

```yaml
stages:
  ingest-nih:
    cmd: python run.py nih
    deps:
      - src/transform.py
      - configs/params.yaml
    outs:
      - data/staging:
          persist: true

  ingest-pneumonia:
    cmd: python run.py pneumonia
    deps:
      - src/pneumonia.py
      - configs/params.yaml

  ingest-mimic:
    cmd: python run.py mimic
    deps:
      - src/mimic_demo.py
      - configs/params.yaml

  validate-quality:
    cmd: python run_tests.py
    deps:
      - tests/test_data_quality.py
      - src/validate.py
```

Grâce à ce découpage, **seules les étapes affectées par une modification de code ou de paramètre sont réexécutées** lors de l'appel à `dvc repro`.

---

## A.6 Contrôles de Qualité des Données et Tests Automatiques

Quatre niveaux de contrôle de qualité sont mis en place :

1. **Vérification de Schéma Pandera :**
   - Contrôle strict des types de colonnes, des valeurs non nulles obligatoires, et des domaines autorisés (`M`/`F`, `PA`/`AP`, labels `NORMAL`/`PNEUMONIA`).
2. **Détection d'Outliers et Valeurs Aberrantes :**
   - L'âge patient est validé via une expression régulière tolérant les valeurs signées (`-?\d+`) et neutralisé en `NULL` s'il sort de l'intervalle $[0, 110]$.
3. **Absence de Fuite de Données (Patient-Level Split) :**
   - L'attribution train/val/test est calculée via le hash HMAC du patient :
     $$\text{hash} = \text{SHA256}(\text{patient\_pseudo\_id}) \pmod{100}$$
   - Si $\text{hash} < 70 \implies \text{train}$, si $< 85 \implies \text{val}$, sinon $\text{test}$.
   - **Garantie mathématique :** 100% des radiographies et données cliniques d'un même individu sont confinées dans un unique ensemble.
   - Les tests ont permis d'isoler et de corriger la fuite présente dans le dataset Kaggle Pneumonia d'origine, où 264 patients se trouvaient simultanément dans les dossiers `train/` et `test/`.
4. **Déduplication au Bit Près :**
   - Hachage cryptographique SHA-256 appliqué sur les octets bruts des pixels après normalisation en niveaux de gris. Élimination de 32 doublons d'images stricts.

---

## A.7 Intégration Continue (GitHub Actions CI)

Le fichier `.github/workflows/ci.yml` déclenche automatiquement sur toute pull request :
1. L'installation de l'environnement Python 3.12.
2. Le linting du code (`flake8`) avec tolérance contrôlée sur la longueur de ligne.
3. L'exécution de la suite complète de tests de qualité des données et de schéma.
4. L'exécution du pipeline sur un échantillon réduit (smoke-test).

---

## A.8 Traçabilité, Observabilité et Fiche de Données (Data Card)

- **Audit en base de données :** La table `ingestion_audit` enregistre à chaque exécution le `run_id`, l'horodatage UTC, l'étape exacte, et les métriques sous forme JSON (volumes traités, doublons rejetés, taux d'incohérence).
- **Data Card (`reports/data_card.md`) :** Document standardisé récapitulant les caractéristiques des cohortes, les licences (Domaine public NIH, CC BY 4.0, PhysioNet ODbL), la minimisation des données (exclusion du texte libre, race, religion, assurance pour conformité RGPD/HIPAA), et les biais de prévalence documentés.

---

## A.9 Interface Utilisateur DataOps Control Center

Une interface utilisateur dédiée a été développée (`ui/app.py` & `app.py`) basée sur FastAPI et un tableau de bord HTML5/Tailwind réactif :
- **Pilotage interactif :** Déclenchement en un clic des ingestions (`NIH`, `Pneumonia`, `MIMIC-IV`, `All`), des contrôles de cohérence, et des tests unitaires.
- **Console d'exécution en temps réel :** Visualisation continue des logs de traitement et du statut d'exécution.
- **Monitoring visuel :** Indicateurs clés de performance (volumes, patients pseudonymisés, contrôles réussis).
- **Studio de fusion :** Simulateur interactif permettant de faire varier la dimension de projection, le nombre de têtes et d'observer le flux tensoriel.

---

# 2. Partie B — Étude des Approches de Fusion Multimodale

## B.1 Synthèse Bibliographique Exhaustive

La fusion multimodale en apprentissage profond consiste à intégrer des informations hétérogènes (ici radiographies $X \in \mathbb{R}^{H \times W}$ et vecteurs cliniques/biologiques $E \in \mathbb{R}^{D}$) pour maximiser la performance diagnostique. On distingue traditionnellement trois niveaux de fusion :

### 1. Fusion Précoce (Early / Data-level Fusion)
- **Principe :** Combinaison des données brutes ou des descripteurs de bas niveau avant tout apprentissage lourd (ex. concaténation d'un vecteur clinique avec les pixels ou aplatissement).
- **Avantages :** Modèle unique à entraîner, capture immédiate des corrélations de bas niveau.
- **Limites :** Inadaptée à des modalités d'hétérogénéité extrême (matrice d'images $1024 \times 1024$ vs 16 constantes biologiques). Le signal dense de l'image écrase les caractéristiques cliniques discrètes (malédiction de la dimensionnalité).

### 2. Fusion Tardive (Late / Decision-level Fusion)
- **Principe :** Entraînement de deux réseaux indépendants (un CNN pour les radiographies, un MLP ou Gradient Boosting pour les données tabulaires), puis combinaison des scores de probabilité prédits $\hat{y}_{\text{img}}$ et $\hat{y}_{\text{ehr}}$ (moyenne pondérée, vote majoritaire, ou régression logistique méta-apprise / stacking).
- **Avantages :** Robustesse absolue aux modalités manquantes (si l'EHR manque, le modèle image fonctionne de manière autonome) ; flexibilité d'architecture.
- **Limites :** Incapacité totale de modéliser les **interactions croisées complexes** (ex. un infiltrat pulmonaire discret sur l'image qui ne prend une signification pathologique certaine que s'il est corrélé à une hyperleucocytose et un pic de CRP dans l'EHR).

### 3. Fusion Intermédiaire / Hybride (Intermediate / Representation-level Fusion)
- **Principe :** Encodeurs dédiés projetant chaque modalité dans un espace latent commun $\mathbb{R}^d$, suivi d'un module d'interaction non linéaire au sein du réseau, entraîné de bout en bout (*end-to-end*).
- **Mécanismes avancés étudiés :**
  - *Concaténation latente :* Simple mais n'offre pas d'attention fine entre régions et variables.
  - *Gated Multimodal Units (GMU) :* Utilise une porte sigmoïde apprenant dynamiquement quelle modalité privilégier pour chaque patient.
  - *Feature-wise Linear Modulation (FiLM) :* Les paramètres statistiques de l'EHR conditionnent affine-ment les cartes de caractéristiques de l'image ($\gamma \cdot x + \beta$).
  - *Tensor Fusion Network (TFN) & Low-rank Multimodal Fusion (LMF) :* Produit de Kronecker des vecteurs latents modélisant explicitement les interactions d'ordre 1, 2 et 3.
  - *Attention Croisée (Cross-Attention Transformers) :* Les requêtes ($Q$) de l'image interrogent les clés/valeurs ($K, V$) des données cliniques, et réciproquement.
  - *Alignement Contrastif (type CLIP) :* Rapproche dans l'espace euclidien les représentations image et clinique d'un même patient via une perte InfoNCE.

---

## B.2 Tableau Comparatif des Mécanismes de Fusion

| Approche | Principe | Avantages | Limites | Conditions d'usage & Réf. |
| :--- | :--- | :--- | :--- | :--- |
| **Fusion précoce (concaténation bas niveau)** | Concaténation des caractéristiques brutes avant l'encodeur | Très simple, modèle unique | Sensible aux différences d'échelle et de dimension ; écrase les modalités rares | Modalités synchronisées de même nature (ex. RGB + Profondeur) |
| **Fusion tardive (moyenne / stacking)** | Modèles indépendants entraînés séparément ; fusion des prédictions $\hat{y}$ | Robuste aux valeurs manquantes ; modulaire ; interprétable | Ignore les synergies et interactions sémantiques fines | Modalités autonomes où chacune suffit partiellement (Snoek et al.) |
| **Intermédiaire — Concaténation simple** | Concaténation des vecteurs latents : $[h_{\text{img}} ; h_{\text{ehr}}]$ | Simple à implémenter, rétropropagation conjointe | Interaction linéaire pauvre ; sensible au déséquilibre dimensionnel | Baseline multimodale standard |
| **Intermédiaire — GMU (Gated Multimodal Units)** | Mécanisme de porte appris : $z = g \odot h_{\text{img}} + (1-g) \odot h_{\text{ehr}}$ | Filtre dynamiquement la modalité la plus fiable ou bruitée | Pas d'alignement spatial localisé | Données avec bruit asymétrique (Arevalo et al., 2017) |
| **Intermédiaire — FiLM (Modulation conditionnelle)** | Les données EHR prédisent les facteurs de gain et biais $\gamma, \beta$ du CNN | Fort pouvoir expressif, n'altère pas la dimension de l'image | Direction unidirectionnelle (EHR $\to$ Image) | VQA, guidage clinique de segmentation (Perez et al., 2018) |
| **Tensor Fusion (TFN / LMF)** | Produit tensoriel extérieur $[h_{\text{img}}, 1] \otimes [h_{\text{ehr}}, 1]$ | Capture toutes les interactions polynomiales bivariées | Explosion combinatoire $O(d^2)$ (résolu partiellement par LMF) | Émotions, signaux de petite dimension (Zadeh et al., 2017 ; Liu et al., 2018) |
| **Intermédiaire — Attention Croisée (Cross-Attention)** | Requêtes d'une modalité projetées sur les clés/valeurs de l'autre | **Alignement sémantique fin** ; explicabilité directe via les cartes d'attention | Coût quadratique en mémoire ; nécessite un volume suffisant | **Choix optimal pour Imagerie + EHR structuré** (Vaswani et al., Tsai et al. Multimodal Transformer) |
| **Alignement Contrastif (CLIP-like)** | Maximisation de la similarité cosinus des paires appariées via InfoNCE | Espace latent unifié ; apprentissage semi-supervisé puissant | Requiert un volume massif de données et des paires strictement appariées | Pré-entraînement de fondation multimodal (Radford et al., 2021) |

---

## B.3 Justification Argumentée du Choix pour Notre Projet

L'analyse objective des caractéristiques mesurées lors des Tâches 1 et 2 dicte notre choix technique :

1. **Hétérogénéité structurelle extrême :** L'imagerie thoracique est une matrice spatiale continue 2D ($1024 \times 1024$), tandis que l'EHR se compose d'attributs discrets scalaires (âge, sexe, leucocytes, créatinine). Une fusion précoce est inapplicable.
2. **Interactions clinico-radiologiques fines :** Le diagnostic de pneumonie ou d'œdème pulmonaire dépend de la conjonction simultanée d'opacités alvéolaires focales et de marqueurs d'infection systémique (globules blancs élevés, fièvre). Une fusion tardive passerait à côté de cette corrélation croisée.
3. **Problématique des modalités manquantes :** Le dataset Pneumonia ne possède aucune métadonnée clinique, tandis que NIH possède des informations démographiques basiques. Notre architecture doit donc impérativement fonctionner même en cas d'absence totale du vecteur EHR lors de l'inférence.
4. **Capacité d'interprétabilité médicale :** Les cliniciens exigent de savoir quelles régions pulmonaires ont motivé la prédiction et quelles analyses de sang y ont contribué. Les cartes de poids de l'attention croisée fournissent cette explicabilité nativement.

**Conclusion :** Nous retenons la **Fusion Intermédiaire par Attention Croisée Bidirectionnelle avec Gating et Jeton d'Absence Appris** (*Gated Cross-Attention Multimodal Transformer*).

---

## B.4 Architecture Précise du Modèle de Fusion

```
[ Image Radiographie ]                     [ Données Cliniques & Labos ]
    (B, 1, 224, 224)                               (B, num_features)
           │                                                │
   ResNet-50 Backbone                              Embedding Tabulaire +
  (Pretrained, Layer4)                             2x Transformer Encoder
           │                                                │
  Tokens Spatiaux h_A                              Tokens Cliniques h_B
    (B, 49, d=256)                                    (B, M, d=256)
           │                                                │
           ├───────────────────────┬────────────────────────┤
           │                       │                        │
           ▼                       ▼                        ▼
     Cross-Attention         Cross-Attention        Modality Dropout
        (Image → EHR)           (EHR → Image)       (Masque missing data)
           │                       │                        │
           └───────────────┬───────┘                        │
                           │                                │
                           ▼                                ▼
                  Gated Multimodal Unit              Absent Token
                Fusion & Pooling Adaptatif          si EHR absent
                           │                                │
                           └───────────────┬────────────────┘
                                           │
                                           ▼
                                 Vecteur Fused z (B, 256)
                                           │
                                  MLP Classification Head
                                           │
                                           ▼
                               Logits (14 Pathologies)
```

1. **Encodeur Image ($E_A$) :** ResNet-50 pré-entraîné (ImageNet), première couche convolutive adaptée à 1 canal (niveaux de gris). Extraction des cartes de caractéristiques avant *global average pooling* ($\text{stride}=32$, soit une grille $7 \times 7 = 49$ tokens spatiaux de dimension 2048), projetés linéairement à $d=256$.
2. **Encodeur EHR ($E_B$) :** Réseau de projection tabulaire avec normalisation par couche (*LayerNorm*) et encodage positionnel, suivi d'un Transformer encodeur à 2 couches ($d=256$, 4 têtes).
3. **Module de Fusion ($F$) :**
   - Requêtes images $Q_A$ interrogent les clés/valeurs cliniques $K_B, V_B$.
   - Requêtes cliniques $Q_B$ interrogent les clés/valeurs images $K_A, V_A$.
   - Agrégation par unité à porte adaptative (*Gated Unit*).
4. **Tête de Prédiction ($H$) :** Perceptron multicouche ($256 \to 128 \to 14$) avec activations ReLU et Dropout ($p=0.2$).

---

## B.5 Formulation Mathématique et Pseudo-Algorithme

### 1. Équations de la Fusion

Soient les représentations projetées $h_A \in \mathbb{R}^{B \times N \times d}$ (image) et $h_B \in \mathbb{R}^{B \times M \times d}$ (EHR).

L'attention croisée s'exprime formellement par :
$$\text{CA}_{A \to B} = \text{Softmax}\left(\frac{(h_A W_Q^A)(h_B W_K^B)^T}{\sqrt{d_k}}\right) (h_B W_V^B)$$
$$\text{CA}_{B \to A} = \text{Softmax}\left(\frac{(h_B W_Q^B)(h_A W_K^A)^T}{\sqrt{d_k}}\right) (h_A W_V^A)$$

Après pooling moyen spatial sur chaque modalité :
$$\bar{z}_A = \text{MeanPool}(\text{LayerNorm}(h_A + \text{CA}_{A \to B})) \in \mathbb{R}^{B \times d}$$
$$\bar{z}_B = \text{MeanPool}(\text{LayerNorm}(h_B + \text{CA}_{B \to A})) \in \mathbb{R}^{B \times d}$$

Le mécanisme de porte (Gating) apprend la contribution respective :
$$g = \sigma(W_g [\bar{z}_A ; \bar{z}_B] + b_g) \in [0, 1]^d$$
$$z = g \odot \bar{z}_A + (1 - g) \odot \bar{z}_B \in \mathbb{R}^{B \times d}$$

Prédiction multi-label finale pour les 14 pathologies :
$$\hat{y} = \sigma(W_2 \cdot \text{ReLU}(W_1 z + b_1) + b_2)$$

### 2. Pseudo-Algorithme de la Passe Avant

```python
# Entrées : batch {x_img, x_ehr, mask_ehr}, encodeurs E_A, E_B, fusion F, tête H
def forward(x_img, x_ehr, mask_ehr):
    # 1. Encodage et projection dans l'espace commun d
    h_A = Proj_A(E_A(x_img))            # h_A in R^{B x 49 x d}
    
    # 2. Gestion de la modalité clinique manquante
    if mask_ehr == 0:
        h_B = LearnableAbsentToken.expand(B, M, d)
    else:
        h_B = Proj_B(E_B(x_ehr))        # h_B in R^{B x M x d}
    
    # 3. Attention croisée bidirectionnelle
    att_A2B = CrossAttention(Q=h_A, K=h_B, V=h_B)
    att_B2A = CrossAttention(Q=h_B, K=h_A, V=h_A)
    
    # 4. Connexions résiduelles et pooling
    z_A = GlobalPool(LayerNorm(h_A + att_A2B))
    z_B = GlobalPool(LayerNorm(h_B + att_B2A))
    
    # 5. Porte multimodale (gating)
    gate = Sigmoid(LinearGate([z_A, z_B]))
    z = gate * z_A + (1 - gate) * z_B
    
    # 6. Classification multi-label
    y_pred = Head(z)                    # Logits in R^{B x 14}
    
    return y_pred, gate, att_A2B
```

---

## B.6 Fonction de Perte et Robustesse aux Données Manquantes

La fonction de perte totale combine trois objectifs :

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}, y) + \lambda \mathcal{L}_{\text{aux}}(y_{\text{img\_seul}}, y) + \gamma \mathcal{L}_{\text{InfoNCE}}(z_A, z_B)$$

1. **$\mathcal{L}_{\text{BCE}}$ (Perte principale) :** Entropie croisée binaire avec logits pondérée pour compenser le déséquilibre de classe :
   $$\mathcal{L}_{\text{BCE}} = -\frac{1}{C}\sum_{c=1}^{14} \left[ w_c y_c \log \sigma(\hat{y}_c) + (1 - y_c) \log (1 - \sigma(\hat{y}_c)) \right]$$
2. **$\mathcal{L}_{\text{aux}}$ (Perte auxiliaire unimodale, $\lambda = 0.3$) :** Force l'encodeur image à demeurer performant même en l'absence de données cliniques.
3. **$\mathcal{L}_{\text{InfoNCE}}$ (Alignement contrastif, $\gamma = 0.1$) :** Encourage les représentations $z_A$ et $z_B$ d'un même patient à converger dans l'espace partagé.
4. **Modality Dropout (Entraînement robuste) :** Lors de l'entraînement, un dropout de modalité de 20% est appliqué artificiellement sur le vecteur EHR (remplacement par le vecteur absent appris) pour immuniser le réseau contre le manque de biologie à l'inférence.

---

## B.7 Baselines et Protocole d'Évaluation Expérimental

Pour prouver rigoureusement l'apport de notre architecture, nous établissons le protocole comparatif suivant :

### 1. Modèles de Comparaison (Baselines)
- **Baseline 1 (Unimodale Image) :** ResNet-50 standard entraîné uniquement sur les radiographies pulmonaires.
- **Baseline 2 (Unimodale EHR) :** Gradient Boosting (XGBoost) et MLP tabulaire entraînés uniquement sur les variables cliniques et constantes biologiques.
- **Baseline 3 (Fusion Tardive) :** Moyenne pondérée optimale des probabilités issues de la Baseline 1 et de la Baseline 2.
- **Baseline 4 (Fusion Intermédiaire Concaténation) :** Concaténation naïve des représentations latentes $[z_A ; z_B]$.
- **Notre Modèle (Cross-Attention Gated Transformer) :** Architecture complète proposée.

### 2. Métriques d'Évaluation
- **AUROC macro et micro** (Aire sous la courbe ROC sur les 14 pathologies).
- **F1-Score macro** au seuil de Youden optimal.
- **Précision moyenne (Average Precision / mAP)** adaptée au déséquilibre des classes rares.

### 3. Étude d'Ablation Prévue
| Configuration | Image | EHR | Mécanisme | AUROC attendu |
| :--- | :---: | :---: | :--- | :---: |
| Image seule | $\checkmark$ | $\times$ | Unimodal ResNet-50 | ~0.81 |
| EHR seul | $\times$ | $\checkmark$ | MLP Tabulaire | ~0.72 |
| Fusion Tardive | $\checkmark$ | $\checkmark$ | Stacking probabilités | ~0.83 |
| Fusion Intermédiaire | $\checkmark$ | $\checkmark$ | Concaténation latente simple | ~0.84 |
| **Notre modèle complet** | $\checkmark$ | $\checkmark$ | **Cross-Attention Bidirectionnelle + Gate** | **~0.88** |
| Notre modèle (EHR manquant) | $\checkmark$ | Absent | Inférence avec Jeton Absent Appris | ~0.81 |

---

# 3. Conclusion & Perspectives

La **Tâche 3** a permis de hisser le projet à un niveau d'ingénierie logicielle et DataOps conforme aux exigences professionnelles :
- Le pipeline de données est **entièrement automatisé, modulaire et versionné avec DVC**.
- Les contrôles automatiques garantissent l'absence de fuite patient et la stricte conformité réglementaire (pseudonymisation HMAC, RGPD/HIPAA).
- Une **interface utilisateur moderne** permet de piloter, d'auditer et d'expérimenter en continu.
- L'étude de fusion débouche sur une architecture moderne à **attention croisée bidirectionnelle**, prête à être entraînée lors des Tâches 4 et 5 avec suivi des expériences sous MLflow.
