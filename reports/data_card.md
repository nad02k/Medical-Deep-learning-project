# Data Card : Medical Multimodal Chest X-Ray & EHR Dataset

## 1. Vue d'ensemble du jeu de données

Ce jeu de données multimodal associe des données d'imagerie thoracique (radiographies pulmonaires) et des données cliniques tabulaires issues de dossiers médicaux électroniques (EHR) pour la détection assistée par ordinateur de pathologies pulmonaires (14 classes CheXpert / NIH).

| Attribut | Description |
| :--- | :--- |
| **Domaine** | Imagerie médicale thoracique & Biologie clinique hospitalière |
| **Modalités** | Image (radiographie 2D en niveaux de gris) + Clinique tabulaire (âge, sexe, position d'incidence, constantes biologiques) |
| **Cibles de prédiction** | 14 pathologies pulmonaires (Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax) |
| **Pipeline DataOps** | Orchestration DVC / Prefect, validation Pandera, stockage PostgreSQL (`image_blobs` & tables relationnelles) |

---

## 2. Sources de données & Licences

| Source | Type de données | Licence | Accès & Réf. |
| :--- | :--- | :--- | :--- |
| **NIH ChestX-ray14** | Radiographies thoraciques (1024x1024) + métadonnées patients (âge, sexe, incidence) | **Domaine Public / NIH Open Access** | Wang et al., CVPR 2017 (`nih-chest-xrays/sample`) |
| **Kaggle Chest X-Ray Pneumonia** | Radiographies pédiatriques antéro-postérieures (JPEG) | **CC BY 4.0** | Kermany et al., Cell 2018 (`paultimothymooney/chest-xray-pneumonia`) |
| **PhysioNet MIMIC-IV Demo (v2.2)** | Dossiers médicaux électroniques (démographie, admissions, diagnostics CIM, biologie) | **Open Database License (ODbL v1.0)** | Johnson et al., PhysioNet (`https://physionet.org/files/mimic-iv-demo/2.2/`) |

---

## 3. Volumes & Caractéristiques Réelles

D'après les audits récents générés automatiquement dans `reports/` :
- **NIH Sample** :
  - Images traitées : 5 606 radiographies
  - Patients distincts : 4 230 patients
  - Taux de valeurs manquantes (âge) : 0.02%
  - Âges aberrants détectés (> 110 ans) : 1 patient (ex. 414 ans dans NIH) neutralisé à `NULL`.
- **Kaggle Pneumonia** :
  - Images brutes : 5 856
  - Images nettoyées : 5 824 (32 doublons de pixels stricts éliminés par déduplication SHA-256)
  - Patients identifiés : 2 789 patients
  - Fuite de données corrigée : 264 patients se trouvaient dans plusieurs sous-dossiers (train/val/test d'origine Kaggle) ; le pipeline recalcule un partitionnement patient strict.
  - Équilibre de classe : 4 273 Pneumonie vs 1 583 Normal.
- **MIMIC-IV Demo** :
  - 100 patients hospitalisés
  - 275 admissions
  - 4 506 codes diagnostics (ICD-9 / ICD-10)
  - 107 727 événements de laboratoire (biochimie, hématologie, gazométrie)

---

## 4. Prétraitements & Pseudonymisation (RGPD / HIPAA)

1. **Pseudonymisation irréversible** :
   - Tous les identifiants bruts (patients, admissions, images) subissent un hachage HMAC-SHA256 avec clé secrète externe (`PSEUDO_SECRET`), préfixés (`P_`, `I_`, `H_`, `S_`).
   - Refus strict d'ingestion si `PSEUDO_SECRET` est absent.
2. **Nettoyage et sanitisation des images** :
   - Conversion systématique en niveaux de gris 8-bit (`L`).
   - Suppression complète de toutes les métadonnées (EXIF, marqueurs DICOM, chunks PNG textuels).
   - Déduplication au bit près via hash SHA-256 des octets décompressés.
3. **Minimisation des données (Data Minimization)** :
   - Conformément aux principes de confidentialité, les attributs sensibles non nécessaires au diagnostic (ethnie/race, religion, statut matrimonial, assurance, texte libre des comptes-rendus) sont écartés dès l'ingestion.
4. **Partitionnement sans fuite (Data Leakage Free)** :
   - Découpage strict au niveau patient : `SHA-256(pseudo_id) % 100` attribuant 70% train, 15% val, 15% test.
   - Aucun patient ne peut avoir d'images ou d'enregistrements répartis sur plusieurs folds.

---

## 5. Biais connus & Limites

1. **Biais démographique et géographique** :
   - Les données NIH proviennent d'un centre hospitalier tertiaire spécialisé (NIH Clinical Center), avec une prévalence accrue de pathologies rares ou avancées.
   - Les radiographies Pneumonia proviennent exclusivement de cohortes pédiatriques (1 à 5 ans, Women and Children’s Medical Center, Guangzhou).
2. **Hétérogénéité des capteurs** :
   - Variations importantes de contraste, de tension (kVp) et de dimensions d'origine entre les sources.
3. **Complétude asymétrique (Modalités manquantes)** :
   - Les images NIH possèdent des données cliniques minimales (âge, sexe, incidence).
   - Les images Pneumonia n'ont aucun historique clinique dans les métadonnées (`age_years = NULL`, `sex = NULL`).
   - Les données MIMIC-IV contiennent une richesse biologique élevée mais sans les radiographies associées dans la version démo.
   - **Conséquence directe pour le modèle de fusion** : nécessité absolue de gérer les modalités manquantes par masquage (`modality dropout`) et jeton d'absence appris (`absent token`).
