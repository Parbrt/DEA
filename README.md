# Projet de Credit Scoring & MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=flat&logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat&logo=docker) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat&logo=mlflow) ![LightGBM](https://img.shields.io/badge/LightGBM-Model-green?style=flat)

## Résumé du Projet et Objectif Métier

Ce projet vise à développer un outil d'aide à la décision pour une société financière ("Home Credit"), permettant d'évaluer automatiquement la solvabilité de clients ayant peu d'historique de crédit.

L'enjeu principal dépasse la simple classification technique : il s'agit d'optimiser la rentabilité en minimisant le risque financier. En effet, accorder un crédit à un client qui ne rembourse pas (Faux Négatif) coûte beaucoup plus cher que de refuser un bon client (Faux Positif).

### Stratégie de "Coût Métier"

Pour répondre à cette exigence, le modèle a été optimisé selon une fonction de coût personnalisée :

* **Coût d'un Faux Négatif (FN) : 10** (Défaut de paiement non détecté = Perte du capital).
* **Coût d'un Faux Positif (FP) : 1** (Client refusé à tort = Manque à gagner).
* **Objectif :** Minimiser le score $Total Cost = 10 \times FN + 1 \times FP$.

---

## Installation et Lancement

```

### 1. Construction de l'image Docker

Le projet est encapsulé dans un conteneur Docker pour assurer la reproductibilité et servir le modèle via une API.
Depuis la racine du projet, lancez la commande suivante :

```bash
docker build -t credit-scoring-app .
```

### 2. Lancement du serveur de Modèe (Serving)

Une fois l'image construite, démarrez le conteneur en exposant le port 1234:

```bash
docker run -p 1234:1234 credit-scoring-app
```

L'API est maintenant accessible localment sur `http://127.0.0.1:1234`

### 3. Test de l'API

```bash
python3 scripts/05_test_real_api.py
```

Ce programme utilise une valeur aléatoire du dataset et l'envoie à l'API et elle renvoie la probabilité d'échec du remboursement ainsi que le seuil et si on accepte ou pas le prêt.

### 4. Mise en place de python et ces dépendances (facultatif)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Décision et Seuil Métier

Le modèle final est un **LightGBM** optimisé. Grâce à l'utilisation de la pondération des classes (`class_weight='balanced'`), le modèle a intégré la pénalité des Faux Négatifs directement lors de l'apprendtissage.

* Seuil Métier Choisi : 0.48
* Règle de Décision:
  * Si `probabilité de Défaut > 0.48`: Crédit refusé (Risque trop élevé)
  * Si `probabilité de Défaut <= 0.48`: Crédit accordé

Ce seuil à été déterminé en simulant les coûts financiers sur l'ensemble du jeu de données validé, garantissant le meilleur compromis économique pour la banque.

## Structure du Dépot

```
credit-scoring/
├── dataset/
│   ├── raw/                 # Données brutes (Kaggle)
│   └── processed/           # Données nettoyées et fusionnées (Parquet)
├── mlruns/                  # Tracking des expérimentations MLflow
├── model/                   # Artefacts du modèle final
│   └── model.pkl            # Modèle sérialisé pour Docker
├── reports/                 # Rapports et figures générées
│   └── figures/             # Courbe coût vs seuil, Feature Importance
├── scripts/                 # Pipeline Data Science complet
│   ├── 02_processing_dataset.py     # Nettoyage et Feature Engineering
│   ├── 05_model_comparison.py       # Comparaison des algorithmes (LGBM, RF, LogReg)
│   ├── 06_threshold_optimization.py # Optimisation Hyperparamètres & Seuil Métier
│   └── 07_test_real_api.py          # Script de test automatisé de l'API
├── app.py                   # Code source de l'API FastAPI
├── Dockerfile               # Configuration de l'image Docker
├── requirements.txt         # Liste des dépendances Python
└── README.md
```
