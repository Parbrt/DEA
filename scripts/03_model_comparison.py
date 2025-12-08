import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import re
import time
import warnings

# Algorithmes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

# Outils d'évaluation et de pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, confusion_matrix

# Affichage
from rich.console import Console
from rich.table import Table

# Gestion des imports optionnels (LightGBM / XGBoost)
try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Suppression des warnings inutiles pour l'affichage
warnings.filterwarnings("ignore")

console = Console()

# ==============================================================================
# 1. CONFIGURATION DU COÛT MÉTIER
# ==============================================================================
COST_FN = 10  # Coût d'un défaut non détecté (Grave)
COST_FP = 1  # Coût d'un client refusé à tort (Moins grave)


def custom_business_cost(y_true, y_pred):
    """
    Calcule le coût métier.
    Attention : Scikit-learn attend un score "plus c'est haut, mieux c'est".
    Comme ici c'est un coût (plus c'est bas, mieux c'est), on devra inverser le signe
    dans make_scorer ou traiter cela comme un score négatif.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * COST_FN) + (fp * COST_FP)


# On transforme cette fonction en métrique utilisable par cross_validate
# greater_is_better=False signifie que le meilleur modèle est celui qui a le score le plus bas
business_scorer = make_scorer(custom_business_cost, greater_is_better=False)

# ==============================================================================
# 2. FONCTIONS UTILITAIRES
# ==============================================================================


def load_and_clean_data(path):
    console.print(f"[dim]Chargement des données depuis {path}...[/dim]")
    df = pd.read_parquet(path)
    # Nettoyage regex pour compatibilité LightGBM/XGBoost
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    return df


def main():
    # --- A. Setup ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = "file://" + os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit_Scoring_Model_Comparison_Extended")

    train_path = os.path.join(
        project_root, "dataset", "processed", "train_final.parquet"
    )
    df = load_and_clean_data(train_path)

    # Pour le dev, on peut sampler, mais pour le résultat final, essayez de prendre tout ou 50%
    # df = df.sample(n=50000, random_state=42)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # Calcul du ratio de déséquilibre pour XGBoost (scale_pos_weight)
    # Ratio = (Nombre de 0) / (Nombre de 1)
    scale_pos_weight_value = (len(y) - y.sum()) / y.sum()

    console.rule(
        "[bold magenta]Démarrage du Comparatif Étendu (Cross Validation)[/bold magenta]"
    )
    console.print(f"Dataset taille : {X.shape}")
    console.print(f"Taux de défaut : {y.mean():.2%}")
    console.print(f"Coût Métier    : FN={COST_FN} | FP={COST_FP}")

    # --- B. Définition des Modèles ---
    models_to_test = []

    # 1. Logistic Regression (Baseline Linéaire)
    models_to_test.append(
        (
            "Logistic_Regression",
            make_pipeline(
                StandardScaler(),  # Indispensable pour LogReg
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="lbfgs"
                ),
            ),
        )
    )

    # 2. Random Forest (Baseline Arbres)
    models_to_test.append(
        (
            "Random_Forest",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ),
        )
    )

    # 3. XGBoost (Boosting Rapide)
    if HAS_XGB:
        models_to_test.append(
            (
                "XGBoost",
                xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight_value,  # Gestion du déséquilibre
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    # 4. LightGBM (Boosting Microsoft)
    if HAS_LGBM:
        models_to_test.append(
            (
                "LightGBM",
                lgb.LGBMClassifier(
                    class_weight="balanced",
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    n_jobs=-1,
                    random_state=42,
                    verbose=-1,
                ),
            )
        )

    # 5. MLP (Réseau de Neurones Simple)
    # Attention : MLP n'a pas de paramètre "class_weight". Il risque d'être moins bon sur le Recall.
    models_to_test.append(
        (
            "MLP_Neural_Net",
            make_pipeline(
                StandardScaler(),  # CRUCIAL pour les réseaux de neurones
                MLPClassifier(
                    hidden_layer_sizes=(100, 50),  # 2 couches cachées
                    max_iter=500,
                    activation="relu",
                    solver="adam",
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        )
    )

    # --- C. Configuration Cross-Validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Dictionnaire des métriques
    scoring = {
        "AUC": "roc_auc",
        "F1": "f1",
        "Recall": "recall",  # Rappel sur la classe positive (Défaut)
        "Precision": "precision",
        "Business_Cost": business_scorer,  # Notre métrique perso
    }

    # --- D. Boucle d'Exécution ---
    final_table = Table(title="Résultats Comparatifs (5-Fold CV)")
    final_table.add_column("Modèle", style="cyan")
    final_table.add_column("AUC", style="magenta")
    final_table.add_column("Recall", style="green")
    final_table.add_column("Coût Métier (Moy)", style="red")
    final_table.add_column("Temps (s)", style="dim")

    for name, model in models_to_test:
        console.print(f"\n[bold yellow]>>> Test du modèle : {name}[/bold yellow]")

        with mlflow.start_run(run_name=f"Compare_{name}"):
            start_time = time.time()
            mlflow.set_tag("model_family", name)

            # Exécution de la Cross-Validation
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            elapsed_time = time.time() - start_time

            # Moyennes
            mean_auc = cv_results["test_AUC"].mean()
            mean_recall = cv_results["test_Recall"].mean()
            mean_f1 = cv_results["test_F1"].mean()

            # Le score "Business_Cost" sera négatif à cause de greater_is_better=False.
            # On prend la valeur absolue pour l'affichage.
            mean_cost = abs(cv_results["test_Business_Cost"].mean())

            # Logging MLflow
            mlflow.log_metric("cv_mean_auc", mean_auc)
            mlflow.log_metric("cv_mean_recall", mean_recall)
            mlflow.log_metric("cv_mean_f1", mean_f1)
            mlflow.log_metric("cv_mean_business_cost", mean_cost)
            mlflow.log_metric("training_time", elapsed_time)

            if "Pipeline" in str(type(model)):
                mlflow.log_param("pipeline_steps", [s[0] for s in model.steps])

            # Affichage
            console.print(f"   AUC: {mean_auc:.4f} | Recall: {mean_recall:.4f}")
            console.print(
                f"   Coût Métier Moyen: [bold red]{mean_cost:,.0f} €[/bold red]"
            )

            final_table.add_row(
                name,
                f"{mean_auc:.4f}",
                f"{mean_recall:.4f}",
                f"{mean_cost:,.0f}",
                f"{elapsed_time:.1f}",
            )

    console.rule("[bold green]Compétition Terminée[/bold green]")
    console.print(final_table)
    console.print(
        "[dim]Note : Le Coût Métier est calculé avec un seuil par défaut de 0.5. L'optimisation du seuil réduira encore ce coût.[/dim]"
    )


if __name__ == "__main__":
    main()
