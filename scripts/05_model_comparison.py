import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import re
import time
import lightgbm as lgb

# Algo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


# Outils d'évaluation et de pipline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, f1_score, recall_score, roc_auc_score

# Affichage
from rich.console import Console
from rich.table import Table


console = Console()


def load_and_clean_data(path):
    console.print(f"[dim]Chargement des données depuis {path}...[/dim]")
    df = pd.read_parquet(path)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    return df


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = "file://" + os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit_Scoring_Optimization")

    train_path = os.path.join(
        project_root, "dataset", "processed", "train_final.parquet"
    )
    df = load_and_clean_data(train_path)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    console.rule(
        "[bold magenta] Démarrage du Comparatif (Cross Validation)[/bold magenta]"
    )
    console.print(f"Dataset taille : {X.shape}")
    console.print(f"Ration Classe 1 (Défaut) : ", {y.mean()})
    models_to_test = []

    models_to_test.append(
        (
            "Logistic_Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, C=0.1, solver="lbfgs"
                ),
            ),
        )
    )

    models_to_test.append(
        (
            "Random_Forest",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            ),
        )
    )

    models_to_test.append(
        (
            "LightGBM",
            lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            ),
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "AUC": "roc_auc",
        "F1": "f1",
        "Recall": "recall",
        "Precision": "precision",
    }

    final_table = Table(title="Résultats Moyens (Cross-Validation 5 Folds)")
    final_table.add_column("Modèle", style="cyan")
    final_table.add_column("AUC Moyen", style="magenta")
    final_table.add_column("Recall Moyen", style="green")
    final_table.add_column("Temps (s)", style="dim")

    for name, model in models_to_test:
        console.print(f"\n[bold yellow]>>> Test du modèle : {name}[/bold yellow]")

        with mlflow.start_run(run_name=f"Compare_{name}"):
            start_time = time.time()

            # 1. Log des paramètres (automatique ou manuel selon le modèle)
            mlflow.set_tag("model_family", name)

            # 2. Validation Croisée
            # cross_validate fait tout le travail : split, train, predict, score
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
            )

            elapsed_time = time.time() - start_time

            # 3. Calcul des moyennes
            mean_auc = cv_results["test_AUC"].mean()
            mean_recall = cv_results["test_Recall"].mean()
            mean_f1 = cv_results["test_F1"].mean()
            mean_precision = cv_results["test_Precision"].mean()

            # 4. Logging MLflow
            mlflow.log_metric("cv_mean_auc", mean_auc)
            mlflow.log_metric("cv_mean_recall", mean_recall)
            mlflow.log_metric("cv_mean_f1", mean_f1)
            mlflow.log_metric("training_time", elapsed_time)

            # Si c'est un Pipeline (LogReg), on veut logger les étapes
            if "Pipeline" in str(type(model)):
                mlflow.log_param("steps", [s[0] for s in model.steps])

            # 5. Affichage Console
            console.print(
                f"   AUC: {mean_auc:.4f} | Recall: {mean_recall:.4f} | F1: {mean_f1:.4f}"
            )

            final_table.add_row(
                name, f"{mean_auc:.4f}", f"{mean_recall:.4f}", f"{elapsed_time:.1f}"
            )

    console.rule("[bold green]Compétition Terminée[/bold green]")
    console.print(final_table)
    console.print("Lancez [bold]mlflow ui[/bold] pour voir les courbes détaillées.")


if __name__ == "__main__":
    main()
