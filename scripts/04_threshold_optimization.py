import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pickle
import shutil

# Scikit-learn & LightGBM
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import lightgbm as lgb

console = Console()

# ==============================================================================
# CONFIGURATION MÉTIER
# ==============================================================================
COST_FN = 10  # Coût d'un défaut non détecté (Très grave)
COST_FP = 1  # Coût d'un client refusé à tort (Peu grave)


def load_and_clean_data(path):
    console.print(f"[dim]Chargement des données depuis {path}...[/dim]")
    df = pd.read_parquet(path)
    # Nettoyage des noms de colonnes (LightGBM n'aime pas les espaces/virgules)
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "_", x))
    return df


def business_cost_score(y_true, y_pred):
    """Calcule le coût total selon la matrice de confusion."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * COST_FN) + (fp * COST_FP)


def main():
    # --- 1. Initialisation ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "model")
    reports_dir = os.path.join(project_root, "reports", "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    tracking_uri = "file://" + os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit_Scoring_Optimization_Finale")

    train_path = os.path.join(
        project_root, "dataset", "processed", "train_final.parquet"
    )
    df = load_and_clean_data(train_path)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    console.print(
        Panel.fit(
            f"[bold white]Optimisation Finale du Modèle LightGBM[/bold white]\n"
            f"Dataset: {X.shape[0]} lignes | Features: {X.shape[1]}\n"
            f"Ratio Déséquilibre: {y.mean():.2%}\n"
            f"Coût Métier: FN={COST_FN} | FP={COST_FP}",
            title="Configuration",
            border_style="blue",
        )
    )

    # --- 2. Grille d'Hyperparamètres (Étoffée) ---
    #
    lgbm = lgb.LGBMClassifier(
        class_weight="balanced",  # Indispensable pour le déséquilibre
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    param_dist = {
        # On autorise un apprentissage plus rapide (0.1)
        "learning_rate": [0.05],
        # On garde beaucoup d'arbres
        "n_estimators": [200],
        # On augmente la capacité du modèle (plus de feuilles = plus complexe)
        "num_leaves": [31],
        # On réduit la régularisation (pour le laisser apprendre plus de détails)
        "reg_alpha": [0, 0.1, 1],  # On autorise 0 (pas de frein)
        "reg_lambda": [0, 0.1, 1],
        "colsample_bytree": [0.7, 0.9],
        "subsample": [0.7, 0.9],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    console.rule(
        "[bold magenta]1. Recherche des Meilleurs Hyperparamètres (RandomizedSearch)[/bold magenta]"
    )

    search = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=20,  # On teste 30 combinaisons pour être sûr de trouver le top
        scoring=make_scorer(roc_auc_score),
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    with mlflow.start_run(run_name="Final_Model_Optimization") as run:
        console.print(
            "[yellow]Exploration de l'espace des paramètres... (Cela peut prendre quelques minutes)[/yellow]"
        )
        search.fit(X, y)

        best_model = search.best_estimator_

        # Affichage des meilleurs paramètres
        console.print(
            f"\n[bold green]✔ Meilleur AUC trouvé : {search.best_score_:.4f}[/bold green]"
        )

        param_table = Table(title="Meilleurs Hyperparamètres")
        param_table.add_column("Paramètre", style="cyan")
        param_table.add_column("Valeur", style="magenta")
        for k, v in search.best_params_.items():
            param_table.add_row(k, str(v))
        console.print(param_table)

        # Logging MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("auc_optim", search.best_score_)

        # --- 3. Optimisation du Seuil Métier ---
        console.rule(
            "[bold magenta]2. Optimisation du Seuil de Décision (Business Cost)[/bold magenta]"
        )

        # On utilise cross_val_predict pour avoir des probabilités propres sur tout le dataset
        console.print("[dim]Calcul des probabilités par validation croisée...[/dim]")
        y_probas = cross_val_predict(
            best_model, X, y, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        thresholds = np.arange(0.01, 1.00, 0.01)
        costs = []

        # Boucle de calcul du coût pour chaque seuil
        for t in thresholds:
            y_pred_t = (y_probas > t).astype(int)
            cost = business_cost_score(y, y_pred_t)
            costs.append(cost)

        # Sélection du meilleur seuil
        min_cost_index = np.argmin(costs)
        best_threshold = thresholds[min_cost_index]
        min_cost = costs[min_cost_index]

        # Comparaison avec le seuil par défaut (0.5)
        default_cost = business_cost_score(y, (y_probas > 0.5).astype(int))
        gain = default_cost - min_cost

        # Tableau Final
        results_table = Table(title="Rapport Financier Final", border_style="green")
        results_table.add_column("Métrique", style="white")
        results_table.add_column("Valeur", style="bold yellow", justify="right")

        results_table.add_row("Seuil Optimal", f"{best_threshold:.2f}")
        results_table.add_row("Coût Minimum (Optimisé)", f"{min_cost:,.0f} €")
        results_table.add_row("Coût Standard (Seuil 0.5)", f"{default_cost:,.0f} €")

        if gain > 0:
            gain_str = f"[bold green]- {gain:,.0f} €[/bold green]"
        else:
            gain_str = "[dim]0 € (Modèle déjà équilibré)[/dim]"

        results_table.add_row("Économie Réalisée", gain_str)

        console.print(results_table)

        # Logging MLflow
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("min_business_cost", min_cost)

        # --- 4. Graphique Coût vs Seuil ---
        plt.figure(figsize=(12, 6))
        plt.plot(
            thresholds, costs, label="Coût Métier Total", color="darkblue", linewidth=2
        )
        plt.axvline(
            best_threshold,
            color="red",
            linestyle="--",
            label=f"Optimal: {best_threshold:.2f}",
        )
        plt.axvline(0.5, color="gray", linestyle=":", label="Standard: 0.50")

        plt.xlabel("Seuil de Probabilité (0 = Laxiste, 1 = Sévère)")
        plt.ylabel("Coût Total (Unités monétaires)")
        plt.title(f"Minimisation du Coût Métier (FN={COST_FN}, FP={COST_FP})")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = "courbe_cout_vs_seuil.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # Copie pour le rapport
        shutil.copy(plot_path, os.path.join(reports_dir, "courbe_cout_vs_seuil.png"))
        os.remove(plot_path)

        # --- 5. Sauvegarde Finale ---
        console.rule("[bold magenta]3. Sauvegarde du Modèle Final[/bold magenta]")

        # Réentraînement final sur tout le dataset
        best_model.fit(X, y)

        # MLflow Registry
        mlflow.sklearn.log_model(best_model, "model")

        # Fichier Local pour Docker
        local_model_path = os.path.join(model_dir, "model.pkl")
        with open(local_model_path, "wb") as f:
            pickle.dump(best_model, f)

        console.print(f"[green]✔ Modèle enregistré : {local_model_path}[/green]")
        console.print(
            f"[bold]Notez ce Seuil ({best_threshold:.2f}) pour le fichier app.py ![/bold]"
        )


if __name__ == "__main__":
    main()
