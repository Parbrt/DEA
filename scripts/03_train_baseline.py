import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from rich.console import Console
from rich.table import Table

# Initialisation de Rich
console = Console()


def load_data(path):
    """Charge les données Parquet nettoyées."""
    console.print(f"[dim]Chargement des données depuis {path}...[/dim]")
    df = pd.read_parquet(path)
    return df


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = "file://" + os.path.join(project_root, "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    console.print(f"[yellow]MLflow Tracking URI : {tracking_uri}[/yellow]")

    experiment_name = "Credit_Scoring_Projet"
    console.rule("[bold magenta]Démarrage de l'Entraînement (Baseline)[/bold magenta]")

    # 1. Chargement
    # Adaptez le chemin si nécessaire
    train_path = "../dataset/processed/train_final.parquet"
    df = load_data(train_path)

    # 2. Préparation (X et y)
    console.print("[blue]Séparation des features et de la target...[/blue]")
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Séparation Train/Validation
    # Stratify est crucial ici à cause du déséquilibre des classes (0 vs 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    console.print(f"[green]✔ Données prêtes.[/green] Train: {X_train.shape}, Val: {X_val.shape}")

    # ==========================================================================
    # CONFIGURATION MLFLOW
    # ==========================================================================
    experiment_name = "Credit_Scoring_Projet"
    mlflow.set_experiment(experiment_name)
    console.print(f"[yellow]MLflow Experiment : {experiment_name}[/yellow]")

    # Activer l'autologging (capture params, metrics, et model.pkl automatiquement)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        console.print("[bold cyan]Run MLflow démarré... Entraînement en cours...[/bold cyan]")

        # Tags pour retrouver l'expérience plus tard
        mlflow.set_tag("developer", "VotreNom")
        mlflow.set_tag("model", "RandomForest")
        mlflow.set_tag("phase", "Baseline")

        # 3. Modélisation (Random Forest simple pour commencer)
        # class_weight='balanced' est important pour le déséquilibre
        clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Utilise tous les coeurs du CPU
        )

        clf.fit(X_train, y_train)
        console.print("[green]✔ Modèle entraîné.[/green]")

        # 4. Évaluation
        # Prédiction des probabilités (important pour l'AUC et le seuil métier)
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred_bin = clf.predict(X_val)

        # Calcul des métriques
        auc = roc_auc_score(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred_bin)

        # 5. Affichage des résultats
        results_table = Table(title="Résultats Validation")
        results_table.add_column("Métrique", style="cyan")
        results_table.add_column("Valeur", style="magenta")
        results_table.add_row("AUC", f"{auc:.4f}")
        results_table.add_row("Accuracy", f"{acc:.4f}")
        console.print(results_table)

        # 6. Logging manuel supplémentaire (si besoin)
        # Autolog le fait déjà, mais c'est bien d'assurer les métriques clés
        mlflow.log_metric("custom_auc", auc)

        # 7. Sauvegarde d'un graphique (Matrice de confusion)
        # C'est important pour voir les Faux Négatifs (le risque métier)
        cm = confusion_matrix(y_val, y_pred_bin)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Réel')
        plt.xlabel('Prédit')

        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)

        # Envoi du fichier image dans MLflow
        mlflow.log_artifact(plot_path)
        console.print(f"[dim]Graphique sauvegardé dans MLflow : {plot_path}[/dim]")

        # Nettoyage fichier local
        os.remove(plot_path)

    console.rule("[bold green]Fin du script avec succès[/bold green]")
    console.print("Pour voir les résultats, lancez dans le terminal : [bold]mlflow ui[/bold]")


if __name__ == "__main__":
    main()