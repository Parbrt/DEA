import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from rich.console import Console

# Si LightGBM est installé, on l'importe, sinon on passe
try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

console = Console()


def load_data(path):
    console.print(f"[dim]Chargement : {path}...[/dim]")
    df = pd.read_parquet(path)

    # --- NETTOYAGE DES NOMS DE COLONNES POUR LIGHTGBM ---
    # On remplace tout ce qui n'est pas (Lettre, Chiffre ou _) par ""
    new_cols = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]

    # Variante plus robuste avec Regex (re) :
    # df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # On applique les nouveaux noms
    df.columns = new_cols

    console.print(
        "[dim]Noms de colonnes nettoyés pour compatibilité JSON/LightGBM.[/dim]"
    )
    return df


def main():
    # --- 1. Configuration MLflow ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = "file://" + os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit_Scoring_Optimization")

    # --- 2. Préparation des Données ---
    df = load_data(
        os.path.join(project_root, "dataset", "processed", "train_final.parquet")
    )

    # Pour l'optimisation, on prend un échantillon si le dataset est trop gros (>100k lignes)
    # Cela permet d'aller plus vite pour tester le code. En prod, commentez la ligne sample.
    # df_sample = df.sample(n=20000, random_state=42)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    console.print(f"[blue]Dataset réduit pour optimisation : {X.shape}[/blue]")

    # --- 3. Définition des Modèles et des Grilles ---
    # On définit ici ce qu'on veut tester
    models_config = {
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {
                "n_estimators": [50, 100, 200, 300, 400, 500],
                "max_depth": [5, 10, 20],
                "min_samples_split": [2, 10, 20],
            },
        }
    }

    if HAS_LGBM:
        models_config["LightGBM"] = {
            "model": lgb.LGBMClassifier(
                class_weight="balanced", random_state=42, verbose=-1
            ),
            "params": {
                "n_estimators": [100, 300, 500, 800, 1000, 2000],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 70],
                "max_depth": [-1, 10, 20],
                "class_weight": ["balanced"],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            },
        }

    # --- 4. Boucle d'Optimisation ---
    console.rule("[bold magenta]Démarrage de l'Optimisation[/bold magenta]")

    # Cross-Validation Stratifiée (Obligatoire selon le sujet)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for model_name, config in models_config.items():
        with mlflow.start_run(run_name=f"Optim_{model_name}"):
            console.print(f"[yellow]Test du modèle : {model_name}[/yellow]")

            # Recherche Aléatoire (Plus rapide que GridSearchCV et souvent aussi efficace)
            # n_iter=5 signifie qu'on teste 5 combinaisons au hasard par modèle
            search = RandomizedSearchCV(
                estimator=config["model"],
                param_distributions=config["params"],
                n_iter=30,  # Augmenter à 20+ pour une vraie recherche
                scoring="roc_auc",
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=42,
            )

            # Entraînement
            search.fit(X, y)

            # Log des meilleurs résultats dans MLflow
            best_score = search.best_score_
            best_params = search.best_params_

            console.print(f"  [green]✔ Meilleur AUC trouvé : {best_score:.4f}[/green]")
            console.print(f"  [dim]Params : {best_params}[/dim]")

            # On loggue manuellement les meilleurs params pour les retrouver facilement
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_auc", best_score)
            mlflow.sklearn.log_model(search.best_estimator_, "best_model")

            # Tag pour dire que c'est un run d'optimisation
            mlflow.set_tag("type", "optimization")

    console.rule("[bold green]Fin de l'optimisation[/bold green]")
    console.print("Lancez [bold]mlflow ui[/bold] pour comparer les modèles.")


if __name__ == "__main__":
    main()
