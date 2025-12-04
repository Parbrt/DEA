import pandas as pd
import numpy as np
import gc
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Initialisation de la console Rich
console = Console()


def print_step(title):
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def print_shape_table(title, df_train, df_test=None):
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="dim")
    table.add_column("Lignes", justify="right")
    table.add_column("Colonnes", justify="right")

    table.add_row("Train", str(df_train.shape[0]), str(df_train.shape[1]))
    if df_test is not None:
        table.add_row("Test", str(df_test.shape[0]), str(df_test.shape[1]))

    console.print(table)


# ---------------------------------------------------------
# 1. Fonctions de Feature Engineering & Nettoyage
# ---------------------------------------------------------

def application_train_test(path_to_data='../dataset/raw/'):
    print_step("1. Chargement & Nettoyage : Application Train/Test")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
    ) as progress:
        task1 = progress.add_task(description="Lecture des fichiers CSV...", total=None)

        df_train = pd.read_csv(path_to_data + 'application_train.csv')
        df_test = pd.read_csv(path_to_data + 'application_test.csv')

        progress.update(task1, completed=True)

    console.print(f"[green]✔ Données chargées avec succès.[/green]")

    # Fusion temporaire
    df = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

    # --- A. Nettoyage des Anomalies ---
    console.print("[yellow]⚠ Traitement de l'anomalie DAYS_EMPLOYED (365243 -> NaN)...[/yellow]")
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    # --- B. Feature Engineering ---
    console.print("[blue]Feature Engineering : Création des ratios financiers...[/blue]")
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # --- C. Synthèse EXT_SOURCE ---
    console.print("[blue]Feature Engineering : Synthèse des scores externes...[/blue]")
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MULT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # --- D. Encodage ---
    console.print("[blue]Encodage des variables catégorielles...[/blue]")
    le = LabelEncoder()
    le_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1

    df = pd.get_dummies(df)

    console.print(f"[green]✔ Traitement initial terminé. Colonnes actuelles : {df.shape[1]}[/green]")
    return df


# ---------------------------------------------------------
# 2. Fonctions d'Agrégation (Tables liées)
# ---------------------------------------------------------

def bureau_and_balance(df, path_to_data='../dataset/raw/'):
    print_step("2. Fusion : Données Bureau (Historique externe)")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
    ) as progress:
        progress.add_task(description="Chargement et agrégation de bureau.csv...", total=None)
        bureau = pd.read_csv(path_to_data + 'bureau.csv')

        # Agrégations
        bureau_agg_ops = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'AMT_ANNUITY': ['max', 'mean']
        }

        bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_agg_ops)
        bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        bureau_agg['BUREAU_COUNT'] = bureau.groupby('SK_ID_CURR').size()

        # Fusion
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

        del bureau, bureau_agg
        gc.collect()

    console.print(f"[green]✔ Données Bureau fusionnées. Colonnes : {df.shape[1]}[/green]")
    return df


def previous_applications(df, path_to_data='../dataset/raw/'):
    print_step("3. Fusion : Previous Applications (Historique interne)")

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
    ) as progress:
        progress.add_task(description="Chargement et agrégation de previous_application.csv...", total=None)
        prev = pd.read_csv(path_to_data + 'previous_application.csv')

        agg_ops = {
            'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
            'DAYS_DECISION': ['min', 'max', 'mean']
        }

        # Filtre sur colonnes existantes uniquement
        existing_cols = [k for k in agg_ops.keys() if k in prev.columns]
        prev_agg = prev.groupby('SK_ID_CURR').agg({k: agg_ops[k] for k in existing_cols})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size()

        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')

        del prev, prev_agg
        gc.collect()

    console.print(f"[green]✔ Données Previous App fusionnées. Colonnes : {df.shape[1]}[/green]")
    return df


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------

def main():
    console.print("[bold red]Démarrage du Pipeline de Préparation de Données[/bold red]", justify="center")

    # 1. Base
    df = application_train_test()

    # 2. Enrichissement
    df = bureau_and_balance(df)
    df = previous_applications(df)

    # 3. Séparation et Imputation
    print_step("4. Gestion des manquants & Séparation Train/Test")

    console.print("[dim]Séparation des jeux de données...[/dim]")
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()

    target = train_df['TARGET']
    train_df = train_df.drop(columns=['TARGET'])
    test_df = test_df.drop(columns=['TARGET'])

    console.print("[bold yellow]Calcul de l'imputation (Médiane) sur le TRAIN...[/bold yellow]")

    imputer = SimpleImputer(strategy='median')

    # Barre de progression pour l'imputation
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn()
    ) as progress:
        task_fit = progress.add_task("[cyan]Fit Imputer...", total=1)
        imputer.fit(train_df)
        progress.update(task_fit, advance=1)

        task_trans = progress.add_task("[cyan]Transform Train & Test...", total=2)
        train_processed = pd.DataFrame(imputer.transform(train_df), columns=train_df.columns)
        progress.update(task_trans, advance=1)
        test_processed = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)
        progress.update(task_trans, advance=1)

    # Réintégration
    train_processed['TARGET'] = target.values

    # 4. Résumé Final
    print_step("5. Résultat Final")
    print_shape_table("Dimensions Finales des Datasets", train_processed, test_processed)

    console.print("[bold green]Sauvegarde en format Parquet...[/bold green]")
    train_processed.to_parquet('../dataset/processed/train_final.parquet')
    test_processed.to_parquet('../dataset/processed/test_final.parquet')

    console.print(":rocket: [bold white on green] Pipeline terminé avec succès ! [/bold white on green]")


if __name__ == "__main__":
    main()