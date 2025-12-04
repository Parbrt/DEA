import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Configuration pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------
# 1. Fonctions de Feature Engineering & Nettoyage
# ---------------------------------------------------------

def application_train_test(path_to_data='./dataset/raw/'):
    """
    Charge et nettoie les données principales (train + test).
    Gère les anomalies et crée les 'Super Features'.
    """
    print("Chargement des tables application_train et application_test...")
    # Chargement
    df_train = pd.read_csv(path_to_data + 'application_train.csv')
    df_test = pd.read_csv(path_to_data + 'application_test.csv')

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Fusion temporaire pour le traitement commun
    df = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

    # --- A. Nettoyage des Anomalies ---
    print("Traitement de l'anomalie DAYS_EMPLOYED...")
    # 365243 correspond à une valeur infinie/erreur dans ce dataset
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # --- B. Feature Engineering (Domain Knowledge) ---
    print("Création des variables métier (Ratios)...")

    # 1. Pourcentage Crédit / Revenu
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # 2. Ratio Annuité / Revenu (Taux d'endettement)
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # 3. Ratio Crédit / Revenu
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # 4. Impact de la famille sur le revenu
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # --- C. Synthèse des EXT_SOURCE (Très corrélées) ---
    print("Création des synthèses EXT_SOURCE...")
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_MULT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # --- D. Encodage des Catégories ---
    print("Encodage des variables catégorielles...")
    # Label Encoding pour les variables binaires (2 choix) et le sexe
    le = LabelEncoder()
    le_count = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            # Si on a 2 valeurs ou moins, on utilise Label Encoder
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1

    # One-Hot Encoding pour les autres variables catégorielles
    df = pd.get_dummies(df)

    print(f"Fin du traitement Application. Shape actuelle: {df.shape}")
    return df

# ---------------------------------------------------------
# 2. Fonctions d'Agrégation (Tables liées)
# ---------------------------------------------------------

def bureau_and_balance(df, path_to_data='./dataset/raw/'):
    """
    Charge et agrège les données du bureau de crédit.
    """
    print("\nChargement et agrégation de la table BUREAU...")
    bureau = pd.read_csv(path_to_data + 'bureau.csv')

    # --- Feature Engineering sur Bureau ---
    # Par exemple : Crédit actif ou fermé ?

    # Agrégations simples
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

    # GroupBy SK_ID_CURR
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(bureau_agg_ops)
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Compte le nombre de crédits précédents
    bureau_agg['BUREAU_COUNT'] = bureau.groupby('SK_ID_CURR').size()

    # Fusion
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

    del bureau, bureau_agg
    gc.collect()
    return df

def previous_applications(df, path_to_data='./dataset/raw/'):
    """
    Charge et agrège les demandes précédentes chez Home Credit.
    """
    print("\nChargement et agrégation de PREVIOUS_APPLICATION...")
    prev = pd.read_csv(path_to_data + 'previous_application.csv')

    # Agrégations numériques
    agg_ops = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'Rate_interest_primary': ['mean'],
        'Rate_interest_privileged': ['mean']
    }

    # Note: On ne prend que les colonnes numériques pour l'agrégation simple ici
    # Pour faire simple, on filtre d'abord
    prev_agg = prev.groupby('SK_ID_CURR').agg({k: v for k, v in agg_ops.items() if k in prev.columns})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Compte
    prev_agg['PREV_APP_COUNT'] = prev.groupby('SK_ID_CURR').size()

    # Fusion
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')

    del prev, prev_agg
    gc.collect()
    return df

# ---------------------------------------------------------
# 3. Exécution du Pipeline
# ---------------------------------------------------------

def main():
    # 1. Chargement & Base
    df = application_train_test()

    # 2. Ajout des tables externes (Vous pouvez commenter si trop lourd pour la RAM)
    df = bureau_and_balance(df)
    df = previous_applications(df)

    # 3. Gestion des valeurs manquantes (Imputation simple)
    # Les modèles comme LightGBM gèrent les NaN, mais pour un dataset "propre" générique :
    print("\nImputation des valeurs manquantes (Médiane)...")

    # On sépare la Target avant d'imputer
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()

    # On ne touche pas à la target
    target = train_df['TARGET']
    train_df = train_df.drop(columns=['TARGET'])
    test_df = test_df.drop(columns=['TARGET'])

    # Imputation (uniquement sur les colonnes numériques créées)
    imputer = SimpleImputer(strategy='median')

    # On fit sur le train et transform sur train et test pour éviter le data leakage
    # Attention: cela peut prendre de la mémoire. Si erreur mémoire, faites-le par chunk ou utilisez un modèle qui gère les NaN natifs.
    imputer.fit(train_df)

    train_processed = pd.DataFrame(imputer.transform(train_df), columns=train_df.columns)
    test_processed = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    # Réintégration de la target et des IDs
    train_processed['TARGET'] = target.values

    # 4. Sauvegarde
    print("\nSauvegarde des fichiers processed...")
    # Format Parquet est beaucoup plus rapide et léger que CSV pour ces gros volumes
    train_processed.to_parquet('./dataset/processed/train_final.parquet')
    test_processed.to_parquet('./dataset/processed/test_final.parquet')

    print(f"Terminé ! \nDataset Train final : {train_processed.shape}")
    print(f"Dataset Test final : {test_processed.shape}")

if __name__ == "__main__":
    main()