import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('../dataset/raw/application_train.csv')
# Configuration esthétique
sns.set_style('whitegrid')
plt.figure(figsize=(12, 10))

# 1. Calcul des corrélations (uniquement sur les colonnes numériques)
# On ne garde que les nombres pour la corrélation de Pearson
correlations = df.select_dtypes(include=[np.number]).corr()

# 2. Focus sur la corrélation avec la TARGET
target_corr = correlations['TARGET'].sort_values()

# Affichage des corrélations les plus fortes (Positives et Négatives)
print("--- Top 10 Corrélations Négatives (Les plus inversement liées au défaut) ---")
print(target_corr.head(10))
print("\n--- Top 10 Corrélations Positives (Les plus liées au défaut) ---")
print(target_corr.tail(10))

# 3. Graphique : Heatmap des variables les plus corrélées
# On sélectionne les 15 variables les plus corrélées (en valeur absolue)
# pour éviter une heatmap illisible de 100x100
top_corr_features = target_corr.abs().sort_values(ascending=False).head(15).index
top_corr_matrix = df[top_corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap des 15 variables les plus corrélées avec TARGET')
plt.show()

# 4. Analyse visuelle détaillée des meilleures variables (KDE Plot)
# Les variables 'EXT_SOURCE' sont souvent les plus importantes dans ce dataset.
# Vérifions leur distribution selon la classe (0 = Remboursé, 1 = Défaut).

features_to_plot = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']

plt.figure(figsize=(12, 10))
for i, feature in enumerate(features_to_plot):
    if feature in df.columns:
        plt.subplot(2, 2, i + 1)
        # Plot des clients en règle (Target = 0)
        sns.kdeplot(df.loc[df['TARGET'] == 0, feature],
                    label='Target 0 (Remboursé)', fill=True, color='blue')
        # Plot des clients en défaut (Target = 1)
        sns.kdeplot(df.loc[df['TARGET'] == 1, feature],
                    label='Target 1 (Défaut)', fill=True, color='red')

        plt.title(f'Distribution de {feature} selon la Target')
        plt.legend()
plt.tight_layout()
plt.show()

print("Préparation des données pour l'analyse d'importance...")

# 1. Préparation temporaire (pour que l'algo puisse tourner)
# On travaille sur une copie pour ne pas abîmer votre dataframe original
temp_df = df.copy()

# Encodage rapide des variables textuelles (Label Encoding)
le = LabelEncoder()
for col in temp_df.columns:
    if temp_df[col].dtype == 'object':
        # On remplit les NaN par une catégorie 'Missing' avant d'encoder
        temp_df[col] = temp_df[col].astype(str)
        temp_df[col] = le.fit_transform(temp_df[col])

# Remplissage basique des NaN (nécessaire pour sklearn)
temp_df = temp_df.fillna(temp_df.median())

# -------------------------------------------------------
# Méthode A : Importance via Corrélation (Linéaire)
# -------------------------------------------------------
print("Calcul des corrélations...")
correlations = temp_df.corr()['TARGET'].abs().sort_values(ascending=False)
# On retire la target elle-même
correlations = correlations.drop('TARGET')
top_corr = correlations.head(15)

# -------------------------------------------------------
# Méthode B : Importance via Random Forest (Non-linéaire)
# -------------------------------------------------------
print("Entraînement rapide d'un Random Forest...")
# On prend un échantillon de 20 000 lignes pour que ça aille vite
sample_df = temp_df.sample(n=20000, random_state=42)
X = sample_df.drop(columns=['TARGET'])
y = sample_df['TARGET']

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Extraction des importances
feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

# -------------------------------------------------------
# Affichage des Graphiques
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Graphique 1 : Corrélation
sns.barplot(x=top_corr.values, y=top_corr.index, ax=axes[0], palette="Blues_r")
axes[0].set_title("Top 15 Variables les plus Corrélées (Absolue) avec TARGET")
axes[0].set_xlabel("Coefficient de Corrélation (Valeur Absolue)")

# Graphique 2 : Feature Importance (Random Forest)
sns.barplot(x=feature_imp.values, y=feature_imp.index, ax=axes[1], palette="Greens_r")
axes[1].set_title("Top 15 Variables Importantes (Random Forest)")
axes[1].set_xlabel("Importance (Gini)")

plt.tight_layout()
plt.show()