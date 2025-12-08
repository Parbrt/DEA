import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Chargement du dataset
df = pd.read_csv("../dataset/raw/application_train.csv")

sns.set_style("whitegrid")
plt.figure(figsize=(12, 10))

# Calcul des corrélations (on utilise seulement les valeurs numériques)
correlations = df.select_dtypes(include=[np.number]).corr()

target_corr = correlations["TARGET"].sort_values()

# Affichage des corrélations les plus fortes (Positives et Négatives)
print("--- Top 10 Corrélations Négatives (Les plus inversement liées au défaut) ---")
print(target_corr.head(10))
print("\n--- Top 10 Corrélations Positives (Les plus liées au défaut) ---")
print(target_corr.tail(10))

# On trace une heatmap et on sélectionne les 15 variables les plus corrélées
top_corr_features = target_corr.abs().sort_values(ascending=False).head(15).index
top_corr_matrix = df[top_corr_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Heatmap des 15 variables les plus corrélées avec TARGET")
plt.show()

# Analyse visuelle des meilleures variables
features_to_plot = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]

plt.figure(figsize=(12, 10))
for i, feature in enumerate(features_to_plot):
    if feature in df.columns:
        plt.subplot(2, 2, i + 1)
        sns.kdeplot(
            df.loc[df["TARGET"] == 0, feature],
            label="Target 0 (Remboursé)",
            fill=True,
            color="blue",
        )
        sns.kdeplot(
            df.loc[df["TARGET"] == 1, feature],
            label="Target 1 (Défaut)",
            fill=True,
            color="red",
        )

        plt.title(f"Distribution de {feature} selon la Target")
        plt.legend()
plt.tight_layout()
plt.show()

print("Préparation des données pour l'analyse d'importance...")

temp_df = df.copy()

le = LabelEncoder()
for col in temp_df.columns:
    if temp_df[col].dtype == "object":
        temp_df[col] = temp_df[col].astype(str)
        temp_df[col] = le.fit_transform(temp_df[col])

temp_df = temp_df.fillna(temp_df.median())

print("Calcul des corrélations...")
correlations = temp_df.corr()["TARGET"].abs().sort_values(ascending=False)
# On retire la target elle-même
correlations = correlations.drop("TARGET")
top_corr = correlations.head(15)

print("Entraînement rapide d'un Random Forest...")
sample_df = temp_df.sample(n=20000, random_state=42)
X = sample_df.drop(columns=["TARGET"])
y = sample_df["TARGET"]

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_imp = (
    pd.Series(rf.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
    .head(15)
)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(x=top_corr.values, y=top_corr.index, ax=axes[0], palette="Blues_r")
axes[0].set_title("Top 15 Variables les plus Corrélées (Absolue) avec TARGET")
axes[0].set_xlabel("Coefficient de Corrélation (Valeur Absolue)")

sns.barplot(x=feature_imp.values, y=feature_imp.index, ax=axes[1], palette="Greens_r")
axes[1].set_title("Top 15 Variables Importantes (Random Forest)")
axes[1].set_xlabel("Importance (Gini)")

plt.tight_layout()
plt.show()

