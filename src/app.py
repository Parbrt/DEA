import uvicorn
import pandas as pd
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Configuration ---
SEUIL_OPTIMAL = 0.48

app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction de risque de crédit",
    version="1.0.0",
)

# --- Chargement du model ---
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Modèle chargé avec succès. Seuil de décision : {SEUIL_OPTIMAL}")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")


class ClientData(BaseModel):
    features: dict


# Routes
@app.get("/")
def index():
    return {
        "message": "Credit Scoring API en ligne. Utilisez POST /predict pour un score"
    }


@app.post("/predict")
def predict_credit(data: ClientData):
    try:
        client_df = pd.DataFrame([data.features])
        proba_faillite = model.predict_proba(client_df)[:, 1][0]
        decision_int = int(proba_faillite > SEUIL_OPTIMAL)

        return {
            "probability": round(float(proba_faillite), 4),
            "threshold": SEUIL_OPTIMAL,
            "decision": decision_int,  # 0 = Accépté, 1 = Refusé
            "decision_label": "Refusé" if decision_int == 1 else "Accordé",
        }

    except Exception as p:
        raise HTTPException(status_code=400, detail=str(p))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)
