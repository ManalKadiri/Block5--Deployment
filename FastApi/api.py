import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn

app = FastAPI()

class PredictionFeatures(BaseModel):
    model_key: str  # Corrigé pour éviter le conflit de nom
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

@app.get("/")
async def index():
    message = "Hello this is my API. Please check out documentation of the API at `/docs`"
    return message

@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    try:
        # Créer un DataFrame à partir des caractéristiques reçues
        data = pd.DataFrame([{
            "model_key": predictionFeatures.model_key,
            "mileage": predictionFeatures.mileage,
            "engine_power": predictionFeatures.engine_power,
            "fuel": predictionFeatures.fuel,
            "paint_color": predictionFeatures.paint_color,
            "car_type": predictionFeatures.car_type,
            "private_parking_available": predictionFeatures.private_parking_available,
            "has_gps": predictionFeatures.has_gps,
            "has_air_conditioning": predictionFeatures.has_air_conditioning,
            "automatic_car": predictionFeatures.automatic_car,
            "has_getaround_connect": predictionFeatures.has_getaround_connect,
            "has_speed_regulator": predictionFeatures.has_speed_regulator,
            "winter_tires": predictionFeatures.winter_tires
        }])

        # Conversion des colonnes en float64 pour correspondre au schéma du modèle MLflow
        data["mileage"] = data["mileage"].astype(float)
        data["engine_power"] = data["engine_power"].astype(float)

        # Obtenez l'ID de run depuis l'environnement ou utilisez une valeur par défaut
        run_id = '3fab0ec510764ad49b185a25c4eae7d0'
        print(f"Run ID utilisé : {run_id}")

        logged_model = f'runs:/{run_id}/model'
        print(f"Chemin du modèle : {logged_model}")

        # Charger le modèle depuis MLflow
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            print("Modèle chargé avec succès.")
            # Effectuer la prédiction
            prediction = loaded_model.predict(data)

            # Retourner la prédiction sous forme de réponse JSON
            response = {"prediction": prediction.tolist()[0]}
        except mlflow.exceptions.MlflowException as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {e}")

        return response

    except Exception as e:
        print(f"Erreur générale: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
