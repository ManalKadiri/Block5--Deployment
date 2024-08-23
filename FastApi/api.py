import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn
from typing import Literal, List, Union



description = """
Welcome to my rental price predictor API !\n
Submit the characteristics of your car and a Machine Learning model, trained on GetAround data, will recommend you a price per day for your rental. 

**Use the endpoint `/predict` to estimate the daily rental price of your car !**
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Use this endpoint for getting predictions"
    }
]



app = FastAPI(
    title="💸 Car Rental Price Predictor",
    description=description,
    version="0.1",
    openapi_tags=tags_metadata
)

class PredictionFeatures(BaseModel):
    model_key: Literal['Citroën','Peugeot','PGO','Renault','Audi','BMW','Mercedes','Opel','Volkswagen','Ferrari','Mitsubishi','Nissan','SEAT','Subaru','Toyota','other'] 
    mileage: Union[int, float]
    engine_power: Union[int, float]
    fuel: Literal['diesel','petrol','other']
    paint_color: Literal['black','grey','white','red','silver','blue','beige','brown','other']
    car_type: Literal['convertible','coupe','estate','hatchback','sedan','subcompact','suv','van']
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

@app.post("/predict", tags=["Predictions"])
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
        run_id = 'f6a7267dffd6426ca9d5c99cec6d328d'
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
