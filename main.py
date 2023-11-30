from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

rfr = joblib.load("./rfr1.pkl")

"""
TODO: the input data is now required to be processed by the processing pipeline used in 
train_model.py. Allow the origional data features to be sent to the API by adding the 
processing pipeline to the API. It improves the user experience. 
"""

class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    rooms_per_household: float
    bedrooms_per_household: float
    population_per_household: float
    ocean_proximity_less_than_one_hour_ocean: float
    ocean_proximity_inland: float
    ocean_proximity_island: float
    ocean_proximity_near_bay: float
    ocean_proximity_near_ocean: float

@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.longitude, data.latitude, data.housing_median_age, 
        data.total_rooms, data.total_bedrooms, data.population, 
        data.households, data.median_income, data.rooms_per_household,
        data.bedrooms_per_household, data.population_per_household,
        data.ocean_proximity_less_than_one_hour_ocean, data.ocean_proximity_inland,
        data.ocean_proximity_island, data.ocean_proximity_near_bay,
        data.ocean_proximity_near_ocean
    ]]

    prediction = rfr.predict(input_data)

    return {"prediction": prediction[0]}

"""
To run the server locally, use the command:
uvicorn main:app --reload
"""
