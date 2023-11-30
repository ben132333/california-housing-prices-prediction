import requests

url = "https://california-housing-prices-prediction-f4i2jppbwa-ew.a.run.app/predict"
# url = "http://localhost:8080/predict" 

example_data = {
    "longitude": -1.42249992,
    "latitude": 0.93580947,
    "housing_median_age": 0.83005085,
    "total_rooms": 0.22152088,
    "total_bedrooms": 0.52739133,
    "population": 0.33158278,
    "households": 0.51451195,
    "median_income": -0.34108801,
    "rooms_per_household": -0.4034185,
    "bedrooms_per_household": -0.01572762,
    "population_per_household": -0.09008839,
    "ocean_proximity_less_than_one_hour_ocean": 0.0,
    "ocean_proximity_inland": 0.0,
    "ocean_proximity_island": 0.0,
    "ocean_proximity_near_bay": 0.0,
    "ocean_proximity_near_ocean": 1.0
}

"""
Note: example_data is already processed by the data processing pipeline in 
train_model.py. This is currenlty required but the next API update will 
give a better user experience by allowing the original district data to be used.
"""

response = requests.post(url, json=example_data)

print(response.json())
