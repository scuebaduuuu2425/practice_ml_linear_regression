import joblib
import numpy as np

# Load trained model
model = joblib.load("models/model.pkl")

print("California House Price Prediction")

# User inputs
MedInc = float(input("Median Income: "))
HouseAge = float(input("House Age: "))
AveRooms = float(input("Average Rooms: "))
AveBedrms = float(input("Average Bedrooms: "))
Population = float(input("Population: "))
AveOccup = float(input("Average Occupancy: "))
Latitude = float(input("Latitude: "))
Longitude = float(input("Longitude: "))

# Create input array
data = np.array([[
    MedInc,
    HouseAge,
    AveRooms,
    AveBedrms,
    Population,
    AveOccup,
    Latitude,
    Longitude
]])

# Prediction
prediction = model.predict(data)

print("\nPredicted House Price:", prediction[0])