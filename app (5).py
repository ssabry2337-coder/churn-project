
import joblib
import numpy as np

model = joblib.load("model.pkl")

input_data = np.array([[1, 0, 0, 1]])

prediction = model.predict(input_data)

print("Prediction:", prediction)
