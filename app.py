from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[
        data['pregnancies'], data['glucose'], data['bp'],
        data['skin_thickness'], data['insulin'],
        data['bmi'], data['dpf'], data['age']
    ]])
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
