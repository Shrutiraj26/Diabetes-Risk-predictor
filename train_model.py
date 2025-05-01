import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as diabetes_model.pkl")
