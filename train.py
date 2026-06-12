from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris Dataset
iris = load_iris()

# Features
X = iris.data

# Labels
y = iris.target

# Create Model
model = RandomForestClassifier()

# Train Model
model.fit(X, y)

# Save Model
joblib.dump(model, "iris_model.pkl")

print("Model trained and saved successfully")