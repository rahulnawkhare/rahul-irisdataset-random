from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("iris_model.pkl")

# Flower Names
flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

# Add HERE 👇
flower_images = {
    "Setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Home Page
@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction="",
        confidence="",
        image=""
    )

# Prediction Route
@app.route("/predict_form", methods=["POST"])
def predict_form():

    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])

    features = [[sl, sw, pl, pw]]

    prediction = model.predict(features)
    probability = model.predict_proba(features)

    flower = flower_names[prediction[0]]

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    return render_template(
        "index.html",
        prediction=f"Predicted Flower: {flower}",
        confidence=f"Confidence: {confidence}%",
        image=flower_images[flower]   # 👈 Use image here
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )