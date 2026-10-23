from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained Linear SVM model
with open("LinearSVMModel.pkl", "rb") as file:
    model = pickle.load(file)

# Load StandardScaler
with open("Scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Features used while training the model
features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity"
]


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from HTML form
        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        concavity = float(request.form["concavity"])

        # Create DataFrame with same features used during training
        input_data = pd.DataFrame(
            [[radius, texture, perimeter, area, concavity]],
            columns=features
        )

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Dataset mapping:
        # 0 = Malignant
        # 1 = Benign
        if prediction == 0:
            result = "Malignant"
        else:
            result = "Benign"

        return render_template(
            "index.html",
            prediction_text=result,
            radius=radius,
            texture=texture,
            perimeter=perimeter,
            area=area,
            concavity=concavity
        )

    except Exception as e:

        print("Error:", e)

        return render_template(
            "index.html",
            error_text="Please enter valid input values."
        )


if __name__ == "__main__":
    app.run(debug=True)