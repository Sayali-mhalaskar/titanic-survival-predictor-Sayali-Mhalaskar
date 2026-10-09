
# ============================================================
# PRACTICAL 7: PYTHON FLASK APPLICATION DEPLOYMENT
# Dataset: Titanic train.csv
# ML Model: Decision Tree Classifier
# Framework: Flask
# ============================================================

import os
import pandas as pd
from flask import Flask, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

app = Flask(__name__)

file_path = os.path.join(os.path.dirname(__file__), "train.csv")

df = pd.read_csv(file_path)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)
print("Dataset shape:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df = df.drop("Cabin", axis=1)
df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)

for column in df.columns:
    if df[column].dtype == "bool":
        df[column] = df[column].astype(int)

print("\nData preprocessing completed.")
print("Processed columns:", df.columns.tolist())

# ------------------------------------------------------------
# 3. SPLIT FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. TRAIN MACHINE LEARNING MODEL
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\nModel training completed successfully!")
print("Model Test Accuracy:", round(accuracy * 100, 2), "%")

# ------------------------------------------------------------
# 5. HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Titanic Survival Prediction</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>
            body {
                font-family: Arial, sans-serif;
                background: #eef2f7;
                text-align: center;
                padding: 25px 10px;
            }

            .container {
                background: white;
                max-width: 450px;
                margin: 20px auto;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px #cbd5e1;
            }

            h1 { color: #174ea6; }

            label {
                display: block;
                text-align: left;
                margin-top: 12px;
                font-weight: bold;
            }

            input, select {
                width: 100%;
                box-sizing: border-box;
                padding: 10px;
                margin-top: 5px;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
            }

            button {
                margin-top: 22px;
                padding: 12px 20px;
                width: 100%;
                background: #174ea6;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }

            button:hover { background: #123b7a; }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>Titanic Survival Prediction</h1>
            <p>Enter passenger details to predict survival.</p>

            <form action="/predict" method="post">

                <label>Passenger Class:</label>
                <select name="pclass">
                    <option value="1">1st Class</option>
                    <option value="2">2nd Class</option>
                    <option value="3" selected>3rd Class</option>
                </select>

                <label>Gender:</label>
                <select name="sex">
                    <option value="female">Female</option>
                    <option value="male">Male</option>
                </select>

                <label>Age:</label>
                <input type="number" name="age"
                       min="1" max="100" value="25" required>

                <label>Siblings / Spouses:</label>
                <input type="number" name="sibsp"
                       min="0" value="0" required>

                <label>Parents / Children:</label>
                <input type="number" name="parch"
                       min="0" value="0" required>

                <label>Fare:</label>
                <input type="number" name="fare"
                       min="0" step="0.01" value="30" required>

                <label>Embarked:</label>
                <select name="embarked">
                    <option value="S">Southampton</option>
                    <option value="C">Cherbourg</option>
                    <option value="Q">Queenstown</option>
                </select>

                <button type="submit">Predict Survival</button>
            </form>
        </div>
    </body>
    </html>
    """

# ------------------------------------------------------------
# 6. PREDICTION ROUTE
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pclass = int(request.form["pclass"])
        sex = request.form["sex"]
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = request.form["embarked"]

        if (
            pclass not in [1, 2, 3]
            or sex not in ["male", "female"]
            or embarked not in ["S", "C", "Q"]
            or not 1 <= age <= 100
            or sibsp < 0
            or parch < 0
            or fare < 0
        ):
            return "Invalid passenger details. Please go back and try again.", 400

        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Sex_male": [1 if sex == "male" else 0],
            "Embarked_Q": [1 if embarked == "Q" else 0],
            "Embarked_S": [1 if embarked == "S" else 0]
        })

        input_data = input_data[X.columns]

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Predicted to SURVIVE"
            color = "#15803d"
        else:
            result = "Predicted NOT TO SURVIVE"
            color = "#b91c1c"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #eef2f7;
                    text-align: center;
                    padding: 40px 10px;
                }}
                .result {{
                    background: white;
                    max-width: 500px;
                    margin: auto;
                    padding: 35px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px #cbd5e1;
                }}
                h2 {{ color: {color}; }}
                a {{ color: #174ea6; }}
            </style>
        </head>
        <body>
            <div class="result">
                <h1>Prediction Result</h1>
                <h2>{result}</h2>
                <p>Prediction generated by the Decision Tree model.</p>
                <a href="/">Try Another Passenger</a>
            </div>
        </body>
        </html>
        """

    except (ValueError, KeyError):
        return "Invalid input. Please return to the home page.", 400

# ------------------------------------------------------------
# 7. ABOUT PAGE
# ------------------------------------------------------------

@app.route("/about")
def about():
    return """
    <h1>About This Application</h1>
    <p>Python Flask web application.</p>
    <p>Machine Learning Model: Decision Tree Classifier.</p>
    <a href="/">Return to Home</a>
    """

# ------------------------------------------------------------
# 8. START APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )