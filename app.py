from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("models/credit_card_model.pkl")
encoders = joblib.load("models/encoders.pkl")
print(type(encoders))
print(encoders)



@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    try:
        print(request.form)
        data = {
            "ID": int(request.form["ID"]),

            "CODE_GENDER": request.form["CODE_GENDER"],
            "FLAG_OWN_CAR": request.form["FLAG_OWN_CAR"],
            "FLAG_OWN_REALTY": request.form["FLAG_OWN_REALTY"],

            "CNT_CHILDREN": int(request.form["CNT_CHILDREN"]),

            "AMT_INCOME_TOTAL": float(request.form["AMT_INCOME_TOTAL"]),

            "NAME_INCOME_TYPE": request.form["NAME_INCOME_TYPE"],
            "NAME_EDUCATION_TYPE": request.form["NAME_EDUCATION_TYPE"],
            "NAME_FAMILY_STATUS": request.form["NAME_FAMILY_STATUS"],
            "NAME_HOUSING_TYPE": request.form["NAME_HOUSING_TYPE"],

            "FLAG_MOBIL": int(request.form["FLAG_MOBIL"]),
            "FLAG_WORK_PHONE": int(request.form["FLAG_WORK_PHONE"]),
            "FLAG_PHONE": int(request.form["FLAG_PHONE"]),
            "FLAG_EMAIL": int(request.form["FLAG_EMAIL"]),

            "OCCUPATION_TYPE": request.form["OCCUPATION_TYPE"],

            "CNT_FAM_MEMBERS": float(request.form["CNT_FAM_MEMBERS"]),

            "AGE": float(request.form["AGE"]),

            "YEARS_EMPLOYED": float(request.form["YEARS_EMPLOYED"]),

        }

        df = pd.DataFrame([data])

        categorical_columns = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "FLAG_MOBIL",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "OCCUPATION_TYPE"
        ]

        print(encoders.keys())

        for col in categorical_columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        print("\n===== DATA SENT TO MODEL =====")
        print(df)
        print(df.dtypes)

        proba = model.predict_proba(df)[0]
        print("Probabilities [Approved, Rejected]:", proba)

        # Only reject if the model is at least 65% confident of risk
        reject_probability = proba[1]

        if reject_probability >= 0.50:
            result = "Rejected"
        else:
            result = "Approved"

        print("Prediction:", result)
        return render_template("result.html", prediction=result)


    except Exception as e:
        return f"<h2>Error:</h2><pre>{e}</pre>"


if __name__ == "__main__":
    app.run(debug=True)