import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

app = Flask(__name__)

def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    input_data = [0] * len(X.columns)
    for symptom in symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = encoder.inverse_transform([final_rf_model.predict(input_data)])[0]
    nb_prediction = encoder.inverse_transform([final_nb_model.predict(input_data)])[0]
    svm_prediction = encoder.inverse_transform([final_svm_model.predict(input_data)])[0]

    predictions_list = [rf_prediction, nb_prediction, svm_prediction]
    prediction_counter = Counter(predictions_list)
    final_prediction = prediction_counter.most_common(1)[0][0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms = request.form["symptoms"]
        predictions = predictDisease(symptoms)
        return render_template("result.html", predictions=predictions)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
