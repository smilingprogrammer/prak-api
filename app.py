import pandas as pd

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from utils import sec_predict, description_list, precautionDictionary

app = Flask(__name__)

training = pd.read_csv('Data/Training.csv')

cols = training.columns[:-1]

x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


@app.route('/')
def home():
    return "I am ready!"
@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json['symptoms']
    days = request.json['days']

    symptoms_exp = []
    for symptom in symptoms:
        if symptom in cols:
            symptoms_exp.append(symptom)

    second_prediction = sec_predict(symptoms_exp)

    present_disease = le.inverse_transform(clf.predict([np.isin(cols, symptoms_exp).astype(int)]))[0]
    description = description_list.get(present_disease, "Description not available")
    precautions = precautionDictionary.get(present_disease, ["Precautions not available"])

    response = {
        'predicted_disease': present_disease,
        'description': description,
        'precautions': precautions,
        'second_prediction': second_prediction[0],
    }

    return jsonify(response)


if __name__ =="__main__":
    app.run(debug=True)
