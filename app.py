import re
import pandas as pd

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from flask import Flask, request, jsonify

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


def getDescription():
    description_list = {}
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]
    return description_list


def getPrecautionDict():
    precautionDictionary = {}
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return precautionDictionary


description_list = getDescription()
precautionDictionary = getPrecautionDict()


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

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


# if __name__ =="__main__":
#     app.run(debug=True)
