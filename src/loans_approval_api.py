from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

import requests
import pandas as pd
import numpy
import pickle

app = Flask(__name__)
api = Api(app)

def totalIncome(X):
    applicant_income = X['ApplicantIncome']
    coapp_income = X['CoapplicantIncome']
    X = X.copy()
    X['total_income'] = applicant_income + coapp_income
    return X

def dropIncome(X):
    return X.drop(['ApplicantIncome', 'CoapplicantIncome'], axis=1)

class applyLog():
    def __init__(self, log_col):
        self.log_col = log_col

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
    
        X = X.copy()
        X[self.log_col] = np.log(X[self.log_col])

        return X

def convertString(data):
    string_data = data.astype(str)
    return string_data


model = pickle.load(open( "loans_approval_model.sav", "rb" ))

class Loans(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict_proba(df)
        # we cannot send numpt array as a result
        return res.tolist()

# assign endpoint
api.add_resource(Loans, '/loans')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)