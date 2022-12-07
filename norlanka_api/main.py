import json
import flask
import pickle
import numpy as np
from flask import request
from flask_restful import Resource, Api
from flask_cors import CORS

app = flask.Flask(__name__)
# app.config["DEBUG"] = True
api = Api(app)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Norlanka API</h1><p>Norlanka API - Prediction/Forecasting</p>"

@app.route('/getSales', methods=['GET'])
def get_sales():
    month = request.args.get('month')
    print(month)
    # load model
    with open('sales_forecasting.pkl', 'rb') as file:
        dataLoaded = pickle.load(file)
    fc_series = dataLoaded['forecast_series']
    Sales = round(fc_series.get(key=month))

    resp = flask.Response(str(Sales))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/predictSales', methods=['GET'])
def predict_sales():
    Pcs_Pk = request.args.get('pcsPk')
    UnitPrice = request.args.get('unitPrice')
    Embelishment_Cost = request.args.get('embelishmentCost')

    # Define the array with all input parameters
    inputs = np.array([[Pcs_Pk, UnitPrice,
                        Embelishment_Cost]])
    inputs = inputs.astype(float)
    # load model
    with open('randomforest_model.pkl', 'rb') as file:
        data = pickle.load(file)

    random_forest_reg = data["model"]
    # Make the prediction
    gross = random_forest_reg.predict(inputs)
    print("Predicted Sales : ", round(gross[0]))

    resp = flask.Response(str(round(gross[0])))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

# app.run(debug=False, host='127.0.0.1', port=5000)
if __name__ == '__main__':
    app.run(debug=True)