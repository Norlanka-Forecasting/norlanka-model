import json
import flask
import pickle
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMAResults

import numpy as np
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Norlanka API</h1><p>Norlanka API - Prediction/Forecasting</p>"


@app.route('/name', methods=['GET'])
def name():
    return "<h1>SIPFAA - Differentiation plan</h1>"


@app.route('/getSales', methods=['GET'])
def get_sales():
    month = request.headers.get('month')
    print("Month", month)
    # load model
    # with open('sales_forecasting.pkl', 'rb') as file:
    #     data = pickle.load(file)
    # load model
    data = ARIMAResults.load('sales_forecasting.pkl')
    fc_series = data['forecast_series']
    Sales = fc_series.get(key=month)
    print("Sales", Sales)
    return str(Sales)

@app.route('/predictSales', methods=['GET'])
def predict_sales():
    Pcs_Pk = request.headers.get('pcsPk')
    UnitPrice = request.headers.get('unitPrice')
    OTIF = request.headers.get('otif')
    Embelishment_Cost = request.headers.get('embelishmentCost')

    # Define the array with all input parameters
    inputs = np.array([[Pcs_Pk, UnitPrice, OTIF,
                        Embelishment_Cost]])
    inputs = inputs.astype(float)
    # load model
    with open('randomforest_model.pkl', 'rb') as file:
        data = pickle.load(file)


    random_forest_reg = data["model"]
    # Make the prediction
    gross = random_forest_reg.predict(inputs)
    print("Predicted Sales : ", round(gross[0]))
    return str(round(gross[0]))



app.run(debug=False, host='192.168.8.118', port=5000)
