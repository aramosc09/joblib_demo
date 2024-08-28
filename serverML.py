from flask import Flask, request, jsonify, render_template
import numpy as np
import load import joblib
import pandas as pd
import os
from werkzeug.utils import secure_filename

# Load model
dt = joblib.load('/Users/ajelandro/Documents/GitHub/reto7mo/rf1.joblib')

# Create Flask
server = Flask(__name__)

# Define a route to send json data
server.route('/predictjson',methods=['POST'])

# Define a route to send JSON data

@server.route('predictjson',method=['POST'])
def predictjson():
    # Process data
    data = request.json
    # print(data)
    # inputData = np.array([
    #     data['alcohol'],
    #     data['volatile acidity'],
    #     data['pH']
    # ])
    inputData = [data['alcohol'], data['volatile acidity'], data['pH']]

    # Predict using input & model
    result = dt.predict(inputData[0])

    # Send answer
    return jsonify({'Prediction': str(result[0])})

if __name__ == '__main__':
    server.run(debug=False,host='0.0.0.0',port=8080)
