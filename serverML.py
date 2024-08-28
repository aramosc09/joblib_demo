from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import pandas as pd
import os
from werkzeug.utils import secure_filename

# Load model
dir = os.getcwd()
print(os.path.join(dir,'rf1.joblib'))
dt = joblib.load(os.path.join(dir,'rf1.joblib'))

# Create Flask
server = Flask(__name__)

# Define a route to send JSON data
@server.route('/predictjson',methods=['POST'])
def predictjson():
    # Process data
    data = request.json
    # print(data)
    inputData = np.array([
        data['alcohol'],
        data['volatile acidity'],
        data['pH']
    ])

    # Predict using input & model
    result = dt.predict([inputData.reshape(1, -1)])

    # Send answer
    return jsonify({'Prediction': str(result[0])})

if __name__ == '__main__':
    server.run(debug=False,host='0.0.0.0',port=8080)