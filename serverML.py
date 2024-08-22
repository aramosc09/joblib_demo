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