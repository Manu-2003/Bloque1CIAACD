# Importar librerias
from flask import Flask, request, jsonify, render_template
import numpy as np
from load import joblib
# File managment
import os 
from werkzeug.utils import secure_filename

# Load model
dt = joblib.load('dt1_ml.joblib')
#Create Flask app
server = Flask(_name_)

#Define a route to send JSON data

@server.route('/predictjson', methods = ['POST'])

def predictjson():
    # Procesar los datos de entrada
    data = request.json 
    print(data)
    inputData = np.array([
        data['ph'],
        data['sulphates'],
        data['alcohol']
    ])
    # Predecir utilizando la entrada y el modelo
    result = dt.predict(inputData.reshape(1,-1))
    # Enviar respuesta
    return jsonify({'Prediction' : str(result[0])})

if __name__ == '_main_':
    server.run(debug= False, host= '0,0,0,0', port= 8080)