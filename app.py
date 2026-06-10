# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from user
    size = float(request.form['size'])
    
    # Make prediction
    prediction = model.predict([[size]])
    result = round(prediction[0], 2)
    
    return render_template('index.html', 
                          prediction=f"Predicted Price: ₹{result}")


if __name__ == '__main__':
 app.run(debug=True, port=5001) 