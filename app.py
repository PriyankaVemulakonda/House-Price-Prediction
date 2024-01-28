import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    income = float(request.form['income'])
    house_age = float(request.form['house_age'])
    num_rooms = float(request.form['num_rooms'])
    num_bedrooms = float(request.form['num_bedrooms'])
    population = float(request.form['population'])
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Avg. Area Income': [income],
        'Avg. Area House Age': [house_age],
        'Avg. Area Number of Rooms': [num_rooms],
        'Avg. Area Number of Bedrooms': [num_bedrooms],
        'Area Population': [population]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
