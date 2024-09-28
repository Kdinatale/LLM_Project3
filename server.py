from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    # Ensure that we received the expected array of features
    try:
        features = data['features']
    except KeyError:
        return jsonify(error="The 'features' key is missing from the request payload."), 400

    # Convert features into the right format and make a prediction
    prediction = model.polarity_scores(features)
    
    compound = int(prediction["compound"])
    
    return_string = "The Sentiment is "

    if compound > 0: 
        return_string += "Positive"
    else: 
        return_string += "Negative"
    
    return return_string

if __name__ == '__main__':
    app.run(debug=True)