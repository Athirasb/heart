import joblib
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the CSV data
heart_data = pd.read_csv('data.csv')

# Feature and target variables
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the Data into Training Data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the trained model using joblib
joblib.dump(model, 'model.joblib')

# Flask app
app = Flask(__name__)

# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(port=5000)
