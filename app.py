import joblib
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint for predictions
# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features = np.array([float(data[key]) for key in data.keys()]).reshape(1, -1)

        prediction = model.predict(features)

        result_label = "The person is suffering from Heart Disease" if prediction[0] == 1 else "  The Person Doesnot  Have Heart Disease"

        return render_template('index.html', prediction=f'Prediction: {result_label}', features=data)
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
