from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)
# Load the trained Linear Regression model
# Ensure 'model.pkl' is in the same directory as this script
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the user request
        data = request.get_json()

        # Extract features. 
        # We handle 'furnishingstatus' to map it to 'furnishingstatus_unfurnished'
        is_unfurnished = 1 if data.get('furnishingstatus', '').lower() == 'unfurnished' else 0

        # Create a dictionary matching the exact feature names the model expects
        input_features = {
            'area': float(data['area']),
            'bedrooms': int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'stories': int(data['stories']),
            'guestroom': int(data['guestroom']), # Assuming 1 for yes, 0 for no
            'basement': int(data['basement']),   # Assuming 1 for yes, 0 for no
            'airconditioning': int(data['airconditioning']), # Assuming 1 for yes, 0 for no
            'parking': int(data['parking']),
            'furnishingstatus_unfurnished': is_unfurnished
        }

        # Convert to a pandas DataFrame (preserves feature names for sklearn)
        df = pd.DataFrame([input_features])

        # Predict the price
        prediction = model.predict(df)

        # Return the predicted value
        return jsonify({
            'status': 'success',
            'predicted_price': round(prediction[0], 2)
        })

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)