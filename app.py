import gradio as gr
import pickle
import pandas as pd

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_price(area, bedrooms, bathrooms, stories, parking, guestroom, basement, airconditioning, furnishingstatus):
    """
    This function takes the inputs from the Gradio UI, formats them to match
    the exact feature names the model expects, and returns the prediction.
    """
    try:
        # Convert "Furnished" / "Unfurnished" to 0 or 1
        is_unfurnished = 1 if furnishingstatus == 'Unfurnished' else 0
        
        # Convert "Yes" / "No" strings to 1 or 0 for the model
        yes_no_map = {"Yes": 1, "No": 0}
        
        # Create a dictionary matching the exact feature names the model expects
        input_features = {
            'area': float(area),
            'bedrooms': int(bedrooms),
            'bathrooms': int(bathrooms),
            'stories': int(stories),
            'guestroom': yes_no_map[guestroom], 
            'basement': yes_no_map[basement],   
            'airconditioning': yes_no_map[airconditioning], 
            'parking': int(parking),
            'furnishingstatus_unfurnished': is_unfurnished
        }

        # Convert to a pandas DataFrame (preserves feature names for sklearn)
        df = pd.DataFrame([input_features])

        # Predict the price
        prediction = model.predict(df)[0]

        # Format the output as a clean currency string
        return f"${round(prediction, 2):,}"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Define the input components for the Gradio interface
inputs = [
    gr.Number(label="Area (sq ft)", value=7420),
    gr.Number(label="Bedrooms", value=4),
    gr.Number(label="Bathrooms", value=2),
    gr.Number(label="Stories", value=2),
    gr.Number(label="Parking Spaces", value=1),
    gr.Radio(choices=["No", "Yes"], label="Guestroom", value="No"),
    gr.Radio(choices=["No", "Yes"], label="Basement", value="No"),
    gr.Radio(choices=["No", "Yes"], label="Air Conditioning", value="Yes"),
    gr.Radio(choices=["Furnished", "Unfurnished"], label="Furnishing Status", value="Unfurnished")
]

# Define the output component
output = gr.Text(label="Predicted Price", show_label=True)

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="🏠 House Price Predictor",
    description="Enter the property details below to predict its market value.",
    allow_flagging="never"
)

# Launch the app
if __name__ == '__main__':
    demo.launch()