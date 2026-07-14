import gradio as gr
import pickle
import pandas as pd

# Load the trained Linear Regression model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_price(area, bedrooms, bathrooms, stories, parking, guestroom, basement, airconditioning, furnishingstatus):
    try:
        # SAFETY CHECK: Ensure no fields are left blank
        inputs_list = [area, bedrooms, bathrooms, stories, parking, guestroom, basement, airconditioning, furnishingstatus]
        if "" in inputs_list or None in inputs_list:
            return "⚠️ Error: Please fill out all fields before calculating!"

        # Convert "Furnished" / "Unfurnished" to 0 or 1
        is_unfurnished = 1 if furnishingstatus == 'Unfurnished' else 0
        
        # Convert "Yes" / "No" strings to 1 or 0 for the model
        yes_no_map = {"Yes": 1, "No": 0}
        
        # Create a dictionary, explicitly converting the text inputs into numbers
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

        # Convert to a pandas DataFrame
        df = pd.DataFrame([input_features])

        # Predict the price
        prediction = model.predict(df)[0]

        # Format the output as a clean currency string
        return f"${round(prediction, 2):,}"
        
    except ValueError:
        return "⚠️ Error: Please ensure you only type numbers (no letters) in the text boxes!"
    except Exception as e:
        return f"Error: {str(e)}"

# Define inputs using Textbox instead of Number to force them to be blank
inputs = [
    gr.Textbox(label="Area (sq ft)"),
    gr.Textbox(label="Bedrooms"),
    gr.Textbox(label="Bathrooms"),
    gr.Textbox(label="Stories"),
    gr.Textbox(label="Parking Spaces"),
    gr.Radio(choices=["No", "Yes"], label="Guestroom"),
    gr.Radio(choices=["No", "Yes"], label="Basement"),
    gr.Radio(choices=["No", "Yes"], label="Air Conditioning"),
    gr.Radio(choices=["Furnished", "Unfurnished"], label="Furnishing Status")
]

# Define the output component
output = gr.Text(label="Predicted Price", show_label=True)

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=output,
    title="🏠 House Price Predictor",
    description="Enter the property details below to predict its market value."
)

# Launch the app
if __name__ == '__main__':
    demo.launch()