import streamlit as st
import pickle
import pandas as pd

# Load the trained model only once to save memory
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Build the UI
st.title("🏠 House Price Predictor")
st.write("Enter the property details below to predict its market value.")

# Create input fields in two columns
col1, col2 = st.columns(2)

with col1:
    # value=None forces the box to be completely empty. 
    # placeholder gives them a hint of what to type!
    area = st.number_input("Area (sq ft)", min_value=0, value=None, step=10, placeholder="e.g., 7420")
    bedrooms = st.number_input("Bedrooms", min_value=0, value=None, step=1, placeholder="e.g., 4")
    bathrooms = st.number_input("Bathrooms", min_value=0, value=None, step=1, placeholder="e.g., 2")
    stories = st.number_input("Stories", min_value=0, value=None, step=1, placeholder="e.g., 2")
    parking = st.number_input("Parking Spaces", min_value=0, value=None, step=1, placeholder="e.g., 1")

with col2:
    # index=None forces the radio buttons to be unselected by default
    guestroom = st.radio("Guestroom", ["No", "Yes"], index=None)
    basement = st.radio("Basement", ["No", "Yes"], index=None)
    airconditioning = st.radio("Air Conditioning", ["No", "Yes"], index=None)
    furnishingstatus = st.radio("Furnishing Status", ["Furnished", "Unfurnished"], index=None)

# Calculate Button
if st.button("Calculate Price", type="primary"):
    
    # SAFETY CHECK: Check if ANY field is still equal to None (left blank)
    all_inputs = [area, bedrooms, bathrooms, stories, parking, guestroom, basement, airconditioning, furnishingstatus]
    
    if None in all_inputs:
        st.warning("⚠️ Please fill out every single box and select all options before calculating!")
    else:
        try:
            # Convert text answers to 1s and 0s for the model
            is_unfurnished = 1 if furnishingstatus == 'Unfurnished' else 0
            yes_no_map = {"Yes": 1, "No": 0}
            
            # Match the exact column names your model.pkl expects
            input_features = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'stories': stories,
                'guestroom': yes_no_map[guestroom], 
                'basement': yes_no_map[basement],   
                'airconditioning': yes_no_map[airconditioning], 
                'parking': parking,
                'furnishingstatus_unfurnished': is_unfurnished
            }

            # Convert to DataFrame and Predict
            df = pd.DataFrame([input_features])
            prediction = model.predict(df)[0]
            
            # Display the result
            st.success(f"**Predicted Price: ${prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
