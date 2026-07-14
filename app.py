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

# Create input fields
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=0.0, value=0.0, step=10.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=0, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=0, step=1)
    stories = st.number_input("Stories", min_value=0, value=0, step=1)
    parking = st.number_input("Parking Spaces", min_value=0, value=0, step=1)

with col2:
    guestroom = st.radio("Guestroom", ["No", "Yes"])
    basement = st.radio("Basement", ["No", "Yes"])
    airconditioning = st.radio("Air Conditioning", ["No", "Yes"])
    furnishingstatus = st.radio("Furnishing Status", ["Furnished", "Unfurnished"])

# Calculate Button
if st.button("Calculate Price", type="primary"):
    # Safety check
    if area == 0 or bedrooms == 0 or bathrooms == 0:
        st.warning("⚠️ Please fill out the Area, Bedrooms, and Bathrooms to get an accurate prediction.")
    else:
        try:
            # Data conversion
            is_unfurnished = 1 if furnishingstatus == 'Unfurnished' else 0
            yes_no_map = {"Yes": 1, "No": 0}
            
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

            df = pd.DataFrame([input_features])
            prediction = model.predict(df)[0]
            
            # Display result
            st.success(f"**Predicted Price: ${prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
