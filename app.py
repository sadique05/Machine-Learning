import streamlit as st
import pickle
import json
import numpy as np

# Load the model and column information
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]  # Assuming locations start from the 4th column

# Function to predict price (same as in your notebook)
def predict_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.title("🏠 House Price Prediction")

# Introduction
st.markdown("""
    Welcome to the House Price Prediction app! 
    This tool helps you estimate the price of a house based on its location, size, and features.
    """)

# Input features in columns
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", __locations)
    sqft = st.number_input("📏 Total Square Feet Area", min_value=0)

with col2:
    bath = st.number_input("🚿 Number of Bathrooms", min_value=0)
    bhk = st.number_input("🛏️ Number of BHK", min_value=0)

# Prediction button
if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Predicted Price: ₹{price:,.2f} Lakhs")

# Footer
st.markdown("---")
st.markdown("© 2023  House Price Prediction App. All rights reserved.")