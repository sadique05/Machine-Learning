import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Load the dataset ---
df1 = pd.read_csv("bengaluru_house_prices.csv")
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df3 = df2.dropna()

# Convert 'size' to 'bhk'
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# Function to convert sqft range to a single number
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None   

df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]

# Feature Engineering
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
df5.location = df5.location.apply(lambda x: x.strip())

# Handling locations with low frequency
location_stats = df5['location'].value_counts(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove outliers based on total_sqft/bhk ratio
df6 = df5[~(df5.total_sqft / df5.bhk < 300)]

# Function to remove price per sqft outliers
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)

# Function to remove BHK outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)

# Remove outliers based on number of bathrooms
df9 = df8[df8.bath < df8.bhk + 2]

# Prepare final dataset
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

X = df12.drop(['price'], axis='columns')
y = df12.price

# Split data for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train Linear Regression model
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# --- Load Pretrained Model and Column Data ---
if os.path.exists("banglore_home_prices_model.pickle"):
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)
else:
    st.error("Model file not found. Please check the deployment environment.")

if os.path.exists("columns.json"):
    with open("columns.json", "r") as f:
        columns_data = json.load(f)
        columns = [col.lower() for col in columns_data['data_columns']]  # Convert columns to lowercase
else:
    st.error("Columns file not found. Please check the deployment environment.")

# --- Streamlit App ---
st.set_page_config(page_title="Bangalore House Price Predictor", page_icon="🏠")

# --- Header ---
st.title("🏠 Bangalore House Price Predictor")
st.markdown("Enter the details of the house to predict its price.")

# --- Input fields ---
col1, col2, col3 = st.columns(3)
with col1:
    location = st.selectbox("Location", options=sorted(list(set(df5['location'].values))))
with col2:
    sqft = st.number_input("Total Square Feet", min_value=100, step=100)
with col3:
    bath = st.number_input("Number of Bathrooms", min_value=1, step=1)

col4, _, _ = st.columns(3)
with col4:
    bhk = st.number_input("Number of BHK", min_value=1, step=1)

# --- Prediction button ---
if st.button("Predict Price"):
    try:
        loc_index = columns.index(location.lower()) if location.lower() in columns else -1

        # Create input array
        x = np.zeros(len(columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        if loc_index != -1:
            x[loc_index] = 1  # Set location value to 1

        # Reshape input and predict price
        predicted_price = model.predict(np.array(x).reshape(1, -1))[0]

        st.success(f"Predicted Price: ₹ {predicted_price:,.2f} Lakhs")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
