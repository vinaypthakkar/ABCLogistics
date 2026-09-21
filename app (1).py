
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the features to predict delivery delay:')

# Get feature names from the original X DataFrame
# This assumes X is available in the environment where the model was trained
# For a standalone app, these should be hardcoded or loaded from a config
feature_names = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    if feature in ['Traffic_Congestion', 'Weather_Condition', 'Delivery_Slot', 'Num_Stops', 'Vehicle_Age', 'Road_Condition_Score']:
        input_data[feature] = st.slider(f'Enter {feature.replace("_", " ")}', min_value=1, max_value=10, value=5)
    elif feature in ['Delivery_Distance', 'Package_Weight', 'Fuel_Efficiency', 'Warehouse_Processing_Time']:
        input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}', value=10.0)
    elif feature == 'Driver_Experience':
        input_data[feature] = st.slider(f'Enter {feature.replace("_", " ")}', min_value=1, max_value=20, value=10)


# Predict button
if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error(f'Prediction: **Delayed** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Prediction: **On Time** (Probability: {prediction_proba[0][0]:.2f})')

st.write("\n\n---")
st.write("To run this Streamlit app, save this code as `app.py` and then run `streamlit run app.py` in your terminal.")
