
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    logi = joblib.load('logistic_regression_model.joblib')
except FileNotFoundError:
    st.error("Model file 'logistic_regression_model.joblib' not found. Please ensure the model is trained and saved in the same directory as this app.py file.")
    st.stop()

st.title("Delivery Delay Prediction App")

st.write("""
This app predicts whether a delivery will be delayed based on various input features.
""")

st.sidebar.header("Input Features")

def user_input_features():
    delivery_distance = st.sidebar.slider("Delivery Distance (km)", 0.0, 100.0, 30.0)
    traffic_congestion = st.sidebar.slider("Traffic Congestion (1-5, 5 being highest)", 1, 5, 3)
    weather_condition = st.sidebar.slider("Weather Condition (1-5, 5 being worst)", 1, 5, 3)
    delivery_slot = st.sidebar.slider("Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)", 1, 3, 2)
    driver_experience = st.sidebar.slider("Driver Experience (years)", 0, 30, 10)
    num_stops = st.sidebar.slider("Number of Stops", 1, 15, 5)
    vehicle_age = st.sidebar.slider("Vehicle Age (years)", 0, 20, 5)
    road_condition_score = st.sidebar.slider("Road Condition Score (1-5, 5 being best)", 1, 5, 3)
    package_weight = st.sidebar.slider("Package Weight (kg)", 0.0, 50.0, 10.0)
    fuel_efficiency = st.sidebar.slider("Fuel Efficiency (km/L)", 5.0, 20.0, 12.0)
    warehouse_processing_time = st.sidebar.slider("Warehouse Processing Time (minutes)", 10, 100, 50)

    data = {
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# Make prediction
prediction = logi.predict(input_df)
prediction_proba = logi.predict_proba(input_df)

st.subheader("Prediction")
delay_status = "Delayed" if prediction[0] == 1 else "Not Delayed"
st.write(f"The delivery is predicted to be: **{delay_status}**")

st.subheader("Prediction Probability")
st.write(f"Probability of Not Delayed: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Delayed: {prediction_proba[0][1]:.2f}")

