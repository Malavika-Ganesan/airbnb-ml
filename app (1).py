import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "final_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏠")

st.title("🏠 Airbnb Price Predictor")
st.caption("Predict the Airbnb listing price using the trained machine-learning model.")

minimum_nights = st.number_input("Minimum Nights", 1, 365, 3, 1)
number_of_reviews = st.number_input("Number of Reviews", 0, 1000, 10, 1)
latitude = st.number_input("Latitude", value=40.75, format="%.6f")
longitude = st.number_input("Longitude", value=-73.98, format="%.6f")
reviews_per_month = st.number_input("Reviews per Month", 0.0, 100.0, 1.0, 0.1)
calculated_host_listings_count = st.number_input(
    "Calculated Host Listings Count", 1, 1000, 1, 1
)
availability_365 = st.number_input("Availability (365 days)", 0, 365, 100, 1)

neighbourhood_group = st.text_input("Neighbourhood Group", "Manhattan")
neighbourhood = st.text_input("Neighbourhood", "Upper West Side")

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

if st.button("Predict Price"):
    row = pd.DataFrame([{
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "latitude": latitude,
        "longitude": longitude,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "room_type": room_type
    }])

    prediction = model.predict(row)[0]

    st.success(f"Predicted Airbnb Price: ${prediction:.2f}")

st.info("Educational demonstration. The predicted price is an estimate from the trained model.")
