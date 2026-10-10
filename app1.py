import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "airbnb.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="RentHop Interest Predictor", page_icon="🏠")

st.title("🏠 Rental Listing Interest Predictor")
st.caption("Educational demonstration using the RentHop rental-listing dataset.")

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    "Harlem"
)

room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)

latitude = st.number_input(
    "Latitude",
    min_value=40.0,
    max_value=41.0,
    value=40.75,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=-75.0,
    max_value=-73.0,
    value=-73.98,
    format="%.6f"
)

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=2
)

number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    max_value=10000,
    value=10
)

reviews_per_month = st.number_input(
    "Reviews Per Month",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

calculated_host_listings_count = st.number_input(
    "Host Listing Count",
    min_value=1,
    max_value=1000,
    value=1
)

availability_365 = st.number_input(
    "Availability (Days Per Year)",
    min_value=0,
    max_value=365,
    value=180
)

days_since_review = st.number_input(
    "Days Since Last Review",
    min_value=0,
    max_value=5000,
    value=30
)

description = st.text_area(
    "Rental Description",
    "Beautiful apartment near public transportation."
)



description = st.text_area(
    "Airbnb description",
    "Beautiful renovated apartment with hardwood floors, elevator, laundry and subway access."
)

if st.button("Predict Interest Level"):
    row = pd.DataFrame([{
        "neighbourhood_group":neighbourhood_group,
        "neighbourhood":neighbourhood,
        "latitude":latitude,
        "longitude":longitude,
        "room_type":room_type,
        "minimum_nights":minimum_nights,
        "number_of_reviews":number_of_reviews,
        "reviews_per_month":reviews_per_month,
        "calculated_host_listings_count":calculated_host_listings_count,
        "availability_365":availability_365,
        "days_since_review":days_since_review
    }])

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    st.success(f"Predicted interest level: {str(prediction).upper()}")

    probability_df = pd.DataFrame({
        "Interest Level": model.classes_,
        "Probability": [f"{p:.2%}" for p in probabilities]
    })

    st.table(probability_df)

st.info("Educational demonstration only. Predictions should not be treated as guaranteed market outcomes.")
