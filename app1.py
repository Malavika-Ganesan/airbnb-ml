
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path



MODEL_PATH = Path(__file__).parent / "airbnb.joblib"

st.set_page_config(
    page_title="RentHop Interest Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Rental Listing Interest Predictor")

st.caption(
    "Educational demonstration using a rental-listing dataset."
)

# Load model safely
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file 'airbnb.joblib' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()




neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]
)

neighbourhood = st.text_input(
    "Neighbourhood",
    value="Harlem"
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
    value=(
        "Beautiful renovated apartment with hardwood floors, "
        "elevator, laundry and subway access."
    )
)



if st.button("Predict Interest Level", type="primary"):

    if not neighbourhood.strip():
        st.warning("Please enter a neighbourhood.")
        st.stop()

    # These columns must match the model's training features.
    row = pd.DataFrame([{
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "latitude": latitude,
        "longitude": longitude,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count":
            calculated_host_listings_count,
        "availability_365": availability_365,
        "days_since_review": days_since_review
    }])

    try:
        prediction = model.predict(row)[0]

        st.success(
            f"Predicted interest level: {str(prediction).upper()}"
        )

        # Display probabilities when supported by the model.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(row)[0]

            probability_df = pd.DataFrame({
                "Interest Level": model.classes_,
                "Probability": [
                    f"{p:.2%}" for p in probabilities
                ]
            })

            st.subheader("Prediction Probabilities")
            st.table(probability_df)

        else:
            st.info(
                "This model does not support probability predictions."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.info(
    "Educational demonstration only. Predictions are estimates, "
    "not guaranteed market outcomes."
)
```
