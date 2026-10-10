
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
    page_title="RentHop Interest Predictor",
    page_icon="🏠",
    layout="centered"
)



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "airbnb.joblib"

if not MODEL_PATH.exists():
    st.error("Model file 'airbnb.joblib' was not found.")
    st.write("Expected location:", str(MODEL_PATH))
    st.info(
        "Upload airbnb.joblib to the same GitHub folder "
        "as app1.py, then restart the Streamlit app."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()



st.title("🏠 Rental Listing Interest Predictor")

st.caption(
    "Educational demonstration using a rental-listing dataset."
)



st.subheader("Enter Rental Listing Details")

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

    # Create input DataFrame
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
        # Generate prediction
        prediction = model.predict(row)[0]

        st.success(
            f"Predicted interest level: {str(prediction).upper()}"
        )

        # Display class probabilities if supported
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(row)[0]

            probability_df = pd.DataFrame({
                "Interest Level": model.classes_,
                "Probability": [
                    f"{probability:.2%}"
                    for probability in probabilities
                ]
            })

            st.subheader("Prediction Probabilities")
            st.table(probability_df)

        else:
            st.info(
                "Probability estimates are not available "
                "for this model."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        st.warning(
            "Check whether the input columns and preprocessing "
            "match those used when training the model."
        )



st.info(
    "Educational demonstration only. Predictions are estimates "
    "and should not be treated as guaranteed market outcomes."
)
```
