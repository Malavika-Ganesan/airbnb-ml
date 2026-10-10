
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

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.is_file():
    st.error("Model file 'airbnb.joblib' was not found.")

    st.write("Expected model location:")
    st.code(str(MODEL_PATH))

    st.write("Files found in the application folder:")

    try:
        for file in BASE_DIR.iterdir():
            st.write(file.name)
    except OSError:
        st.write("Unable to list application files.")

    st.warning(
        "Upload the trained airbnb.joblib file to the same "
        "GitHub folder as app1.py, then restart the app."
    )

    st.stop()

try:
    model = load_model(str(MODEL_PATH))

except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(f"{type(e).__name__}: {e}")

    st.info(
        "Check whether the model file is valid and whether "
        "its serialization format is compatible with the "
        "installed Python and scikit-learn versions."
    )

    st.stop()


st.title("🏠 Rental Listing Interest Predictor")

st.caption(
    "Educational demonstration using a rental-listing dataset."
)

st.markdown(
    """
    Enter the rental listing details below to predict
    its interest level using the trained machine-learning model.
    """
)


st.subheader("📍 Location Details")

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


st.subheader("🏡 Room Details")

room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)

minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    max_value=365,
    value=2
)


st.subheader("⭐ Review Details")

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

days_since_review = st.number_input(
    "Days Since Last Review",
    min_value=0,
    max_value=5000,
    value=30
)


st.subheader("👤 Host and Availability")

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



st.subheader("📝 Rental Description")

description = st.text_area(
    "Describe the Rental",
    value=(
        "Beautiful renovated apartment with hardwood floors, "
        "elevator, laundry and subway access."
    )
)

st.caption(
    "Note: The description is collected for display but is not "
    "used by this model unless text processing was included "
    "in the original training pipeline."
)


if st.button("🔮 Predict Interest Level", type="primary"):

    if not neighbourhood.strip():
        st.warning("Please enter a neighbourhood.")
        st.stop()

    
    input_data = pd.DataFrame([{
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
        
        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted Interest Level: {str(prediction).upper()}"
        )

        
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = getattr(
                model,
                "classes_",
                range(len(probabilities))
            )

            probability_df = pd.DataFrame({
                "Interest Level": classes,
                "Probability": [
                    f"{probability:.2%}"
                    for probability in probabilities
                ]
            })

            st.subheader("📊 Prediction Probabilities")

            st.table(probability_df)

            st.bar_chart(
                pd.DataFrame({
                    "Probability": probabilities
                }, index=classes)
            )

        else:
            st.info(
                "This model does not support probability predictions."
            )

        # Display the submitted input features
        with st.expander("View Submitted Listing Details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")

        st.code(f"{type(e).__name__}: {e}")

        st.warning(
            "Verify that the model expects these exact feature "
            "names and that its preprocessing matches the "
            "original training pipeline."
        )



st.divider()

st.info(
    "Educational demonstration only. Predictions are estimates "
    "and should not be treated as guaranteed market outcomes."
)

