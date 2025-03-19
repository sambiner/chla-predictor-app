import streamlit as st
import pandas as pd
import pickle

data_file = "CHLA_clean_data_2024_Appointments.csv"
appointments_df = pd.read_csv(data_file)

st.title("CHLA No-Show Appointment Predictor")
st.write("Use the controls on the sidebar to filter appointments and view predictions.")

# Sidebar: Input parameters
st.header("Input Parameters")
clinic = st.selectbox(
    "Clinic Name", options=appointments_df["CLINIC"].unique().tolist()
)
start_date_input = st.text_input("Start Date (mm/dd/yyyy)", value="1/1/2024")
end_date_input = st.text_input("End Date (mm/dd/yyyy)", value="1/31/2024")

# Validate and convert the date strings
try:
    start_date = pd.to_datetime(start_date_input, format="%m/%d/%Y").date()
    end_date = pd.to_datetime(end_date_input, format="%m/%d/%Y").date()
except Exception as e:
    st.error("Invalid date format. Please use mm/dd/yyyy.")
    st.stop()

# Load the 2024 appointments data and convert the appointment date column to datetime
appointments_df["APPT_DATE"] = pd.to_datetime(
    appointments_df["APPT_DATE"], format="%m/%d/%y %H:%M"
)

# Load the saved model pipeline from the pickle file
with open("final_no_show_model.pkl", "rb") as f:
    model = pickle.load(f)

# Filter the appointments based on clinic name and appointment date range
filtered_appts = appointments_df[
    (appointments_df["CLINIC"] == clinic)
    & (appointments_df["APPT_DATE"].dt.date >= start_date)
    & (appointments_df["APPT_DATE"].dt.date <= end_date)
]

if st.button("Submit"):
    if filtered_appts.empty:
        st.write(
            f"No appointments found for the selected criteria: Clinic {clinic}, Start date {start_date}, and End date {end_date}"
        )
    else:
        # Predict the no-show probability using the model pipeline.
        # Note: The pipeline is assumed to include all necessary preprocessing steps.
        X_new = filtered_appts.copy()
        probabilities = model.predict_proba(X_new)[:, 1]
        predictions = ["Y" if p > 0.5 else "N" for p in probabilities]

        # Construct an output table with the required columns
        output = pd.DataFrame(
            {
                "MRN": filtered_appts["MRN"],
                "APPT_ID": filtered_appts["APPT_ID"],
                "Date": filtered_appts["APPT_DATE"].dt.date,
                "Time": filtered_appts["APPT_DATE"].dt.time,
                "No Show": predictions,
                "Prob": probabilities,
            }
        )

        st.write(f"Predicted No-Show Appointments for {clinic}:")
        st.dataframe(output)
