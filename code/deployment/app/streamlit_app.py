import json
import streamlit as st
import requests
import pandas as pd


# Define the FastAPI endpoint URL
API_URL = "http://api:8000/predict/"

st.title("Blond hair image classifier")
st.write("Upload a CSV file with the following columns: `ID`, `image_id`")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    input_df = pd.read_csv(uploaded_file)
    # Ensure that the CSV has the correct columns

    if st.button("Predict"):
        # Convert the file to a bytes stream and send it as a POST request
        response = requests.post(API_URL, json={"dataframe": input_df.to_dict()})
        if response.status_code == 200:
            predictions = response.json().get("output").get("predictions")
            input_df['Prediction'] = predictions
            for i in predictions.items():
                input_df.loc[int(i[0]),'Prediction'] = i[1]  # Add predictions to the DataFrame
            st.write("Predictions:")
            st.dataframe(input_df)  # Display the DataFrame with predictions
        else:
            st.error(f"Error: {response.status_code}")
