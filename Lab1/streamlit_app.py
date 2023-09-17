import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('clf_svm.joblib')

def main():
    st.title("Seed Type Prediction")

    # Define input fields
    columns_names = ['area', 'perimeter', 'compactness', 'lengthOfKernel', 'widthOfKernel', 'asymmetryCoefficient', 'lengthOfKernelGroove']
    inputs = {feature: st.number_input(f"Enter {feature}", value=0.0) for feature in columns_names}

    # Predict button
    if st.button("Predict"):
        data = [inputs[feature] for feature in columns_names]
        result = model.predict([data])
        st.write(f"Predicted seed type: {result[0] + 1}")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file, delim_whitespace=True, header=None).values
        predictions = model.predict(data).tolist()
        for i in range(len(predictions)):
            predictions[i] += 1

        # Convert data and predictions to a list of dictionaries
        data_dict_list = []
        for i in range(len(data)):
            row_dict = {
                'area': data[i][0],
                'perimeter': data[i][1],
                'compactness': data[i][2],
                'lengthOfKernel': data[i][3],
                'widthOfKernel': data[i][4],
                'asymmetryCoefficient': data[i][5],
                'lengthOfKernelGroove': data[i][6],
                'predictions': predictions[i]
            }
            data_dict_list.append(row_dict)

        st.write(pd.DataFrame(data_dict_list))

if __name__ == "__main__":
    main()