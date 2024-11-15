# main.py

import streamlit as st
from insurance import train_insurance_model

def get_user_inputs():
    """Function to get user inputs from the Streamlit interface."""
    st.header("User Input Parameters")

    # Input fields in the center column
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    sex_input = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker_input = st.selectbox("Smoker", options=["yes", "no"])
    region_input = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    # Map the categorical inputs to the expected integer values
    sex = 1 if sex_input == "male" else 0
    smoker = 1 if smoker_input == "yes" else 0
    region_map = {"northeast": 2, "northwest": 1, "southeast": 3, "southwest": 4}
    region = region_map[region_input]

    return age, sex, bmi, children, smoker, region

def main():
    """Main function to run the Streamlit app."""
    st.title("US Insurance Charges Prediction")
    st.write("Predict insurance charges based on user inputs.")

    # Centered layout using columns
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column width ratios to center content

    with col2:
        age, sex, bmi, children, smoker, region = get_user_inputs()

        # Path to the dataset
        csv_path = "insurance.csv"  # Update this with the actual path to your dataset

        # Predict button
        if st.button("Predict"):
            # Call the model function with user inputs
            predicted_charge = train_insurance_model(
                age=age,
                sex=sex,
                bmi=bmi,
                children=children,
                smoker=smoker,
                region=region,
                csv_path=csv_path
            )
            st.write(f"Predicted Insurance Charge: ${predicted_charge:.2f}")


if __name__ == "__main__":
    main()
