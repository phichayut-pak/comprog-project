# -*- coding: utf-8 -*-
"""Insurance.ipynb

Original file is located at
    https://colab.research.google.com/drive/1oVvU6d6qfxh1WVkw5OHZd5aOOyUeGtXJ
"""

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


#Train an insurance model based on the provided parameters and dataset.

#Parameters:
#age (int): Age of the individual.
#sex (str): Sex of the individual ('male' or 'female').
#bmi (float): Body Mass Index of the individual.
#children (int): Number of children the individual has.
#smoker (str): Smoking status of the individual ('yes' or 'no').
#region (str): Region where the individual lives ('northwest', 'northeast', 'southeast', 'southwest').
#csv_path (str): Path to the CSV file containing the insurance dataset.

#Returns:
#None

def train_insurance_model(age, sex, bmi, children, smoker, region, csv_path):
    # Load and preprocess data
    raw_data = pd.read_csv(csv_path)
    raw_data.replace(to_replace=dict(female=0, male=1), inplace=True)
    raw_data.replace(to_replace=dict(no=0, yes=1), inplace=True)
    raw_data.replace(to_replace=dict(northwest=1, northeast=2, southeast=3, southwest=4), inplace=True)
    raw_data.dropna(inplace=True)




#Load and preprocess data from a CSV file.

#This function reads data from the specified CSV file, encodes categorical variables into numerical values,
#and removes any rows with missing values.

#Parameters:
#csv_path (str): The file path to the CSV file containing the data.

#Returns:
#pandas.DataFrame: A DataFrame containing the preprocessed data.

def load_and_preprocess_data(csv_path):
    raw_data = pd.read_csv(csv_path)
    replacements = {
        'sex': {'female': 0, 'male': 1},
        'smoker': {'no': 0, 'yes': 1},
        'region': {'northwest': 1, 'northeast': 2, 'southeast': 3, 'southwest': 4}
    }
    for column, mapping in replacements.items():
        raw_data.replace(to_replace={column: mapping}, inplace=True)
    raw_data.dropna(inplace=True)
    return raw_data


#Builds and trains a neural network model.

#Parameters:
#X_train (numpy.ndarray): Training data features.
#y_train (numpy.ndarray): Training data labels.
#X_val (numpy.ndarray): Validation data features.
#y_val (numpy.ndarray): Validation data labels.

#Returns:
#keras.Sequential: Trained neural network model.

def build_and_train_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(6, input_dim=6, activation='relu'),
        Dense(4, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
    return model


#Trains an insurance model using the provided dataset and predicts the insurance charge for given input values.
#Parameters:
#age (int): Age of the individual.
#sex (int): Sex of the individual (0 for female, 1 for male).
#bmi (float): Body Mass Index of the individual.
#children (int): Number of children/dependents covered by insurance.
#smoker (int): Smoking status of the individual (0 for non-smoker, 1 for smoker).
#region (int): Region where the individual resides (encoded as an integer).
#csv_path (str): Path to the CSV file containing the insurance dataset.
#Returns:
#float: Predicted insurance charge for the given input values.

def train_insurance_model(age, sex, bmi, children, smoker, region, csv_path):
    # Load and preprocess data
    raw_data = load_and_preprocess_data(csv_path)

    # Define input and output
    input_data = raw_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].to_numpy()
    output_data = raw_data['charges'].to_numpy()

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build and train the model
    model = build_and_train_model(X_train_scaled, y_train, X_val_scaled, y_val)

    # Make a prediction for the given input values
    input_values = np.array([[age, sex, bmi, children, smoker, region]])
    input_values_scaled = scaler.transform(input_values)
    predicted_charge = model.predict(input_values_scaled)

    return float(predicted_charge[0][0])
