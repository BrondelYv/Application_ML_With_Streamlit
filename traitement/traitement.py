# traitement/traitement.py
import pandas as pd
import streamlit as st

df = pd.read_csv("./dataset/vin.csv")

# Function to calculate the number of rows and columns
def calculate_shape(data):
    num_rows, num_cols = data.shape
    return num_rows, num_cols


def find_similar_columns(data):
    similar_columns = []
    index_values = data.index.tolist()
    for column in data.columns:
        if data[column].tolist() == index_values:
            similar_columns.append(column)
    return similar_columns


def identify_column_types(data):
    numerical_columns = []
    string_columns = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            numerical_columns.append(column)
        elif pd.api.types.is_string_dtype(data[column]):
            string_columns.append(column)
        # You can add more conditions for different data types if needed

    return numerical_columns, string_columns


# Function to handle missing values
def handle_missing_values(data):
    if data.isnull().sum().any():
        missing_columns = data.columns[data.isnull().any()].tolist()
        st.write('Colonnes avec des valeurs manquantes:', missing_columns)

        # Choose handling method
        handling_method = st.radio("Choisir une méthode de gestion des valeurs manquantes:",
                                   ("Supprimer les lignes", "Imputer les valeurs"))

        if handling_method == "Supprimer les lignes":
            data.dropna(inplace=True)
            st.write("Les lignes contenant des valeurs manquantes ont été supprimées.")
        elif handling_method == "Imputer les valeurs":
            # You can implement different imputation strategies here
            # For example, impute numerical columns with mean, and categorical columns with mode
            data.fillna(data.mean(), inplace=True)  # Example: fill NaNs with mean values

    return data
