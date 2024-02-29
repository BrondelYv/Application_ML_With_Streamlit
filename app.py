import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from traitement.traitement import *

# Set page configuration
st.set_page_config(
    page_title="Projet ML",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared variable to store processed data
processed_data = None

# Title
st.title('App de Traitement de Donnée et Machine Learning')

# Define tabs
tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

with tabs_1:
    # Upload CSV data
    uploaded_file = st.file_uploader("Uploader un fichier csv", type="csv")

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader('Aperçu des données')

        # Checkboxes to select columns for analysis
        columns_to_keep = st.multiselect('Sélectionner les colonnes à inclure dans l\'analyse:', df.columns.tolist(), default=df.columns.tolist())

        # Filter the data based on selected columns
        df = df[columns_to_keep]

        # Display the subset of data
        st.write(df)

        # Calculate number of rows and columns
        num_rows, num_cols = calculate_shape(df)
        st.write(f'Nombres de lignes: {num_rows}')
        st.write(f'Nombres de colonnes: {num_cols}')

        # Basic statistics
        st.subheader('Tableau Descriptif')
        st.write(df.describe())

        # Identify column types
        numerical_columns, string_columns = identify_column_types(df)

        # Display column types
        st.subheader('Types de colonnes')
        st.write('Colonnes numériques:', numerical_columns)
        st.write('Colonnes texte:', string_columns)

        # Check for similar columns as the index
        similar_columns = find_similar_columns(df)

        # Display similar columns and allow the user to choose to drop them
        if similar_columns:
            st.subheader("Constat: Vous avez des colonnes avec des valeurs similaire à l'index:")
            for column in similar_columns:
                drop_column = st.checkbox(f'Supprimer {column}')
                if drop_column:
                    df.drop(column, axis=1, inplace=True)
                    st.write(f'{column} dropped successfully!')

        # Check for missing values
        if df.isnull().sum().any():
            st.subheader("Constat: Vous avez des colonnes avec des valeurs manquantes")
            df = handle_missing_values(df)

            # Display the updated data
            st.subheader('Données mise à jour')
            st.write(df)
        else:
            st.subheader("Constat: Pas de valeurs manquantes détecter.")

        # Store processed data in the shared variable
        data = df
        processed_data = data

        # Display the updated data
        st.subheader('Données mise à jour')
        st.write(data)

with tabs_2:
    pass

with tabs_3:
    if processed_data is not None:
        # Machine learning
        st.subheader('Entrainement Modèle de Machine Learning')

        # Select target variable
        target_variable = st.selectbox('Choisissez la variable à prédire (colonne à predire)', processed_data.columns)

        # Select features
        features = [col for col in processed_data.columns if col != target_variable]

        # Display selected features
        st.write(f"Features (columns used for prediction): {', '.join(features)}")

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(processed_data[features], processed_data[target_variable], test_size=0.2,
                                                            random_state=42)

        # Model training
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write('Accuracy:', accuracy)
    else:
        st.write("Veuillez d'abord charger et traiter les données dans l'onglet 'Traitement des données'.")

with tabs_4:
    pass
