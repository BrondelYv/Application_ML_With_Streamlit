#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Importez les bibliothèques nécessaires au début de votre script
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# get_ipython().system('pip install streamlit')


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from traitement.traitement import *

<<<<<<< HEAD
st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")


# Charger les jeux de données
wine_data = pd.read_csv("./dataset/vin.csv", delimiter=',')
diabetes_data = pd.read_csv('./dataset/diabete.csv', sep=',')
=======
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
>>>>>>> 2cb6d9cc2b2251e11874f7f6306c80ef18a80713


<<<<<<< HEAD
wine_data.info()


wine_data.head()


wine_data.shape


wine_data.describe()


diabetes_data.info()


diabetes_data.head()


diabetes_data.shape


diabetes_data.describe()


# Étape 1: Construction de Streamlit
st.title("Application Machine Learning - Vin")
st.sidebar.title("Paramètres")

# Initialiser df
df = None

# Étape 2: Chargement du jeu de données
uploaded_file = st.sidebar.file_uploader("Charger le fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu du jeu de données :")
    st.write(df.head())
else:
    st.warning("Veuillez charger un fichier CSV pour continuer.")

# Étape 3: Effectuer un bloc de traitement de données
st.sidebar.subheader("Étape 3: Traitement des données")

# Vérifier si df est défini
if df is not None:
    selected_columns = st.sidebar.multiselect("Sélectionnez les colonnes", df.columns)
    target_column = st.sidebar.selectbox("Sélectionnez la colonne cible", df.columns)

# Analyse descriptive
if st.sidebar.checkbox("Analyse descriptive"):
    st.write("Statistiques descriptives du jeu de données :")
    st.write(df.describe())

# Graphique de distribution et pairplot
if st.sidebar.checkbox("Graphique de distribution et pairplot"):
    st.write("Graphique de distribution :")
    st.pyplot(sns.pairplot(df).fig)

# Corrélation avec la cible
if st.sidebar.checkbox("Corrélation avec la cible"):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()
    st.write("Matrice de corrélation :")
    st.write(correlation_matrix)

# Fréquences
if st.sidebar.checkbox("Fréquences"):
    st.write("Fréquence des valeurs dans la colonne cible :")
    st.write(df[target_column].value_counts())

# Standardisation
if st.sidebar.checkbox("Standardisation"):
    st.write("Standardisation des colonnes numériques :")
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
    st.write(df.head())


# Étape 4: Bloc de machine learning pipeline
st.sidebar.subheader("Bloc de Machine Learning")
# Initialisation du modèle à None
model = None

# Sélection du modèle
model_selection = st.sidebar.selectbox("Sélectionnez le modèle",
                                       ["Random Forest", "Decision Tree", "Logistic Regression"])

# Entraînement du modèle
if st.sidebar.button("Entraîner le modèle"):
    if model_selection == "Random Forest":
        model = DecisionTreeClassifier()
    elif model_selection == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_selection == "Logistic Regression":
        model = LogisticRegression()
    else:
        st.warning("Modèle non pris en charge : {}".format(model_selection))
        model = None

    if model is not None:
        # Assurez-vous que df, selected_columns et target_column sont définis auparavant
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)

        # Vérifier les dimensions et les types de données
        st.write("Dimensions de X_train :", X_train.shape)
        st.write("Dimensions de y_train :", y_train.shape)
        st.write("Types de données de X_train :", X_train.dtypes)
        st.write("Types de données de y_train :", y_train.dtypes)

        # Vérifier les valeurs manquantes
        st.write("Valeurs manquantes dans X_train :", X_train.isnull().sum())
        st.write("Valeurs manquantes dans y_train :", y_train.isnull().sum())

        model.fit(X_train, y_train)
        st.success("Le modèle a été entraîné avec succès!")


def get_user_input(selected_columns):
    user_input = {}
    for column in selected_columns:
        value = st.text_input(f"Entrez la valeur pour {column}:")
        user_input[column] = value
    return pd.DataFrame([user_input])

# Prédictions sur de nouvelles données
if st.sidebar.checkbox("Prédictions sur de nouvelles données"):
    st.write("Entrez les nouvelles données à prédire :")

    # Ajoutez la fonction get_user_input pour obtenir les nouvelles données
    new_data = get_user_input(selected_columns)

    if model is not None:
        prediction = model.predict(new_data)
        st.write("Prédiction :", prediction)


# Étape 5: Bloc d'évaluation
st.sidebar.subheader("Bloc d'Évaluation")
st.write(model)
# Évaluation du modèle
if st.sidebar.button("Évaluer le modèle"):
    if model is not None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Précision du modèle :", accuracy)

        # Rapport de classification
        st.write("Rapport de classification :")
        classification_rep = classification_report(y_test, y_pred)
        st.write(classification_rep)

        # Matrice de confusion
        st.write("Matrice de confusion :")
        confusion_matrix_display = plot_confusion_matrix(model, X_test, y_test, display_labels=df[target_column].unique(), cmap=plt.cm.Blues, normalize='true')
        st.pyplot(confusion_matrix_display.figure_)
    else:
        st.warning("Entraînez d'abord le modèle avant de l'évaluer.")



# Étape 6: Fonctionnalités supplémentaires
st.sidebar.subheader("Fonctionnalités Supplémentaires")

# Lazy Predict
if st.sidebar.checkbox("Lazy Predict"):
    st.write("Lazy Predict :")
    lazy_predict(X_train, X_test, y_train, y_test)

# GridSearchCV
if st.sidebar.checkbox("GridSearchCV"):
    st.write("GridSearchCV :")
    grid_search_cv(X_train, y_train)

# Modèle de Deep Learning (Exemple avec Keras)
if st.sidebar.checkbox("Modèle de Deep Learning (Keras)"):
    st.write("Modèle de Deep Learning avec Keras :")
    keras_model(X_train, X_test, y_train, y_test)
=======
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
>>>>>>> 2cb6d9cc2b2251e11874f7f6306c80ef18a80713
