#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
# Importez les bibliothèques nécessaires au début de votre script
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import expon
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import *

import streamlit as st

from evaluation.evaluator import evaluate_model
from traitement.distributions import visualize_normal_distribution, visualize_exponential_distribution
from traitement.nettoyage import *

st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")

# Étape 1: Construction de Streamlit
st.sidebar.title("Paramètres")

st.title("Application Machine Learning")
# Étape 2: Chargement du jeu de données
df = load_data()

# ------------------------------------------------------------------------------
# Étape 3: Traitement des données
selected_columns, target_column = show_data_processing_options(df)

# Analyse descriptive
descriptive_analysis(df)

# Graphique de distribution et pairplot
distribution_pairplot(df, selected_columns, target_column)

# Corrélation avec la cible
correlation_with_target(df, target_column)

# Fréquences
column_frequencies(df, target_column)

# Standardisation
standardization(df)

# Visualisation de la distribution normale
visualize_normal_distribution(df, selected_columns)

# Visualisation de la distribution exponentielle
visualize_exponential_distribution(df, selected_columns)

# --------------------------------------------------------------------------
# Étape 4: Bloc de machine learning pipeline
st.sidebar.subheader("Bloc de Machine Learning")
# Initialisation du modèle à None
model = None

# Entraînement du modèle
train_model = st.sidebar.checkbox("Entraîner le modèle")

if train_model:
    selected_model = st.sidebar.selectbox("Sélectionnez le modèle", ["Linear Regression", "Logistic Regression", "Decision Tree", "SVM", "Naive Bayes", "Random Forest", "Dimensionality Reduction Algorithms"])

    if selected_columns and target_column:
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2, random_state=42)

    # Sélection du modèle
    model = None
    if selected_model == "Linear Regression":
        model = LinearRegression()
    elif selected_model == "Logistic Regression":
        model = LogisticRegression()
    elif selected_model == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_model == "SVM":
        model = SVC()
    elif selected_model == "Naive Bayes":
        model = GaussianNB()
    elif selected_model == "Random Forest":
        model = RandomForestClassifier()
    elif selected_model == "Dimensionality Reduction Algorithms":
        model = PCA()  # Vous pouvez ajuster le nombre de composants principaux
    else:
        st.warning("Modèle non pris en charge : {}".format(selected_model))

    if model is not None:
        model.fit(X_train, y_train)
        st.success("Le modèle a été entraîné avec succès!")

# Prédictions sur de nouvelles données
if st.sidebar.checkbox("Prédictions sur de nouvelles données"):
    st.write("Entrez les nouvelles données à prédire :")

    # Ajoutez la fonction get_user_input pour obtenir les nouvelles données
    new_data = get_user_input(selected_columns)

    if model is not None:
        try:
            prediction = model.predict(new_data)
            st.write("Prédiction :", prediction)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")

# -------------------------------------------------------------------------------------------------------
# Étape 5: Bloc d'évaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Évaluation du modèle
if st.sidebar.button("Évaluer le modèle"):
    evaluate_model(model, X_test, y_test, target_column, df)

# ------------------------------------------------------------------------------------------------------------------------------------
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

# In[ ]:
