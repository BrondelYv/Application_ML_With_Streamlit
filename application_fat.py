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

# In[102]:


# get_ipython().system('pip install streamlit')


# In[104]:


import streamlit as st

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

# Sélection du modèle
model_selection = st.sidebar.selectbox("Sélectionnez le modèle",
                                       ["Random Forest", "Decision Tree", "Logistic Regression"])

# Entraînement du modèle
if st.sidebar.button("Entraîner le modèle"):
    # Assurez-vous que df, selected_columns et target_column sont définis auparavant
    selected_columns = st.sidebar.multiselect("Sélectionnez les colonnes", df.columns)
    target_column = st.sidebar.selectbox("Sélectionnez la colonne cible", df.columns)

    # Assurez-vous que les colonnes sélectionnées et la colonne cible sont définies
    if selected_columns and target_column:
        # Assurez-vous que df, selected_columns et target_column sont définis auparavant
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)

        # Reste du code pour entraîner le modèle
        if model_selection == "Random Forest":
            model = RandomForestClassifier()
        elif model_selection == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_selection == "Logistic Regression":
            model = LogisticRegression()
        else:
            st.warning("Modèle non pris en charge : {}".format(model_selection))
            model = None

        if model is not None:
            model.fit(X_train, y_train)
            st.success("Le modèle a été entraîné avec succès!")

# Entraîner un modèle de régression linéaire
if st.sidebar.button("Entraîner une régression linéaire"):
    if selected_columns and target_column:
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        st.success("Le modèle de régression linéaire a été entraîné avec succès!")


def get_user_input(selected_columns):
    user_input = {}
    for column in selected_columns:
        value = st.text_input(f"Entrez la valeur pour {column}:")
        user_input[column] = value
    return pd.DataFrame([user_input])


# Prédire avec le modèle de régression linéaire
if st.sidebar.checkbox("Prédictions avec la régression linéaire"):
    st.write("Entrez les nouvelles données à prédire :")
    new_data = get_user_input(selected_columns)
    if model is not None:
        prediction = model.predict(new_data)
        st.write("Prédiction (Régression Linéaire) :", prediction)

# Prédictions sur de nouvelles données
if st.sidebar.checkbox("Prédictions sur de nouvelles données"):
    st.write("Entrez les nouvelles données à prédire :")

    # Ajoutez la fonction get_user_input pour obtenir les nouvelles données
    new_data = get_user_input(selected_columns)

    if model is not None:
        prediction = model.predict(new_data)
        st.write("Prédiction :", prediction)

# In[98]:

# -------------------------------------------------------------------------------------------------------
# Étape 5: Bloc d'évaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Évaluation du modèle
if st.sidebar.button("Évaluer le modèle"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Précision du modèle :", accuracy)

    # Matrice de confusion
    st.write("Matrice de confusion :")
    confusion_matrix_display = plot_confusion_matrix(model, X_test, y_test, display_labels=df[target_column].unique(),
                                                     cmap=plt.cm.Blues, normalize='true')
    st.pyplot(confusion_matrix_display.figure_)

# In[99]:

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
