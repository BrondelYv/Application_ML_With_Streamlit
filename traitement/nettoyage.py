# In traitement/data_processing.py
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler



# Function to load data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Charger le fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Aperçu du jeu de données :")
        st.write(df.head())
        return df
    else:
        st.warning("Veuillez charger un fichier CSV pour continuer.")
        return None


def show_data_processing_options(df):
    st.sidebar.subheader("Étape 3: Traitement des données")

    if df is not None:
        selected_columns = st.sidebar.multiselect("Sélectionnez les colonnes", df.columns, key='a')
        target_column = st.sidebar.selectbox("Sélectionnez la colonne cible", df.columns)
        return selected_columns, target_column
    else:
        return None, None


def descriptive_analysis(df):
    if st.sidebar.checkbox("Analyse descriptive"):
        st.write("Statistiques descriptives du jeu de données :")
        st.write(df.describe())


def distribution_pairplot(df, selected_columns, target_column):
    if st.sidebar.checkbox("Graphique de distribution et pairplot"):
        st.write("Graphique de distribution :")
        if target_column:
            st.pyplot(sns.pairplot(df[selected_columns + [target_column]]).fig)
        else:
            st.write("S'il vous plait selectionnez une colonne cible pour afficher les graphiques")


def correlation_with_target(df, target_column):
    if st.sidebar.checkbox("Corrélation avec la cible"):
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        correlation_matrix = df[numeric_columns].corr()
        st.write("Matrice de corrélation :")
        st.write(correlation_matrix)


def column_frequencies(df, target_column):
    if st.sidebar.checkbox("Fréquences"):
        st.write("Fréquence des valeurs dans la colonne cible :")
        st.write(df[target_column].value_counts())


def standardization(df):
    if st.sidebar.checkbox("Standardisation"):
        st.write("Standardisation des colonnes numériques :")
        numeric_columns = df.select_dtypes(include=['float64']).columns
        df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
        st.write(df[numeric_columns].head())
