#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[102]:


# get_ipython().system('pip install streamlit')


# In[104]:


import streamlit as st

st.title("Mon Application Streamlit")
st.write("Bonjour depuis Streamlit!")


# In[107]:





# In[60]:


# Charger les jeux de données
wine_data = pd.read_csv(r"C:\Users\berra\PycharmProject\Projet-ML\dataset\vin.csv", delimiter=',')
diabetes_data = pd.read_csv(r'C:\Users\berra\PycharmProject\Projet-ML\dataset\diabete.csv', sep=',')


# In[47]

wine_data.info()


# In[66]:


wine_data.head()


# In[50]:


wine_data.shape


# In[52]:


wine_data.describe()


# In[61]:


diabetes_data.info()


# In[62]:


diabetes_data.head()


# In[63]:


diabetes_data.shape


# In[64]:


diabetes_data.describe()


# In[56]:


# get_ipython().run_line_magic('run', '-i C:\\\\Users\\\\berra\\\\PycharmProject\\\\Projet_MachineLearning\\\\App.ipynb')


# In[68]:


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


# In[97]:


# Étape 4: Bloc de machine learning pipeline
st.sidebar.subheader("Bloc de Machine Learning")
# Initialisation du modèle à None
model = None

# Sélection du modèle
model_selection = st.sidebar.selectbox("Sélectionnez le modèle",
                                       ["Random Forest", "Decision Tree", "Logistic Regression"])

# Entraînement du modèle
if st.sidebar.button("Entraîner le modèle"):
    # Ajoutez la fonction get_model pour créer le modèle sélectionné
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
        # Assurez-vous que df, selected_columns et target_column sont définis auparavant
        X_train, X_test, y_train, y_test = train_test_split(df[selected_columns], df[target_column], test_size=0.2,
                                                            random_state=42)
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


# In[98]:


# Étape 5: Bloc d'évaluation
st.sidebar.subheader("Bloc d'Évaluation")

# Évaluation du modèle
if st.sidebar.button("Évaluer le modèle"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Précision du modèle :", accuracy)

    # Matrice de confusion
    st.write("Matrice de confusion :")
    confusion_matrix_display = plot_confusion_matrix(model, X_test, y_test, display_labels=df[target_column].unique(), cmap=plt.cm.Blues, normalize='true')
    st.pyplot(confusion_matrix_display.figure_)


# In[99]:


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




