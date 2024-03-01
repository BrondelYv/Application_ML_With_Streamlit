import streamlit as st

st.set_page_config(
    page_title="Projet ML",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

tabs_1, tabs_2, tabs_3, tabs_4 = st.tabs(["Traitement des données", "Visualisations", "Modelisation", "Evaluation"])

with tabs_1:
    st.title("Traitement des données")

    # Chargement des données
    @st.cache
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    df = load_data("../dataset/vin.csv")

    # Affichage du DataFrame
    st.subheader("Aperçu des données")
    st.write(df)

    # Analyse descriptive
    st.subheader("Analyse descriptive")
    st.write(df.describe())

    # Autres traitements de données...

with tabs_2:
    st.title("Visualisations")

    # Chargement des données
    @st.cache
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    df = load_data("../dataset/vin.csv")

    # Graphiques de distribution
    st.subheader("Graphiques de distribution")
    for column in df.columns:
        st.write(f"### Distribution de {column}")
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        st.pyplot()

    # Autres visualisations...


with tabs_3:
    st.title("Modélisation")

    # Chargement des données
    @st.cache
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    df = load_data("../dataset/vin.csv")

    # Construction du modèle
    # À compléter avec votre code de modélisation

with tabs_4:
    st.title("Évaluation")

    # Chargement des données
    @st.cache
    def load_data(file_path):
        df = pd.read_csv(file_path)
        return df

    df = load_data("../dataset/vin.csv")

    # Évaluation du modèle
    # À compléter avec votre code d'évaluation
