# Script streamlit pour l'application du projet 2 

# Import des librairies 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image


def definir_style():
    style_monapp = """
    .row_heading.level0 {display:none}
"""
    st.markdown(f"<style>{style_monapp}</style>", unsafe_allow_html=True)


@st.cache()
def import_data_ml():
    st.write("import des fichiers")
    # file for machine learning
    df_one = pd.read_csv("wildflix_project_one.csv")
    df_two = pd.read_csv("wildflix_project_two.csv")

    df_final = pd.concat([df_one,df_two])
    df_final.set_index("title", inplace=True)

    return(df_final)

def import_data_info():
    st.write("import du fichier")
    # file with infos
    df = pd.read_csv("wildflix__BDD_cine (2).csv")
    return(df)    
    
def choix_page():
    pages = {
        'Accueil': accueil,
        'Etude du cinéma': stat_cinema,
        'Recommandation': recommand_film
        }
    
    with st.sidebar:
        page = st.selectbox("Choisir une page :", list(pages.keys()))
    
    pages[page]()
    

    
    
def accueil():
    st.write("Choisir une page sur la gauche")
    
    
    
def stat_cinema():
    
    df_final, df = import_data_info()
    
    ### selectionner villes
   #select_villes = st.multiselect("Choisissez une/des ville(s) :", data_crime["city"].unique())

def to_X():
    df_final = import_data_ml()
    X = df_final.select_dtypes('number')
    return X

def select_title():
    df_final = import_data_ml()
    X = to_X()
    # Selection du film
    titre = st.text_input('Veuillez saisir un titre de film :', '')
    st.write('Vous avez choisi :', titre)
    return titre  

def recommand_film():
    X = to_X()
    titre = select_title()
    df = import_data_info()
     # Recuperation des data pour le film selectionne 
    film = X.loc[titre]
 
    if type(film) == pd.core.series.Series : # ATTENTION !!! C - moche  - au cas ou marche aussi : len(film) == 28
        film = film.to_frame().T

    # Initialisation modele scaler & entrainement 
    scaler = StandardScaler().fit(X)
    # Transformation des data avec le model scaler
    X_scaled = scaler.transform(X)
    # Definition du Modele KNN avec 10 voisins et entrainement sur les data transformees
    distanceKNN_s = NearestNeighbors(n_neighbors=11).fit(X_scaled)

    # Transfo des data du film selectionne
    film_scaled = scaler.transform(film)

    # Recherche des plus proches voisins du film selectionne
    neighbors = distanceKNN_s.kneighbors(film_scaled)
    an_array = neighbors[1] # select juste les numeros d'index des neighbors

    # Affichage des resultats - boucle car possibilité d'avoir plusieurs films correspondant au même titre donc reco pour tous
    for i in range(0,len(an_array)):

        # pour les recommandations :
        reco = df.iloc[an_array[i][1:10]]
        reco.set_index("title", inplace=True)
        reco = reco[["startYear","genres","directors_name","averageRating"]] # select des colonnes interessantes 
        reco = reco.rename(columns = {"startYear":"Année","genres":"Genres","directors_name":"Réalisateur·rice","averageRating": "Note"}) # rename des colonnes
        reco.index.names = ['Titre'] # rename de l'index
        # pour le film selectionne :
        film_find = df.iloc[an_array[i][0]]
        film_find = film_find[["title","startYear","genres", "directors_name"]]# select des colonnes interessantes
        film_find = film_find.to_frame().T # transfo en df
        film_find.set_index("title", inplace=True)
        film_find = film_find.rename(columns = {"startYear":"Année","genres":"Genres","directors_name":"Réalisateur·rice"}) # rename des colonnes
        film_find.index.names = ['Titre'] # rename de l'index

        st.write(f"\nFilm trouvé : \n")
        st.write(film_find)
        st.write("\nRecommandations :\n")
        st.write(reco)



if __name__ == "__main__":
    definir_style()
    choix_page()