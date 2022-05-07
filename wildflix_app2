### Script streamlit pour l'application du projet 2 

# Import des librairies 
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image

# add page icon
st.set_page_config(layout="centered", page_icon = ":clapper:")

image = Image.open('logo_wildflix.png')
st.image(image, caption='Logo Wildflix')

# Ajout titre
st.title('Wildflix project !')

# Import des fichiers
# file for machine learning
df_one = pd.read_csv("wildflix_project_one.csv")
df_two = pd.read_csv("wildflix_project_two.csv")

df_final = pd.concat([df_one,df_two])
        
# file with infos
df = pd.read_csv("wildflix__BDD_cine (2).csv")

# Variables numeriques a prendre en compte 
X = df_final.select_dtypes('number')

# Selection du film
titre = st.text_input('Veuillez saisir un titre de film :', '')
st.write('Le titre sélectionné est :', titre)

# Fonction :
# Mettre en fonction l'algorithme de recommandation

def RecommandFilm() : 


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
        reco = reco[["startYear","genres","directors_name","actors_name","averageRating"]] # select des colonnes interessantes 
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

# Appel de la fonction
RecommandFilm()