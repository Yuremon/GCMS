# GCMS

## Projet
Le but de ce projet est de pouvoir exploiter des données de GCMS afin de faire de la détection de maladies.

## Installation


## Fonctionnement
Le fichier [readData.ipynb](readData.ipynb) décrit le fonctionnement de la lecture des données de chromatographie et présente les diférentes fonction utiles. <br>
Ces fonctions sont écrites dans [tools.py](tools.py) et seront utilisés dans tout le projet. Ce fichier contient également un certain nombre de constantes utilisés pour la lecture des données.

Pour voir des comparaisons avant/après traitement il y a le fichier [compareFiles.ipynb](compareFiles.ipynb).<br>
Remarque : les deux courbes sont mise sur le même intervalle de temps pour faciliter la comparaison mais les données brutes sont mesurées sur un intervalle plus grand, environ [6, 60] minutes.