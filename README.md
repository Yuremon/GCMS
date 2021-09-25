# GCMS

## Projet
Le but de ce projet est de pouvoir exploiter des données de GCMS afin de faire de la détection de maladies.

## Données
Les données disponibles sont 1367 analyses de GCMS. <br>
<br>
Les données brutes se trouve dans le dossier [all-data](data/all-data), la base de donnée correspondante étant [database.csv](data/all-data/database.csv). 
Les fichiers de chaque analyse sont associés à un numéro. Les données sont extraites dans deux fichiers CSV, un premier contenant les valeur du chromatogramme au cours du temps, nommé [numéro]-chromatogram.csv. Le second contenant les noms de molécules détectés, nommé [numéro]-ms.csv.<br>
<br>
Pour faciliter la mise en place du machine learning, les dossiers [test](data/test) et [train](data/train) contiennent les données prétraités séparés pour les phases d'entraiment. Les fichiers [train/database.csv](data/train/database.csv) et [test/database.csv](data/test/database.csv) permettent de lier le numéro de l'analyse au label correspondant.

## Installation

Pour installer les dépendances de ce projet, il faut executer la commmande `pip install requirement.txt` ou `conda install requirement.txt` pour un environnment anaconda. <br>
Pour la génération de graphique et leur insertion dans les pdf préalablement il faut utiliser la commande suivante `conda install -c plotly plotly-orca`. Cela est utile pour lancer le programme main mais ce n'est pas obligatoire pour utiliser les autre fichiers python ou les notebooks.


## Fonctionnement
Le fichier [readData.ipynb](readData.ipynb) décrit le fonctionnement de la lecture des données de chromatographie et présente les diférentes fonction utiles. <br>
Ces fonctions sont écrites dans [tools.py](tools.py) et seront utilisés dans tout le projet. Ce fichier contient également un certain nombre de constantes utilisés pour la lecture des données.

Pour voir des comparaisons avant/après traitement il y a le fichier [compareFiles.ipynb](compareFiles.ipynb).<br>
Remarque : les deux courbes sont mise sur le même intervalle de temps pour faciliter la comparaison mais les données brutes sont mesurées sur un intervalle plus grand, environ [6, 60] minutes.