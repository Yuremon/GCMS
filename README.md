# GCMS

## Projet
Le but de ce projet est de pouvoir exploiter des données de GCMS afin de faire de la détection de maladies.

## Données
Les données disponibles sont 1367 analyses de GCMS provenant d'analyses faites entre janvier 2019 et juillet 2021. <br>
Remarque : toutes les données sont anonymisées, chaque nom est remplacé par un numéro unique. <br>
<br>
Les données brutes se trouve dans le dossier [all-data](data/all-data), la base de donnée correspondante [database.csv](data/all-data/database.csv) permet de faire le lien entre le fichier et le diagnostic qui à été donné. 
Les données sont stockées dans deux fichiers CSV.
- Un premier contenant les valeur du chromatogramme au cours du temps, nommé `[numéro]-chromatogram.csv`.
- Un second contenant les noms de molécules détectés, nommé `[numéro]-ms.csv`.

Pour faciliter la mise en place du machine learning, les dossiers [test](data/test) et [train](data/train) contiennent les données prétraités séparés pour les phases d'entraiment et de test. Les fichiers [train/database.csv](data/train/database.csv) et [test/database.csv](data/test/database.csv) permettent de lier le numéro de l'analyse au label correspondant.

## Installation

Pour installer les dépendances de ce projet, il faut executer la commmande `pip install requirement.txt` ou `conda install requirement.txt` pour un environnment anaconda. <br>
Pour la génération de graphiques et leur insertion dans les pdf préalablement il faut utiliser la commande suivante `conda install -c plotly plotly-orca`. Cela est utile pour lancer le fichier main mais ce n'est pas obligatoire pour utiliser les autre fichiers python ou les notebooks.

Au premier lancement du programme `main`, il sera demander d'indiquer plusieurs chemins. Pour utiliser la base de donnée présente sur ici les chemins seront les suivants :
- Base de donnée : [data/all_data_transformed/database.csv](data/all_data_transformed/database.csv)
- Dossier contenant les chromatogrammes : [data/all_data_transformed](data/all_data_transformed)
- Dossier de fichier temporaires : un nouveau dossier vide (par exemple appelé temp).

## Fonctionnement

La classification des données de GCMS en deux classes (normaux/ non normaux) se fait en plusieurs étapes, détaillées dans le [rapport](report/rapport_de_stage_levarlet).
- Traitement des données : réduire un chromatogramme de 9000 points en un vecteur de taille plus raisonnable pour l'apprentissage, sans perdre d'information utile.
- Réduction de dimension des données après traitement : divers algorithmes de réduction de dimension et divers tailles de réduction sont testés dans le but d'améliorer l'apprentissage.
- Apprentissage : divers algorithmes d'apprentissages sont testés. Le but est d'obtenir dans un premier temps la meileur précision pour la combinaison réduction-apprentissage. Puis l'objectif est d'éviter les profils non-normaux prédit normaux.

## Organisation du code

Fichiers présent dans le dossier src ont diférents but:

- Le programme principale : 
    - Le fichier [main.py](src/main.py) correspond au programme principal, dont le fonctionnement est décrit à la fin du [rapport](report/rapport_de_stage_levarlet.pdf) de stage.
    - Le fichier [tools.py](src/tools.py) regroupe les fonctions et classes utilisés dans tout le projet. Ce fichier contient également un certain nombre de constantes utilisés pour la lecture des données.

- La présentation du fonctionnement des différents composant du programme 
    - Le fichier [readData.ipynb](src/readData.ipynb) décrit le fonctionnement de la lecture des données de chromatographie et présente les diférentes fonction utiles. 
    - Le fichier [spikeDectection.ipynb](src\spikeDectection.ipynb) présente la strategie de détection des pics.
    - Le fichier [compareFiles.ipynb](src/compareFiles.ipynb) permet de visualiser le traitement des données avec des graphiques avant/après taritement.<br>
    Remarque : les deux courbes sont mise sur le même intervalle de temps pour faciliter la comparaison mais les données brutes sont mesurées sur un intervalle plus grand, environ [6, 60] minutes.
    - Le fichier [database.ipynb](src\database.ipynb) à été utilisé pour construire la base de donnée, à partir des fichiers disponibles au laboratoire. Il est acompagné de [readExcelFile.ipynb](src\readExcelFile.ipynb) qui présente la lecture du fichier obtenu sur la base de donnée du laboratoire.<br> Il a été necessaire de faire une vérification manuelle de toutes les données après - Le fichier [checkFiles.ipynb](src/checkFiles.ipynb) permet de vérifier que tous les fichier `[nom]-chromatogram.csv` on un correspondant `[nom]-ms.csv`.

- L'entrainement et la comparaison des algorithmes de machine learning
    - Le fichier [Nearest_neighbors.ipynb](src\Nearest_neighbors.ipynb) et [Random_forest.ipynb](src\Random_forest.ipynb) présentent chacun un exemple d'application d'algorithme de machine learning sur les données, avec recherche d'hyperparamètres.
    - La recherche du meilleur algorithme (réduction de dimention et classification) est cherchée dans [machineLearningSelection.ipynb](src\machineLearningSelection.ipynb).