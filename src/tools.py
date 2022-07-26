import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.signal import butter, filtfilt
from os import listdir
from os.path import join
from sklearn.metrics import plot_roc_curve, classification_report
from confusion_matrix.confusion_matrix import plot_confusion_matrix_from_data
import joblib

verbose = False

# labels utilisés pour la seconde phase de l'IA
labels = [
    "normal",                          # 0
    "cétose",                          # 1
    "maladies métabolique connues",    # 2
    "métaboliques bactériens",         # 3
    "métabolique médicamenteux",       # 4
    "autre"                            # 5
]

# molécules interessantes à récupérer
MOLECULES = [
    ["Acide Lactique", "ACIDE LACTIQUE", "13C3-Lact", "Krebs-Acide Lactique*", "Krebs-Acide Lactique", "01_LACTIC"], 
    ["Acide 4-Phenylbutyrique", "4 phenylbutyrique", "Acide 4 phenylbutyrique"], 
    ["Acide O-OH-Phenylacetiqu", "x o-OHphenylacetic", "o-OH-Phenyl acetique"], 
    ["C17-Heptadecanoique", "68. ETALON INT 2"], 
    ["Acide 3OH-Propionique", "Acide 3OH-Propionique 20", "Acide 3OH-Propionique*", "AC. 3OH PROPIONIQUE"], 
    ["Acide Methylmalonique", "Acide Methylmalonique 20", "Acide Methylmalonique*", "AC. METHYLMALONIQUE"], 
    ["Acide Fumarique", "Acide Fumarique 20200722", "ACIDE FUMARIQUE", "Acide Fumarique 20170708", "Acide Fumarique*", "Acide Fumarique 20190708"], 
    ["Acide Glutarique", "Acide Glutarique 2020072", "Acide Glutarique 2019071", "Acide Glutarique*"], 
    ["Acide 3CH3-Glutarique", "Acide 3CH3-Glutarique 20", "ACIDE 3CH3GLUTARIQUE"], 
    ["3 methyl-glutaconique", "31_3-METHYLGLUT", "Acide 3CH3-Glutaconique"],
    ["Acide Adipique", "Acide Adipique 20200722", "Acide Adipique 20190712", "Acide Adipique*"], 
    ["Acide 3OH-Glutarique", "Acide 2OH-Glutarique 202", "Acide 3OH-Glutarique 201"], 
    ["Acide Homovanillique", "X HOMOVANILLIC"],
    ["Vanillyl propionique", "X Vanillilpropanoic chk"],
    ["Vanillyl Lactique", "ac benzene propanoic 3 metho"],
    ["Isovaleryl Glycine", "Isovalerylglycine"]
]

# 3 méthyl glutarique et glutaconique à partir de 15%
# Homovanilic à partir de 25%
# Adipique 25%
# Acide lactique 300%
RULE_THRESHOLD = [3, None, None, None, 0.2, 0.2, 0.25, 0.2, 0.15, 0.15, 0.25, 0.2, 0.25, 0.2, 0.2, 1]

class ReadDataException(Exception):
    """Exception levée en cas de problème de lecture de la base de donnée.
    """
    

class MachineLearningTechnique():
    """Classe permettant d'utiliser un modèle de machine learning associé avec un algorithme de réduction de dimension.
    """

    def __init__(self, model=None, reduction=None):
        """Création d'un nouvel objet `MachineLearningTechnique` à partir des modeles de machine learning entrainés auparavant.

        Args:
            model (optional): classifieur utilisé (tiré de sklearn ou xgboost) par exemple RandomForestClassifier(). Defaults to None.
            reduction (optional): reduction de dimension utilisé (). Defaults to None.
        """
        self.model = model
        self.reduction = reduction

    def score(self, X_test:np.ndarray, y_test:np.ndarray) -> float: 
        """Calcul du score obtenu sur les échantillons de test.

        Args:
            X_test (np.ndarray): échantillons d'entrée.
            y_test (np.ndarray): labels associés.

        Returns:
            float: La valeur du score obtenu sur la base de donnée de test.
        """
        self.var_score = self.model.score(X_test, y_test) # changer par la fonction qui permet de discriminer les models
        return self.var_score 

    def predict(self, X:np.ndarray)->np.ndarray:
        """Prediction de la classe pour un ou plusieurs chromatogramme.

        Args:
            X (np.ndarray): matrice de taille n * 505 avec n le nombre d'entrées, chaque ligne représentant un chromatogramme.

        Returns:
            np.ndarray: vecteur de taille n, chaque valeur du vecteur est le numéro de la classe prédite.
        """
        if self.reduction is not None:
            X = self.reduction.transform(X)
        return self.model.predict(X)        

    def save(self, path:str=''):
        """Enregistre le model dans deux fichiers : model.sav et reduction.sav.

        Args:
            path (str, optional): chemin du dossier dans lequel seront enregistrés les fichiers. Defaults to ''.
        """
        joblib.dump(self.model, path + 'model.sav')
        joblib.dump(self.reduction, path + 'reduction.sav')

    def load(self, path:str=''):
        """Charge dans l'instance courante le model enregistré avec la focntion save, à partir des fichiers model.sav et reduction.sav.

        Args:
            path (str, optional): chemin du dossier dans lequel à été enregistré le model. Defaults to ''.
        """
        self.model = joblib.load(path + 'model.sav')
        self.reduction = joblib.load(path + 'reduction.sav')

    def displayMetrics(self, X_test:np.ndarray, y_test:np.ndarray):
        """Affiche dans le terminal les métriques utilisées pour déterminer si le model est performant sur des données de test.
            - caractérisques du model (type de réduction, taille de l'entrée, type de model)
            - justesse
            - classification_report
            - matrice de confusion
            - ROC curve

        Args:
            X_test (np.ndarray): échantillons d'entrée.
            y_test (np.ndarray): labels associés.
        """
        if self.reduction is not None:
            print('Best reduction', self.reduction)
            print('Best size : ', self.reduction.n_components)
            X_test = self.reduction.transform(X_test)
                   
        predictions = self.model.predict(X_test)
        print('Model : ', self.model)
        print('Best score : ', self.score(X_test, y_test))
        sk_report = classification_report(
            digits=6,
            y_true=y_test, 
            y_pred=predictions
        )
        print(sk_report)
        plot_confusion_matrix_from_data(y_test, predictions,columns=['non normal','normal'])
        plot_roc_curve(self.model, X_test, y_test)

class Data:
    """Classe utilisée pour facilité le taitement et la lecture des données.
    La structure des données est basée sur les DataFrame de Pandas.
    """

    def __init__(self, name, df : pd.DataFrame = None, state : bool = None, molecules : list = MOLECULES):
        """Création d'une instance de la classe data, aucune lecture ni traitement n'est fait à cette étape.

        Args:
            name (str): nom du fichier sans extension (-chromatogram.csv ni -ms.csv)
            df (pd.DataFrame, optional): valeurs de l'intensité mesurés au cours du temps pour un chromatogramme. Defaults to None.
            molecules (list, optional): liste de liste contenant les molecules à détecter, chaque élément de la liste étant une liste des noms possibles, lu dans les fichier -ms.csv. Defaults to MOLECULES.
        """
        self.name = name
        self.df = df
        #self.state = state
        self.spikes = None
        self.molecules = molecules
        self.problems = []

    def readCSV(self, path):
        """Lecture du fichier [name]-chromatogram.csv, stockage des valeurs dans le DataFrame self.df.

        Args:
            path (str): chemin du dossier dans lequel se trouve les fichiers [name]-chromatogram.csv et [name]-ms.csv.
        """
        self.df = readCSV(join(path, self.name + CHROM_EXT))

    def detectSpikes(self, path):
        """Détection des pics dont le nom se trouve dans self.molecules. Leur temps de rétention est stocké dans self.spikes.

        Args:
            path (str): chemin du dossier dans lequel se trouve les fichiers [name]-chromatogram.csv et [name]-ms.csv.

        Raises:
            ReadDataException: Si les pics de référence ne sont pas trouvés (les 4 premieres molécules dans self.molecules).
            ReadDataException: Si un des temps de rétention est en dehors de l'intervalle mesuré.
        """
        self.spikes = detectSpikes(join(path, self.name + MOL_EXT), MOLECULES) # on ne passe que les molecules de référence
        if None in self.spikes[0:4]:
            raise ReadDataException("Un des pics de référence n'est pas détecté")
        for time in self.spikes:
            if time != None and (self.df.index[0] > time or self.df.index[-1] < time):
                raise ReadDataException("Une des molécules est détectée en dehors de l'intervalle de temps de la mesure")

    def alignSpikes(self):
        """Ajuste les valeurs du chromatogramme de façon linéaire pour faire correspondre les pics à leurs valeurs de référence.
        """
        self.df = alignSpikes(self.df, self.spikes)

    def ruleBasedCheck(self):
        """Vérification que les intensités mesurées pour les molécules ne dépasse leur seuil de tolérance.
        Nom, temps de rétention et valeurs sont stokées dans une liste : self.problems.

        Raises:
            ReadDataException: Erreur de valeur sur le pic de référence 1.
            ReadDataException: Erreur sur le nombre de molécules à parcourir.
        """
        df = self.df['values']-self.df['values'].min()
        reference = df.iloc[getTimeIndex(self.df.index, self.spikes[1])]
        if reference == 0:
            raise ReadDataException("Erreur de détection du pic de référence")
        if len(MOLECULES) != len(RULE_THRESHOLD):
            raise ReadDataException('Problème de nombre de molécules')
        for i in range(len(self.spikes)):
            if self.spikes[i] == None :
                if verbose:
                    print(MOLECULES[i][0], ' non détectée')
                continue
            if RULE_THRESHOLD[i] == None: # si la molécule n'est pas intéressante (par ex les pics de référence) on ne prend pas en compte
                continue
            value = df.iloc[getTimeIndex(self.df.index, self.spikes[i])]
            value = value / reference
            if value > RULE_THRESHOLD[i]:
                self.problems.append((self.spikes[i], value * 100, MOLECULES[i][0])) # abscisse, valeur en % du pic par rapport à la réference et nom de la molecule
    
    def problemsDescription(self):
        """Description des problèmes détectés lors du ruleBasedCheck.
        Les problèmes sont triés par valeur d'intensité.

        Returns:
            str: chaine de caractère décrivant les problèmes rencontrés, chaine vide si il n'y a eu aucun problèmes.
        """
        # mettre dans l'ordre d'importance
        string = ''
        if len(self.problems)<1:
            return string
        self.problems.sort(reverse=True, key=getValue) 
        for pb in self.problems:
            string += f"La molécule {pb[2]} est présente à {int(pb[1])} % du pic de référence à {pb[0]} minutes\n"
        return string
        
    def printProblems(self):
        """Affiche dans le terminal les problèmes détectés lors du ruleBasedCheck.
        """
        print(self.problemsDescription())
        
def getValue(x):
    """Fonction utilisée pour trier la liste des problèmes par valeur d'intensité mesurée, dans Data.problemsDescription
    """
    return x[1]
#######################################
# Constantes                          #
#######################################

INTERVAL = [6.01,45.01]
NOISE_FREQ = None #0.09
BIAS_FREQ = 0.002
THRESHOLD = 1.6
PADDING = 45 # attention la valeur doit être supérieur à 1
PERIOD = [1, 2, 6, 20] 
RESAMPLE_MS = [str(p) +'s' for p in PERIOD]
ENTRY_SIZE = 504

#######################################
# Lecture des données                 #
#######################################

CHROM_EXT = '-chromatogram.csv'
MOL_EXT = '-ms.csv'

def readCSV(path : str)->pd.DataFrame:
    """Lecture du fichier contenant le chromatogrammme.

    Args:
        path (str): chemin du fichier contenant le chromatogramme (terminant par -chromatogram.csv).

    Raises:
        ReadDataException: En cas de fichier inexistant.

    Returns:
        pd.DataFrame: valeurs du chromatogramme au cours du temps d'indice le temps en minutes et d'une colonne ("values") contenant les valeurs.
    """
    # lecture du fichier csv
    try:
        df = pd.read_csv(path, header=None, skiprows=[0,1,2,3,4,5], index_col=0)
    except FileNotFoundError :
        raise ReadDataException("Le fichier chromatogramme n'a pas pu être trouvé")
    #suppression de la colonne vide
    df.drop(df.columns[1], axis=1, inplace=True)
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df['values'] = df['values'].fillna(0) # on enlève les éventuels nan
    return df


def readAndAdaptDataFromCSV(path, name) -> Data:
    """Lecture d'un chromatogramme (fichiers [name]-chromatogram.csv et [name]-ms.csv) et traitement de celui-ci.

    Args:
        path (str): chemin du dossier contenant les deux fichiers décrivant le chromatogramme.
        name (str): nom des fichiers sans l'extension.

    Raises:
        ReadDataException: En cas de problème au cours de la lecture.

    Returns:
        Data: L'instance décrivant le chromatogramme après traitement.
    """
    dt = Data(name)
    dt.readCSV(path)
    dt.detectSpikes(path)
    dt.ruleBasedCheck()
    dt.df = normalise(dt.df,dt.spikes)
    df = adaptCurve(dt.df, dt.spikes)
    dt.df = substractBias(df)
    # Etalonner sur le premier pic (taille de 10 pour celui-ci) 
    # remarque : attention si la valeur n'est pas précise les pics seront plus grand que prévu
    timeSpike1 = getTimeIndex(dt.df.index, SPIKES_EXPECTED_TIME[1])
    reference = dt.df.iloc[timeSpike1]['values'] / 10
    if reference == 0:
        raise ReadDataException("Valeur au pic de référence de 0")
    dt.df = dt.df / reference
    if len(dt.df) != ENTRY_SIZE and len(dt.df) != ENTRY_SIZE+1:
        raise ReadDataException("Problème de taille sur le chromatogramme : ", len(dt.df))
    return dt

def readAllData(path : str):#->list[Data]
    """Lecture et traitement de l'ensemble des chromatogrammes presents dans un dossier.

    Args:
        path (str): Chemin du dossier contenant les chromatogrammes

    Returns:
        list[Data]: Liste des instances correspondant à chaques chromatogrammes dont le traitement est possible.
    """
    files = [f for f in listdir(path) if f.endswith(CHROM_EXT)]
    dataList = []
    for file in files:
        dt = readAndAdaptDataFromCSV(path, file[:-17]) # données traitées
        dataList.append(dt)
    return dataList

def readListOfData(files : np.ndarray, path : str):#->list[Data]
    """Lecture et traitement de l'ensemble des chromatogrammes presents dans une liste de noms.
    Remarque : les fichiers doivent se trouver dans un même dossier

    Args:
        files (np.ndarray): array contenant une liste de noms de fichier (sans extension -chromatogram.csv ni -ms.csv) à lire
        path (str): dossier contenant l'ensemble des fichiers de la liste

    Returns:
        list[Data]: Liste des instances correspondant à chaques chromatogrammes dont le traitement est possible.
    """
    dataList = []
    for file in files:
        try:
            dt = readAndAdaptDataFromCSV(path, file)
        except ReadDataException as e :
            print(e)
        finally:
            dataList.append(dt)
        
    return dataList

def getData(file_path : str, db_path : str, binary : bool = True):
    """Lecture des données d'une base de donnée de chromatogrammes brutes

    Args:
        file_path (str): chemin du fichier contenant la base de donnée (fichier csv contenant la liste des fichiers et leur diagnostique)
        db_path (str): chemin du dossier contenant tout les fichier auquel la base de donnée fait référence.
        binary (bool, optional): si vrai, la classification est binaire (normal = 1, non normal = 0).
        Sinon la classification est faite selonn les 5 classes possibles Defaults to True.

    Returns:
        np.ndarray, np.ndarray: Tuple contenant les données des chromatogrammes en position 0 et la classe correspondante en position 1.
        Les chromatogrammes sont décrit par une matrice contenant pour chaques lignes les valeurs d'un chromatogramme au cours du temps.
        La classe correspondante est donnée dans un vecteur contenant l'indice de la classe de chaque chromatogrammes
    """
    # lecture du fichier représentant la base de donnée
    db = pd.read_csv(file_path)
    # y représente la sortie (normale ou non)
    if binary:
        y = db['status'].to_numpy()
    else :
        y = db['label_num'].to_numpy()

    # lecture de tous les chromatogrammes listées
    files = db['file'].to_numpy()
    n = len(files)
    # X représente toutes les entrées (une entrée par ligne)
    X = np.zeros((n, ENTRY_SIZE))
    lostFiles = 0
    for i in range(n):
        # chaque ligne correspond aux valeurs au cours du temps du chromatogramme
        file_name = str(files[i])
        try:
            result = readAndAdaptDataFromCSV(db_path, file_name).df['values'].to_numpy()
            X[i, :] = result[:ENTRY_SIZE]
        except ReadDataException as e:
            if verbose :
                print("Erreur sur le fichier : ", file_name, "message : ", e)
            lostFiles += 1
    if verbose :
        print('Nombre de fichiers perdus : ', lostFiles)
    return X, y

def getDataTransformed(file_path : str, db_path : str, binary : bool = True):
    """Lecture des données d'une base de donnée de chromatogrammes préalablement traités

    Args:
        file_path (str): chemin du fichier contenant la base de donnée (fichier csv contenant la liste des fichiers et leur diagnostique)
        db_path (str): chemin du dossier contenant tout les fichier auquel la base de donnée fait référence.
        binary (bool, optional): si vrai, la classification est binaire (normal = 1, non normal = 0).
        Sinon la classification est faite selonn les 5 classes possibles Defaults to True.

    Returns:
        np.ndarray, np.ndarray: Tuple contenant les données des chromatogrammes en position 0 et la classe correspondante en position 1.
        Les chromatogrammes sont décrit par une matrice contenant pour chaques lignes les valeurs d'un chromatogramme au cours du temps.
        La classe correspondante est donnée dans un vecteur contenant l'indice de la classe de chaque chromatogrammes
    """
    db = pd.read_csv(file_path)
    # y représente la sortie (normale ou non)
    if binary:
        y = db['status'].to_numpy()
    else :
        y = db['label_num'].to_numpy()

    # lecture de tous les chromatogrammes listées
    files = db['file'].to_numpy()
    n = len(files)
    # X représente toutes les entrées (une entrée par ligne)
    X = np.zeros((n, ENTRY_SIZE))
    lostFiles = 0
    for i in range(n):
        # chaque ligne correspond aux valeurs au cours du temps du chromatogramme
        file_name = str(files[i])
        try:
            result = pd.read_csv(join(db_path,file_name + '.csv'))['values'].to_numpy()
            X[i, :] = result[:ENTRY_SIZE]
        except FileNotFoundError as e:
            if verbose :
                print("Erreur sur le fichier : ", file_name, "message : ", e)
            lostFiles += 1
    if verbose :
        print('Nombre de fichiers perdus : ', lostFiles)
    return X, y

#######################################
# Traitement des données              #
#######################################

SPIKES_EXPECTED_TIME = [7, 21.5, 23, 38]
#                   0    1    2   3     4     5     6   7     8   9     10  11  12
TIMES = [INTERVAL[0], 8.3, 9.6, 21, 21.4, 24.5, 25.5, 28, 29.6, 30, 31.5, 32, 38, INTERVAL[1]] # dermines les zones sur les quelles il faut être plus ou moins précis
SECTORS = [[3, 9], [1,5,7], [0,2,4, 8,10, 12], [11, 6]] # indice des zones à échantillonage [[très élévé], [élevé], [moyen], [faible]]


def detectSpikes(path : str, molecules : list):#->list[float]
    """Détecte les temps de rétention pour chaque molécules.

    Args:
        path (str): chemin du fichier -ms.csv
        molecules (list): liste des molécules dont on souhaite connaitre le temps de rétention

    Raises:
        ReadDataException: en cas de d'inpossibilité d'ouvrir le fichier

    Returns:
        list[float]: La liste des temps de rétention trouvés (la valeur du temps est None si un des pics n'est pas présent)
    """
    # mise en place du DataFrame
    try:
        df = pd.read_csv(path, header=None, skiprows=range(17), usecols=[1,2], encoding="ISO-8859-1")
    except FileNotFoundError:
        raise ReadDataException("Impossible de lire le fichier : " + path)
    
    index_nan = df[df[1].isnull()].index[0]
    df = df.drop(df.index[index_nan-1 : index_nan+7])
    # detection de chaque molecules
    times = []
    for molecule in molecules:
        time = df[df[2].isin(molecule)][1].values # liste des temps correspondant
        if len(time)>0:
            times.append(float(time[0])) # on prend le premier trouvé (en général c'est le plus précis)
        else:
            times.append(None)
    return times


def normalise(df : pd.DataFrame, spikes : list)->pd.DataFrame:
    """Passage en log, normalisation, conversion de l'indice en temps, et resample."""
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    # alignement des pics
    df = alignSpikes(df, spikes)
    # selection de l'intervalle
    df = df.drop(df[df.index >= INTERVAL[1]].index)
    df = df.drop(df[df.index <= INTERVAL[0]].index)
    # passage en log
    df[df['values'] == 0] = 0.1  # pour ne pas avoir de - inf
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    return df
    
    
def adaptCurve(df : pd.DataFrame, spikes : list)->pd.DataFrame:
    """Ré-échantillonnage du chromatogramme pour avoir des écart de temps constant et réduire le nombre de points.

    Args:
        df (pd.DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention .
        spikes (list): [n'est plus utilisé].

    Returns:
        pd.DataFrame: dataframe de départ après ré-échantillonnage.
    """
    # re-echantillonnage    
    df = resampleByPart(df)
    df.index = df.index.total_seconds()/60 # on remet le temps de manière plus exploitable
    return df

def resampleByPart(df : pd.DataFrame) -> pd.DataFrame:
    """Echantillonage du chromatogramme en appliquant la bonne fréquence pour chaque zones de temps.

    Args:
        df (pd.DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention. 

    Returns:
        pd.DataFrame: chromatogramme ré-échantilloné par zones.
        Remarque : l'indice est maintenant en TimeDelta et plus en minutes.
    """
    # sélection des intervalles
    parts = [df[(df.index > TIMES[i-1]) & (df.index <= TIMES[i])] for i in range(1, len(TIMES))] 
    # conversion de la durée en timedelta pour pouvoir faire un resample
    for p in range(len(parts)): 
        parts[p].index = pd.to_timedelta(parts[p].index, 'min') 

    for i in range(len(parts)):
        # zones à fréquence d'échantilonnage très élevée
        if i in SECTORS[0] : 
            parts[i] = parts[i].resample(rule=RESAMPLE_MS[0]).max().interpolate(method='polynomial', order=3)
        # zones à fréquence d'échantilonnage élevée
        if i in SECTORS[1] : 
            parts[i] = parts[i].resample(rule=RESAMPLE_MS[1]).max().interpolate(method='polynomial', order=3)
        # zone à fréquence d'échantilonnage moyenne
        if i in SECTORS[2] :
            parts[i] = parts[i].resample(rule=RESAMPLE_MS[2]).max().interpolate(method='polynomial', order=3)
        # zone à fréquence d'échantilonnage faible
        if i in SECTORS[3] :
            parts[i] = parts[i].resample(rule=RESAMPLE_MS[3]).max().interpolate(method='polynomial', order=3)

    return pd.concat(parts)

#ajouter un pic de référence à acide lactique 
def alignSpikes(df : pd.DataFrame, spikes : list) -> pd.DataFrame:
    """Aligne les pics détectés avec leurs référence en faisant un décalage temporel
    Remarque : l'argument df peut être modifié dans cette fonction.

    Args:
        df (pd.DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention.
        spikes (list): les temps de rétention de chaque pics (l'indice devant correspondre à la liste MOLECULE).

    Returns:
        pd.DataFrame: le chromatogramme dont les temps de rétention ont été ajustés.
    """
    time = df.index.copy().to_numpy() # pour ne pas modifier df
    # recherche des indices des pics (second pic pas toujours présent)
    if spikes[0] != None:
        zeroSpikeIndex = getTimeIndex(time, spikes[0])
    else :
        zeroSpikeIndex = getTimeIndex(time, SPIKES_EXPECTED_TIME[0])
    if spikes[1] != None:
        firstSpikeIndex = getTimeIndex(time, spikes[1])
    else :
        firstSpikeIndex = getTimeIndex(time, SPIKES_EXPECTED_TIME[1])
    if spikes[2] is not None:
        secondSpikeIndex = getTimeIndex(time, spikes[2])
    if spikes[3] != None:
        thirdSpikeIndex = getTimeIndex(time, spikes[3])
    else :
        thirdSpikeIndex = getTimeIndex(time, SPIKES_EXPECTED_TIME[3])
    shiftValues(time[:zeroSpikeIndex], time[0], SPIKES_EXPECTED_TIME[0])
    # Alignement du premier pic
    shiftValues(time[zeroSpikeIndex:firstSpikeIndex], SPIKES_EXPECTED_TIME[0], SPIKES_EXPECTED_TIME[1])
    # si présent alignement du second si présent et du troisième
    if spikes[1] is not None:
        shiftValues(time[firstSpikeIndex:secondSpikeIndex], SPIKES_EXPECTED_TIME[1], SPIKES_EXPECTED_TIME[2])
        shiftValues(time[secondSpikeIndex:thirdSpikeIndex], SPIKES_EXPECTED_TIME[2], SPIKES_EXPECTED_TIME[3])
    else :
        shiftValues(time[firstSpikeIndex:thirdSpikeIndex], SPIKES_EXPECTED_TIME[1], SPIKES_EXPECTED_TIME[3])
    # alignement du troisième pic à la fin
    shiftValues(time[thirdSpikeIndex:], SPIKES_EXPECTED_TIME[3], time[-1])
    df.index = time
    return df

def shiftValues(t : np.ndarray, start : int, end : int) -> None:
    """Décale les valeurs de temps linéairement dans le but de recaler les extrémités sur les références.

    Args:
        t (np.ndarray): vecteur contenant les temps de rétention.
        start (int): temps de référence pour le début de l'intervalle.
        end (int): temps de référence pour la fin de l'intervalle.
    """
    oldStart = t[0]
    oldEnd = t[-1]
    t -= (t[0] - start)
    for i in range(len(t)):
        t[i] = start + (end-start) * (t[i] - start)/(oldEnd-oldStart)


#######################################
# Compensation du biais               #
#######################################

def butter_lowpass(cutoff : float, fs : float, order=5):
    """Création d'un filtre passe bas.

    Args:
        cutoff (float): fréquence de coupure du filtre.
        fs (float): fréquence du signal.
        order (int, optional): ordre du filtre. Defaults to 5.

    Returns:
        Tuple de np.ndarray : filtre passe bas.
    """
    b, a = butter(order, cutoff, btype='lowpass', analog=False, fs=fs)
    return b, a

def butter_lowpass_filter(data : pd.DataFrame, cutoff : float, fs : float, order=5) -> pd.DataFrame:
    """Application d'un filtre passe bas sur le signal

    Args:
        data (pd.DataFrame): valeurs du signal avec en indice le temps.
        cutoff (float): fréquence de coupure du filtre.
        fs (float): fréquence du signal.
        order (int, optional): ordre du filtre. Defaults to 5.

    Returns:
        pd.DataFrame: signal après filtrage.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data) # n'introduit de déphasage
    return y

def substractBias(df : pd.DataFrame)->pd.DataFrame:
    """Calcul et retire le bias du chromatogramme.

    Args:
        df (pd.DataFrame): chromatogramme dont les valeurs contienent un bias.

    Returns:
        pd.DataFrame: chromatogramme sans bias
    """
    # filtrage
    bias = computeBiasWithResample(df)

    if (NOISE_FREQ is None):
        substractBiasResampled(df, bias)
    else :
        df = butter_lowpass_filter(df['values'], NOISE_FREQ, 1/PERIOD[1]) # attention on perd la hauteur relative entre les pics parfois
        substractBiasResampled(df, bias)
    # seuillage
    df.loc[df["values"] < THRESHOLD, 'values'] = 0
    return df

def computeBiasWithResample(values : pd.DataFrame) -> pd.DataFrame:
    """Calcul du bias dans un chromatogramme, ce calcul prend en compte les zones de fréquences d'échantillonnage différentes en faisant un ré-échantillonnage, permettant d'éviter des discontinuités dans le calcul du bias entre les zones d'échantillonnage.

    Args:
        values (pd.DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention.

    Returns:
        pd.DataFrame: les valeurs du bias avec en indice le temps de rétention
    """
    # attention parfois une petite baisse sans explication sur les derniers points
    bias = values.copy()
    # resample
    bias.index = pd.to_timedelta(bias.index, 'min') # pour le resample
    bias = bias.resample(rule=RESAMPLE_MS[0]).pad()    
    # ajout de valeurs à gauche et à droite pour ne pas avoir d'effet de bord sur le filtrage
    padding = pd.DataFrame([bias['values'].iloc[0]] * PADDING)    
    padding.rename(columns = {padding.columns[0]: 'values'}, inplace=True)
    padding.index = pd.timedelta_range('1s', '5s', periods=PADDING)
    bias = pd.concat([padding, bias]) # pour compenser le padding du filtre butter à 0 et éliminer l'effet de bord
    padding['values'] = [bias['values'].iloc[-1]] * PADDING
    bias = pd.concat([bias, padding])
    bias.index = bias.index.total_seconds()/60 # on remet le temps de manière plus exploitable    
    # filtrage
    bias = bias.rolling(30, center=True).min()
    # remplissage des nan introduits
    bias = bias.fillna(method='ffill') # on retire les Nan en forward et backward (pour être sûr qu'il n'y en ai plus)
    bias = bias.fillna(method='bfill')    
    bias['values'] = butter_lowpass_filter(bias['values'].to_numpy(), BIAS_FREQ, 1 / PERIOD[0])
    return bias

def substractBiasResampled(df : DataFrame, bias : pd.DataFrame) -> pd.DataFrame:
    """Soustraction du bias au chromatogramme quand les échelles de temps sont différentes

    Args:
        df (DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention.
        bias (pd.DataFrame): les valeurs du bias avec en indice le temps de rétention.

    Returns:
        pd.DataFrame: le chromatogramme sans bias
    """
    values = df['values'].to_numpy()
    for i in range(len(df)-1):
        dfValue = df['values'].iloc[i]
        biasValue = bias['values'].iloc[getTimeIndex(bias.index, df.index[i])]
        values[i] = dfValue - biasValue
    values[-1] -= bias['values'].iloc[len(bias) - PADDING] # compenser le cas ou l'on cherche plus grand que 45 (qui n'existe pas)
    df['values'] = values
    return df

def resizeBias(df : DataFrame, bias: DataFrame) -> np.ndarray:
    """Fait correspondre les temps de rétention du bias et du chromatogramme. Utile pour soustraire les valeurs du bias à celle du chromatogramme.
    Remarque : cette fonction n'est plus utilisée, car l'échantillonage par zone empêche le bon calcul du bias par cette méthode. 

    Args:
        df (DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention.
        bias (DataFrame): les valeurs du bias avec en indice le temps de rétention.

    Returns:
        np.ndarray: les temps de rétention corrigés.
    """
    values = [[]] * len(df)
    for i in range(len(df)):
        biasValue = bias['values'].iloc[getTimeIndex(bias.index, df.index[i])]
        values[i] = biasValue
    values[-1] = bias['values'].iloc[len(bias) - PADDING] # compenser le cas ou l'on cherche plus grand que 45 (qui n'existe pas)
    return values

def computeBiasByPart(values : pd.DataFrame) -> np.ndarray:
    """Calcul du bias dans un chromatogramme, ce calcul prend en compte les zones de fréquences d'échantillonnage différentes en calculant le bias sur chaque zone indépendement.
    Remarque : cette fonction n'est plus utilisée, car l'échantillonage par zone empêche le bon calcul du bias par cette méthode. 

    Args:
        values (pd.DataFrame): les valeurs du chromatogramme avec en indice le temps de rétention.

    Returns:
        np.ndarray: les valeurs du bias pour chaque temps de rétention du chromatogramme
    """
    # ajout de valeurs à gauche et à droite pour ne pas avoir d'effet de bord sur le filtrage
    padding = pd.DataFrame([np.NaN] * PADDING)
    padding.index = pd.timedelta_range('1s', '5s', periods=PADDING)
    temp = pd.concat([padding, values]) # pour compenser le padding du filtre butter à 0 et éliminer l'effet de bord
    temp = pd.concat([temp, padding])
    temp = temp.fillna(method='ffill') # on retire les Nan en forward et backward (pour être sûr qu'il n'y en ai plus)
    temp = temp.fillna(method='bfill')
    #temp.index = temp.index.total_seconds()/60 # mise du temps en seconde pour faciliter les calculs suivant
    indexes = [getTimeIndex(temp.index, i) for i in TIMES]
    indexes[-1] = len(temp) - PADDING # on corrige la position du dernier point au cas où il n'est pas exactement à 45.01

    # sélection des intervalles avec une marge de PADDING de chaque coté de l'intervalle pour éviter les problèmes de continuité
    bias = temp['values'].to_numpy()
    parts = [bias[indexes[i-1] - PADDING : indexes[i] + PADDING] for i in range(1, len(indexes))] 
    result = [[]] * len(parts)
    # application du filtre
    for i in range(len(parts)): 
        # zones à fréquence d'échantilonnage très élevée
        if i in SECTORS[0] : 
            result[i] = butter_lowpass_filter(parts[i], BIAS_FREQ, 1/PERIOD[0])[PADDING:-PADDING]
        # zones à fréquence d'échantilonnage élevée
        if i in SECTORS[1] : 
            result[i] = butter_lowpass_filter(parts[i], BIAS_FREQ, 1/PERIOD[1])[PADDING:-PADDING]
        # zones à fréquence d'échantilonnage moyenne
        if i in SECTORS[2] : 
            result[i] = butter_lowpass_filter(parts[i], BIAS_FREQ, 1/PERIOD[2])[PADDING:-PADDING]
        # zone à fréquence d'échantilonnage faible
        if i in SECTORS[3] :
            result[i] = butter_lowpass_filter(parts[i], BIAS_FREQ, 1/PERIOD[3])[PADDING:-PADDING]
    
    return np.concatenate(result)

#######################################
# Autres                              #
#######################################

def getTimeIndex(t : np.ndarray, timeValue) -> int:
    """Donne l'indice de la première valeur dans t qui est supérieur ou égale à timeValue

    Args:
        t (np.ndarray): vecteur correspondant au temps
        timeValue ([type]): valeur recherchée

    Returns:
        int: indice de la valeur la plus proche (et inférieur) à la valeur recherchée.
    """
    return np.argmax(t>=timeValue)


