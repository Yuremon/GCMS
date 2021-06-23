import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.signal import butter, filtfilt
from os import listdir
from os.path import join
from typing import Tuple

class Data:
    def __init__(self, name, df : pd.DataFrame = None, state : bool = None):
        self.name = name
        self.df = df
        self.state = state
        self.spikes = None
    def readCSV(self, path):
        self.df = readCSV(join(path, self.name))
    def detectSpikes(self, path):
        self.spikes = path
    def alignSpikes(self):
        self.df = alignSpikes(self.df, self.spikes)
    def ruleBasedCheck(self):
        return ruleBasedCheck(self.df)
        

#######################################
# Constantes                          #
#######################################

INTERVAL = [6.01,45.01]
NOISE_FREQ = None #0.09
BIAS_FREQ = 0.002
THRESHOLD = 3.2
PADDING = 45 # attention la valeur doit être supérieur à 1
PERIOD = [1, 2, 6, 20] 
RESAMPLE_MS = [str(p) +'s' for p in PERIOD]
ENTRY_SIZE = 492

#######################################
# Lecture des données                 #
#######################################

def readCSV(path : str)->pd.DataFrame:
    """Création du DataFrame a partir d'un csv."""
    # lecture du fichier csv
    df = pd.read_csv(path, header=None, skiprows=[0,1,2,3,4,5], index_col=0)
    #suppression de la colonne vide
    df.drop(df.columns[1], axis=1, inplace=True)
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df['values'] = df['values'].fillna(0) # on enlève les éventuels nan
    return df


def readAndAdaptDataFromCSV(path, name, spikes : list = None) -> Data:
    """Lit le fichier et retourne le DataFrame après traitement."""
    dt = Data(name)
    dt.readCSV(path)
    df = dt.df
    # si l'on ne connait pas les positions des pics de référence
    if spikes is None:
        dt.detectSpikes(SPIKES)
    else:
        dt.detectSpikes(spikes)
    df = adaptCurve(df, dt.spikes)
    dt.df = substractBias(df)
    return dt

def readAllData(path : str) -> list[Data]:
    """retourne la liste de DataFrame correspondant à tous les fichiers csv présent dans path"""
    files = [f for f in listdir(path) if f.endswith(".csv") or f.endswith(".CSV")]
    dataList = []
    for file in range(len(files)):
        if (file < len(SPIKES_LIST)): # tant qu'on a pas la détermination de pics automatique
            dt = readAndAdaptDataFromCSV(path, files[file], SPIKES_LIST[file]) # données traitées
        else :
            dt = readAndAdaptDataFromCSV(path, files[file])
        dataList.append(dt)
    return dataList

def readListOfData(db : np.ndarray, path : str) -> list[Data]:
    """retourne la liste de DataFrame correspondant à tous les fichiers csv de db au chemin path"""
    files = [file + '.CSV' for file in db]
    dataList = []
    for file in files:
        dataList.append(readAndAdaptDataFromCSV(path, file))
    return dataList

def getData(file_path : str, db_path : str) -> Tuple[np.ndarray, np.ndarray]:
    """en connaissant le fichier où sont stockées les informations sur les données et le chemin jusqu'aux chromatogrammes
    retourne un tableau numpy contenant les données d'entrées 
    ainsi qu'un second contenant les données de sortie pour l'entrainement (normale (1) /non normale (0))"""
    # lecture du fichier représentant la base de donnée
    db = pd.read_csv(file_path, index_col=False, names=['file', 'normal', 'label'])
    # y représente la sortie (normale ou non)
    y = db['normal'].to_numpy()

    # lecture de tous les chromatogrammes listées
    files = [file + '.CSV' for file in db['file']]
    n = len(files)
    # X représente toutes les entrées (une entrée par ligne)
    X = np.zeros((n, ENTRY_SIZE))
    
    for i in range(n):
        # chaque ligne correspond aux valeurs au cours du temps du chromatogramme
        X[i, :] = readAndAdaptDataFromCSV(db_path, files[i]).df['values'].to_numpy()
    return X, y

#######################################
# Traitement des données              #
#######################################

SPIKES = [7, 21.978, 23.204, 38.005] # [22.528, 23.749, 38.614]
SPIKES_EXPECTED_TIME = [7, 21.5, 23, 38]
#                   0    1    2   3     4     5     6   7     8   9     10  11  12
TIMES = [INTERVAL[0], 8.3, 9.6, 21, 21.4, 24.5, 25.5, 28, 29.6, 30, 31.5, 32, 38, INTERVAL[1]] # dermines les zones sur les quelles il faut être plus ou moins précis
SECTORS = [[3, 9], [1,5,7], [0,2,4, 8,10, 12], [11, 6]] # indice des zones à échantillonage [[très élévé], [élevé], [moyen], [faible]]
# pour calibrer correctement les pics sans avoir la détection automatique
SPIKES_LIST = [[7, 21.978, 23.204, 38.005], [7, 22.604, 23.821, 38.661], [7, 21.421, 22.645, 37.468], [7, 21.949, 23.1666, 37.985], [7, 22.046, 23.265, 38.100],
    [7, 21.415, 22.650, 37.459], [7, 22.527, 23.747, 38.605], [7, 22.586, 23.804, 38.655], [7, 22.586, 23.804, 38.655], [7, 22.528, 23.748, 38.611],
    [7, 21.952, 23.181, 37.988], [7, 21.953, 23.181, 37.988]]

def detectSpikes(path : str, molecules : list) -> list[str]:
    """Retourne la liste des temps de rétention des molecules d'après le fichier undiqué dans path"""
    # mise en place du DataFrame
    df = pd.read_csv(path, header=None, skiprows=range(17), usecols=[1,2])
    index_nan = df[df[1].isnull()].index[0]
    df = df.loc[0:index_nan - 1, :]
    # detection de chaque molecules
    times = []
    for molecule in molecules:
        time = df[df[2] == molecule][1].values # liste des temps correspondant
        if (len(time)<1):
            time = None
        else:
            time = float(time[0]) # normalement de taille 1 donc on prend le premier
        times.append(time)
    return times


def adaptCurve(df : pd.DataFrame, spikes : list)->pd.DataFrame:
    """Passage en log, normalisation, conversion de l'indice en temps, et resample."""
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df = df.drop(df[df.index >= INTERVAL[1]].index)
    df = df.drop(df[df.index <= INTERVAL[0]].index)
    # alignement des pics
    df = alignSpikes(df, spikes)
    # passage en log
    df[df['values'] == 0] = 0.01  # pour ne pas avoir de - inf
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    # Etalonner sur le premier pic (taille de 10 pour celui-ci) 
    # remarque : attention si la valeur n'est pas précise les pics seront plus grand que prévu
    timeSpike1 = getTimeIndex(df.index, SPIKES_EXPECTED_TIME[0])
    df = df / df.iloc[timeSpike1] * 10
    # re-echantillonnage    
    df = resampleByPart(df)
    df.index = df.index.total_seconds()/60 # on remet le temps de manière plus exploitable
    return df

def resampleByPart(df : pd.DataFrame) -> pd.DataFrame:
    """Fait un échantillonnage avec une période différente pour chaque partie de la courbe"""
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
    """Aligne les pics détectés sur les références attention df sera modifié"""
    time = df.index.copy().to_numpy() # pour ne pas modifier df
    # recherche des indices des pics (second pic pas toujours présent)
    zeroSpikeIndex = getTimeIndex(time, spikes[0])
    firstSpikeIndex = getTimeIndex(time, spikes[1])
    if spikes[1] is not None:
        secondSpikeIndex = getTimeIndex(time, spikes[2])
    thirdSpikeIndex = getTimeIndex(time, spikes[3])
    shiftValues(time[:zeroSpikeIndex], time[0], SPIKES_EXPECTED_TIME[0])
    # Alignement du premier pic
    shiftValues(time[zeroSpikeIndex:firstSpikeIndex], time[1], SPIKES_EXPECTED_TIME[1])
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
    """Décale les valeurs pour que t ai des valeurs de temps entre start et end"""
    #print('start : ', start, '  end : ', end)
    oldStart = t[0]
    oldEnd = t[-1]
    #print('old start : ', oldStart, '  old end : ', oldEnd)
    t -= (t[0] - start)
    for i in range(len(t)):
        t[i] = start + (end-start) * (t[i] - start)/(oldEnd-oldStart)


#######################################
# Compensation du biais               #
#######################################

def butter_lowpass(cutoff : float, fs : float, order=5)-> Tuple[np.ndarray, np.ndarray]:
    """Création du filtre."""
    b, a = butter(order, cutoff, btype='lowpass', analog=False, fs=fs)
    return b, a

def butter_lowpass_filter(data : pd.DataFrame, cutoff : float, fs : float, order=5) -> pd.DataFrame:
    """Application du filtre."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data) # n'introduit de déphasage
    return y

def substractBias(df : pd.DataFrame)->pd.DataFrame:
    """Filtrage et suppression du bias."""
    # filtrage
    #bias = computeBiasByPart(bias)
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
    """calcul du bias avec un resample pour ne pas avoir de fréquences differentes par zones"""
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

def resizeBias(df : DataFrame, bias: DataFrame) -> np.ndarray:
    """Permet de donner au bias le même échantillonnage que df"""
    values = [[]] * len(df)
    for i in range(len(df)):
        biasValue = bias['values'].iloc[getTimeIndex(bias.index, df.index[i])]
        values[i] = biasValue
    values[-1] = bias['values'].iloc[len(bias) - PADDING] # compenser le cas ou l'on cherche plus grand que 45 (qui n'existe pas)
    return values

def substractBiasResampled(df : DataFrame, bias : pd.DataFrame) -> pd.DataFrame:
    """Soustraction du bias au DataFrame quand les échelles de temps sont différentes"""
    values = df['values'].to_numpy()
    for i in range(len(df)-1):
        dfValue = df['values'].iloc[i]
        biasValue = bias['values'].iloc[getTimeIndex(bias.index, df.index[i])]
        values[i] = dfValue - biasValue
    values[-1] -= bias['values'].iloc[len(bias) - PADDING] # compenser le cas ou l'on cherche plus grand que 45 (qui n'existe pas)
    df['values'] = values
    return df

def computeBiasByPart(values : pd.DataFrame) -> np.ndarray:
    """calcul du bias pour chaque intervalle, peut induire des problèmes de continuité"""
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
# Vérification des règles prédéfinies #
#######################################

PROBLEM_TIME = [10.03, 12.524, 16.69, 19.65, 20.1, 22.16, 24.3, 24.55] # en minutes
PROBLEM_NAME = ["Acide 30H-Propionique", "Acide Méthylmalonique", "Acide Fumarique", "Acide Glutarique", "Acide 3CH3-Glutarique", "Acide Adipique", "Acide 20H-Glutarique", "Acide Homovanillic"]
#fumaruque spectre à 245
#urée spectre 189
AREA_SIZE = 0.05 # en minute
PROBLEM_THRESHOLD = 2 # hauteur à partir de laquelle on considère anormale à tester

def ruleBasedCheck(df : DataFrame) -> str:
    """Vérification de la présence de pics évidents permettant de détecter un problème
    retourne la liste des prblèmes et leur pourcentage d'augmentation (en tuples) ou une liste vide"""
    # citer la liste avec le pourcentage d'augmentation
    problems = []
    for i in range(len(PROBLEM_NAME)):
        spikeHeight = np.max(df['values'].loc[PROBLEM_TIME[i] - AREA_SIZE : PROBLEM_TIME[i] + AREA_SIZE])
        if (spikeHeight > PROBLEM_THRESHOLD):
            problems.append((spikeHeight, PROBLEM_NAME[i]))
    return problems


#######################################
# Autres                              #
#######################################

def getTimeIndex(t : np.ndarray, timeValue) -> float:
    """Donne l'indice de la première valeur dans t qui est supérieur ou égale à timeValue"""
    return np.argmax(t>=timeValue)


