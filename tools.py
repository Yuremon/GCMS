import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from os import listdir
from os.path import join
from typing import Tuple

PERIOD = [1, 2, 6, 20] 
RESAMPLE_MS = [str(p) +'s' for p in PERIOD]
INTERVAL = [6.01,45.01]
NOISE_FREQ = None #0.09
BIAS_FREQ = 0.002
THRESHOLD = 1.5
SPIKES = [21.5, 22.5, 38]
SPIKES_EXPECTED_TIME = [21.5, 22.5, 38]
PADDING = 45 # attention la valeur doit être supérieur à 1
TIMES = [INTERVAL[0], 8, 10, 21.5, 22, 24.3, 26.5, 27.8, 30.5, 31.5, 32, 38, INTERVAL[1]] # dermines les zones sur les quelles il faut être plus ou moins précis
SECTORS = [[3, 8], [1,5,7], [0,2,4,9, 11], [10, 6]] # indice des zones à échantillonage [[élevé], [moyen], [faible]]

# 21.5 et 22 essayer de faire plus precis (1s) entre 30.5 et 31.5
# faire en sorte de normaliser pour la taille du pic 1 soit tjr la meme

def readCSV(path : str)->pd.DataFrame:
    """Création du DataFrame a partir d'un csv."""
    # lecture du fichier csv
    df = pd.read_csv(path, header=None, skiprows=[0,1,2,3,4,5], index_col=0)
    #suppression de la colonne vide
    df.drop(df.columns[1], axis=1, inplace=True)
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df['values'] = df['values'].fillna(0) # on enlève les éventuels nan
    return df

def adaptCurve(df : pd.DataFrame)->pd.DataFrame:
    """Passage en log, normalisation, conversion de l'indice en temps, et resample."""
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df = df.drop(df[df.index >= INTERVAL[1]].index)
    df = df.drop(df[df.index <= INTERVAL[0]].index)
    # alignement des pics
    df = alignSpikes(df)
    # passage en log
    df[df['values'] == 0] = 0.01  # pour ne pas avoir de - inf
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    # Etalonner sur le premier pic ( taille de 10 celui-ci)
    timeSpike1 = getTimeIndex(df.index, SPIKES_EXPECTED_TIME[0])
    print(df.iloc[timeSpike1])
    df = df / df.iloc[timeSpike1] * 10
    # re-echantillonnage    
    df = resampleByPart(df)
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
    bias = df.rolling(15).min()
    # filtrage
    bias = computeBiasByPart(bias)
    #df['bias'] = bias
    if (NOISE_FREQ is None):
        df['values'] -= bias
    else :
        df = butter_lowpass_filter(df['values'], NOISE_FREQ, 1/PERIOD[1]) # attention on perd la hauteur relative entre les pics parfois
        df -= bias
    # seuillage
    df.loc[df["values"] < THRESHOLD, 'values'] = 0
    return df

def computeBiasByPart(values : pd.DataFrame) -> np.ndarray:
    """calcul du bias pour chaque intervalle"""
    padding = pd.DataFrame([np.NaN] * PADDING)
    padding.index = pd.timedelta_range('1s', '5s', periods=PADDING)
    temp = pd.concat([padding, values]) # pour compenser le padding du filtre butter à 0 et éliminer l'effet de bord
    temp = pd.concat([temp, padding])
    temp = temp.fillna(method='ffill') # on retire les Nan en forward et backward (pour être sûr qu'il n'y en ai plus)
    temp = temp.fillna(method='bfill')
    temp.index = temp.index.total_seconds()/60
    indexes = [getTimeIndex(temp.index, i) for i in TIMES]
    indexes[-1] = len(temp) - PADDING # on corrige la position du dernier point au cas où il n'est pas exactement à 45.01

    # sélection des intervalles avec une marge de LENGTH_ADDED de chaque coté de l'intervalle pour éviter les problèmes de continuité
    bias = temp['values'].to_numpy()
    parts = [bias[indexes[i-1] - PADDING : indexes[i] + PADDING] for i in range(1, len(indexes))] 
    result = [[]] * len(parts)
    for i in range(len(parts)):  
        #print('Taille de la partie ', i, ' : ', len(parts[i]) - 2 * PADDING)      
        #print('partie : ', i, parts[i])
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

def readAndAdaptDataFromCSV(path : str) -> pd.DataFrame:
    """Lit le fichier et retourne le DataFrame après traitement."""
    df = readCSV(path)
    df = adaptCurve(df)
    df = substractBias(df)
    return df

def compareCurves(oldDf : pd.DataFrame, newDf : pd.DataFrame) -> float:
    """Retourne la valeur de l'erreur quadratique moyenne entre oldDf et newDf"""
    value = mean_squared_error(oldDf, newDf)
    return value

def alignSpikes(df : pd.DataFrame) -> pd.DataFrame:
    """Aligne les pics détectés sur les références attention df sera modifié"""
    time = df.index.copy().to_numpy() # pour ne pas modifier df
    # recherche des indices des pics
    firstSpikeIndex = getTimeIndex(time, SPIKES[0])
    if SPIKES[1] is not None:
        secondSpikeIndex = getTimeIndex(time, SPIKES[1])
    thirdSpikeIndex = getTimeIndex(time, SPIKES[2])
    # Alignement du premier pic
    shiftValues(time[:firstSpikeIndex], time[0], SPIKES_EXPECTED_TIME[0])
    # si présent alignement du second si présent et du troisième
    if SPIKES[1] is not None:
        shiftValues(time[firstSpikeIndex:secondSpikeIndex], SPIKES_EXPECTED_TIME[0], SPIKES_EXPECTED_TIME[1])
        shiftValues(time[secondSpikeIndex:thirdSpikeIndex], SPIKES_EXPECTED_TIME[1], SPIKES_EXPECTED_TIME[2])
    else :
        shiftValues(time[firstSpikeIndex:thirdSpikeIndex], SPIKES_EXPECTED_TIME[0], SPIKES_EXPECTED_TIME[2])
    # alignement du troisième pic à la fin
    shiftValues(time[thirdSpikeIndex:], SPIKES_EXPECTED_TIME[2], time[-1])
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


def getTimeIndex(t : np.ndarray, timeValue) -> float:
    """Donne l'indice de la première valeur dans t qui est supérieur ou égale à timeValue"""
    return np.argmax(t>=timeValue)

def readAllData(path : str) -> list[pd.DataFrame]:
    files = [f for f in listdir(path) if f.endswith(".csv") or f.endswith(".CSV")]
    dataList = []
    for file in files:
        dataList.append(readAndAdaptDataFromCSV(join(path, file)))
    return dataList

def readListOfData(db : np.ndarray, path : str):
    files = [join(path, file + '.CSV') for file in db]
    dataList = []
    for file in files:
        dataList.append(readAndAdaptDataFromCSV(file))
    return dataList
