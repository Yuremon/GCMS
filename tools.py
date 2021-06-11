import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from os import listdir
from os.path import join
from typing import Tuple

PERIOD = 3000 
RESAMPLE_MS = str(PERIOD) +'ms'
INTERVAL = [6.01,45.01]
NOISE_FREQ = None #0.09
BIAS_FREQ = 0.007
THRESHOLD = 1.7
SPIKES = [22, 25, 38]
SPIKES_EXPECTED_TIME = [22, 25, 38]

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
    # alignement des pics
    df = alignSpikes(df)
    # réduction de l'intervalle
    df = df.drop(df[df.index > INTERVAL[1]].index)
    df = df.drop(df[df.index < INTERVAL[0]].index)
    # passage en log
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    # re-echantillonnage
    df.index = pd.to_timedelta(df.index, 'min')  # conversion de la durée en timedelta pour pouvoir faire un resample
    df = df.resample(rule=RESAMPLE_MS).max().interpolate(method='polynomial', order=3) # on change l'échantillonage pour qu'il soit constant.
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    return df

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
    LENGHT_ADDED = 45
    bias = df['values'].rolling(15).min()
    bias = pd.concat([pd.DataFrame([np.NaN] * LENGHT_ADDED), bias]) # pour compenser le padding du filtre à 0 et éliminer l'effet de bord
    bias = pd.concat([bias, pd.DataFrame([np.NaN] * LENGHT_ADDED)])
    bias = bias.fillna(method='ffill') # on retire les Nan en forward et backward (pour être sûr qu'il n'y en ai plus)
    bias = bias.fillna(method='bfill')
    # filtrage
    bias = bias.squeeze()
    bias = butter_lowpass_filter(bias, BIAS_FREQ, 1/PERIOD*1000)
    bias = bias[LENGHT_ADDED : -LENGHT_ADDED]
    if (NOISE_FREQ is None):
        df['values'] -= bias
    else :
        df = butter_lowpass_filter(df['values'], NOISE_FREQ, 1/PERIOD*1000) # attention on perd la hauteur relative entre les pics parfois
        df -= bias
    # seuillage
    df.loc[df["values"] < THRESHOLD, 'values'] = 0
    return df

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
    """Donne l'indice de la première valeur dans t qui est supérieur ou égale à time"""
    return np.argmax(t>=timeValue)

def readAllData(path : str) -> list[pd.DataFrame]:
    files = [f for f in listdir(path) if f.endswith(".csv") or f.endswith(".CSV")]
    dataList = []
    for file in files:
        dataList.append(readAndAdaptDataFromCSV(join(path, file)))
    return dataList
