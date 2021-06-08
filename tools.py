from typing import Tuple
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.metrics import mean_squared_error

PERIOD = 3000 
RESAMPLE_MS = str(PERIOD) +'ms'
INTERVAL = [6.01,45.01]
NOISE_FREQ = None #0.09
BIAS_FREQ = 0.007
THRESHOLD = 1.7
SPIKES = [21.5,25.6, 30.2]
SPIKES_EXPECTED_TIME = [10, 24, 38]

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
    # réduction de l'intervalle
    df = df.drop(df[df.index > INTERVAL[1]].index)
    df = df.drop(df[df.index < INTERVAL[0]].index)
    # passage en log
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    # re-echantillonnage
    df['time'] = pd.to_timedelta(df.index, 'min')  # conversion de la durée en timedelta pour pouvoir faire un resample
    df.set_index('time', inplace=True) # met le temps en indice du DataFrame
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
    y = lfilter(b, a, data)
    return y

def substractBias(df : pd.DataFrame)->pd.DataFrame:
    """Filtrage et suppression du bias."""
    bias = df['values'].rolling(10).min()
    bias = bias.fillna(method='ffill')
    bias = bias.fillna(method='bfill')
    bias = butter_lowpass_filter(bias, BIAS_FREQ, 1/PERIOD*1000)
    if (NOISE_FREQ is None):
        df['values'] -= bias
    else :
        df = butter_lowpass_filter(df, NOISE_FREQ, 1/PERIOD*1000) # attention on perd la hauteur relative entre les pics parfois
        df -= bias
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