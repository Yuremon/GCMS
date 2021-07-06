import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy.signal import butter, filtfilt
from os import listdir
from os.path import join
from typing import Tuple

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
    """Exception levée en cas de problème de lecture de la base de donnée"""

class Data:
    def __init__(self, name, df : pd.DataFrame = None, state : bool = None, molecules : list = MOLECULES):
        self.name = name
        self.df = df
        self.state = state
        self.spikes = None
        self.molecules = molecules
        self.problems = []

    def readCSV(self, path):
        self.df = readCSV(join(path, self.name + CHROM_EXT))

    def detectSpikes(self, path):
        self.spikes = detectSpikes(join(path, self.name + MOL_EXT), MOLECULES) # on ne passe que les molecules de référence

    def alignSpikes(self):
        self.df = alignSpikes(self.df, self.spikes)

    def ruleBasedCheck(self):
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
    
    def printProblems(self):
        for pb in self.problems:
            print(f"La molécule {pb[2]} est présente à {int(pb[1])} % du pic de référence à {pb[0]} minutes")
        

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
ENTRY_SIZE = 504

#######################################
# Lecture des données                 #
#######################################

CHROM_EXT = '-chromatogram.csv'
MOL_EXT = '-ms.csv'

def readCSV(path : str)->pd.DataFrame:
    """Création du DataFrame a partir d'un csv."""
    # lecture du fichier csv
    df = pd.read_csv(path, header=None, skiprows=[0,1,2,3,4,5], index_col=0)
    #suppression de la colonne vide
    df.drop(df.columns[1], axis=1, inplace=True)
    df.rename(columns = {df.columns[0]: 'values'}, inplace=True)
    df['values'] = df['values'].fillna(0) # on enlève les éventuels nan
    return df


def readAndAdaptDataFromCSV(path, name) -> Data:
    """Lit le fichier et retourne le DataFrame après traitement."""
    dt = Data(name)
    dt.readCSV(path)
    dt.detectSpikes(path)
    dt.ruleBasedCheck()
    dt.df = normalise(dt.df,dt.spikes)
    df = adaptCurve(dt.df, dt.spikes)
    dt.df = substractBias(df)
    if len(dt.df) != ENTRY_SIZE and len(dt.df) != ENTRY_SIZE+1:
        raise ReadDataException("Problème de taille sur le chromatogramme")
    return dt

def readAllData(path : str) -> list[Data]:
    """retourne la liste de DataFrame correspondant à tous les fichiers csv présent dans path"""
    files = [f for f in listdir(path) if f.endswith(CHROM_EXT)]
    dataList = []
    for file in files:
        dt = readAndAdaptDataFromCSV(path, file[:-17]) # données traitées
        dataList.append(dt)
    return dataList

def readListOfData(files : np.ndarray, path : str) -> list[Data]:
    """retourne la liste de DataFrame correspondant à tous les fichiers csv de db au chemin path"""
    dataList = []
    for file in files:
        try:
            dt = readAndAdaptDataFromCSV(path, file)
        except ReadDataException as e :
            print(e)
        finally:
            dataList.append(dt)
        
    return dataList

def getData(file_path : str, db_path : str) -> Tuple[np.ndarray, np.ndarray]:
    """en connaissant le fichier où sont stockées les informations sur les données et le chemin jusqu'aux chromatogrammes
    retourne un tableau numpy contenant les données d'entrées 
    ainsi qu'un second contenant les données de sortie pour l'entrainement (normale (1) /non normale (0))"""
    # lecture du fichier représentant la base de donnée
    db = pd.read_csv(file_path)
    # y représente la sortie (normale ou non)
    y = db['status'].to_numpy()

    # lecture de tous les chromatogrammes listées
    files = db['file'].to_numpy()
    n = len(files)
    # X représente toutes les entrées (une entrée par ligne)
    X = np.zeros((n, ENTRY_SIZE))
    lostFiles = 0
    for i in range(n):
        # chaque ligne correspond aux valeurs au cours du temps du chromatogramme
        try:
            result = readAndAdaptDataFromCSV(db_path, files[i]).df['values'].to_numpy()
            X[i, :] = result[:ENTRY_SIZE]
        except ReadDataException as e:
            if verbose :
                print("Erreur sur le fichier : ", files[i], "message : ", e)
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


def detectSpikes(path : str, molecules : list) -> list[str]:
    """Retourne la liste des temps de rétention des molecules d'après le fichier undiqué dans path"""
    # mise en place du DataFrame
    try:
        df = pd.read_csv(path, header=None, skiprows=range(17), usecols=[1,2])
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
    df = df.drop(df[df.index >= INTERVAL[1]].index)
    df = df.drop(df[df.index <= INTERVAL[0]].index)
    # passage en log
    df[df['values'] == 0] = 0.1  # pour ne pas avoir de - inf
    df = np.log(df)
    # normalisation
    df = (df - df.mean())/df.std()
    # Etalonner sur le premier pic (taille de 10 pour celui-ci) 
    # remarque : attention si la valeur n'est pas précise les pics seront plus grand que prévu
    timeSpike1 = getTimeIndex(df.index, spikes[1])
    df = df / df.iloc[timeSpike1] * 10
    return df
    
    
def adaptCurve(df : pd.DataFrame, spikes : list)->pd.DataFrame:
    # alignement des pics
    df = alignSpikes(df, spikes)
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
# Autres                              #
#######################################

def getTimeIndex(t : np.ndarray, timeValue) -> int:
    """Donne l'indice de la première valeur dans t qui est supérieur ou égale à timeValue"""
    return np.argmax(t>=timeValue)


