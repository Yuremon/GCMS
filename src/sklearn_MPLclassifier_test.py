import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import tools
from importlib import reload
import os


reload(tools)

nb_col = 505
path_train = "./data/train/"
path_test = "./data/test/"

X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)

#files_train = os.listdir(path_train)
#files_test = os.listdir(path_test)
#nb_train_ex = len(files_train)
#X_train = np.zeros((nb_train_ex,nb_col))

#i = 0
#for f in files_train:
#    if not os.path.isdir(f):
#        try:
#            train_csv = pd.read_csv(path_train+"/"+f)
#            X_train[i, :] = train_csv[:nb_col]
#        except FileNotFoundError as e:
#            print("Not such file: "+f+" message: "+ e)

#print("flag")

clf = MLPClassifier(hidden_layer_sizes=(), momentum=0.9,max_iter=600,activation="relu")
clf.fit(X_train_origin,y_train)
print(clf.score(X_test_origin,y_test))
#print(clf.predict_proba(X_test_origin))
