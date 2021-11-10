from shutil import copyfile
import pandas as pd
import numpy as np

def remove_name_chromatogram(f, path):
    """retire les informations en début de chromatograme du descripteur de fichier f et sauvegarde le contenu au chemin path"""
    for i in range(5):
        f.readline()
    content = f.read()
    f2 = open(path,'w')
    f2.write('\n\n\n\n\n')
    f2.write(content)
    f2.close()
    f.close()

def remove_name_ms(f, path):
    """retire les informations en début de chromatograme du descripteur de fichier f et sauvegarde le contenu au chemin path"""
    string = ''
    for i in range(4):
        string += f.readline()
    for i in range(9):
        f.readline()
    content = f.read()
    f2 = open(path,'w')
    f2.write(string + '\n' * 7)
    f2.write(content)
    f2.close()
    f.close()

names = []

#f = open('data\\data-temp\\aafifl-20190313-chromatogram.csv')
#remove_first_lines(f,'data/data/1.csv')
#f.close()

old_path_train = './data/train-temp/'
old_path_test = './data/test-temp/'
new_path_train = './data/train/'
new_path_test = './data/test/'

db_main = pd.read_csv('./data/all_data.csv')
db_test = pd.read_csv(old_path_test +'database.csv')
db_train = pd.read_csv(old_path_train +'database.csv')

for i in range(len(db_main['file'])):
    name = db_main['file'][i]
    names.append([name,i])
    # replacement des noms dans la base
    db_main.replace(name, i, inplace=True)
    if np.isin(name, db_test['file']):
        try:
            db_test.replace(name,i, inplace=True)
            copyfile(old_path_test+name+'.csv', new_path_test+str(i)+'.csv') 
        except FileNotFoundError:
            print("Problème 1 : ", old_path_test+name+'.csv')
            db_test.replace(i,name, inplace=True)
            db_main.replace(i,name, inplace=True)
            continue
    else:
        if not np.isin(name, db_train['file']):
            print('AHHHHHHHHHHHHHHH')
        try:
            db_train.replace(name,i, inplace = True)
            copyfile(old_path_train+name+'.csv', new_path_train+str(i)+'.csv') 
        except FileNotFoundError:
            print("Problème 2 : ", old_path_train+name+'.csv')
            db_train.replace(i,name, inplace=True)
            db_main.replace(i,name, inplace=True)
            continue
    # Deplacement du fichier non traite
    try:
        f=open('./data/data-temp/'+name+'-chromatogram.csv','r')
        remove_name_chromatogram(f,'./data/all-data/'+str(i)+'-chromatogram.csv')
        f=open('./data/data-temp/'+name+'-ms.csv','r')
        remove_name_ms(f,'./data/all-data/'+str(i)+'-ms.csv')
    except FileNotFoundError:        
        print("Problème 3 : ", './data/data-temp/'+name+'-chromatogram.csv')
        db_train.replace(i,name, inplace=True)
        db_test.replace(i,name, inplace=True)
        db_main.replace(i,name, inplace=True)
        continue

print(db_main.head())
db_main.to_csv('./data/all-data/database.csv')
print(db_test.head())
db_test.to_csv('./data/test/database.csv')
print(db_train.head())
db_train.to_csv('./data/train/database.csv')

table = pd.DataFrame(names,columns=['name','number'])
print(table.head())
table.to_csv('data/table.csv')