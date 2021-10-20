import tools
import sys
# gestion de fichiers
import os
from os.path import split, join, exists
# gestion de fenetres
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# creation des pdf
from fpdf import FPDF

#pyinstaller --noconfirm --onefile --console --icon "C:/Users/jleva/Downloads/chromato_final.ico" --name "IA-GCMS" --debug "all"  "C:/Users/jleva/Documents/Telecom/2A/stage/GCMS/src/main.py"

def draw_figure(canvas, figure):
    """Integration de matplotlib dans la fenetre de PySimpleGUI"""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas_agg.get_tk_widget().size = (400,400)
    return figure_canvas_agg

def read(path, name):
    """Affiche les courbes pour comparer avant / après traitement directement dans la fenetre"""
    # lecture des données brutes :
    file_path = join(path, name + tools.CHROM_EXT)
    print(file_path)
    try:
        df = tools.readCSV(file_path)
    except FileNotFoundError:
        sg.popup('Erreur', 'Problème de lecture, le fichier -chromatogram.csv semble ne pas être présent')
        return None,None
    df = df.drop(df[df.index > tools.INTERVAL[1]].index)
    df = df.drop(df[df.index < tools.INTERVAL[0]].index)
    # lecture des données traités
    try:
        data = tools.readAndAdaptDataFromCSV(path, name)
    except tools.ReadDataException as e:
        sg.popup('Erreur', 'Problème de lecture' + str(e))
        return None,None
    return df,data

def plot(df, data):
    """Creation du graphique pour affichage dans la fenetre et enregistrement pour l'utiliser dans le pdf"""
    fig = plt.figure()
    plt.subplot(2,1,1)
    df['values'].plot()
    plt.title('Avant traitement')
    plt.subplot(2,1,2)
    data.df['values'].plot()
    plt.title('Après traitement')
    fig.tight_layout(pad=1.0)
    plt.savefig(fname=TEMP_PATH+'plotdf.png')
    fig_canvas_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
    problems = data.problemsDescription()
    if problems != '':
        sg.popup('Information - molécules détectés', problems)
    return fig_canvas_agg

def checkEntries(values):
    """Vérification que les valeurs de la fenetre correspondent bien à une entrée corecte dans la base de donnée"""
    label = values['label'] # vérifier qu'il y a pas de virgules
    if '"' in label:
        sg.popup('Erreur', "Le caractère '\"' est interdit dans le commentaire d'un chromatogramme")
    if ',' in label or '\n' in label:
        label = '"' + label + '"'
    label_num = values['label_num'].split(';')
    for num in label_num:
        try :
            num = int(num)
        except ValueError:
            sg.popup('Erreur', 'La valeur entrée dans le champ "Interpretation" doit être un entier compris entre 0 et 5')
            return None
        if num > 5 or num < 0:
            sg.popup('Erreur', 'La valeur entrée dans le champ "Interpretation" doit être comprise entre 0 et 5')
            return None
    if name is None:
        sg.popup('Erreur', 'Aucun fichier sélectionné')
        return None
    status = 1 if label == 0 else 0
    line = f"{name},{status},{label},{values['label_num']}\n"
    return line

def predict(df):
    """Fait la prediction à partir de l'algo d'IA entrainé"""
    entry = df['values'].to_numpy()[0:tools.ENTRY_SIZE]
    result = ia.predict([entry])[0]
    return "normal" if result==1 else "non-normal"

def createPdf(pdf_name,name, df, data, ruleBasedText, IaText):
    """Creation du pdf"""
    # cration remplissage du pdf
    pdf = PDF()
    pdf.add_page()
    pdf.insert_title(name)
    pdf.insert_charts(TEMP_PATH+'plotdf.png',0)
    pdf.insert_text("Résultat de l'intelligence artificielle : \n" + IaText,0)
    pdf.insert_text("Détection de pics : \n" + ruleBasedText,1)
    # export du pdf
    pdf.output(pdf_name,'F').encode('latin-1')


class PDF(FPDF):
    """Classe de gestion des pdf hérité de FPDF, remarque : taille du pdf en A4 210 * 297"""

    def insert_title(self, text):
        self.set_xy(0.0,0.0)
        self.set_font('Arial', 'B', 16)
        self.cell(w=210.0, h=20.0, align='C', txt=text, border=0)
    
    def insert_text(self, text, number):
        self.set_xy(0.0, 190 + 10 * number)
        self.set_font('Arial', '', 12)
        self.multi_cell(w=210.0, h=5.0, align='C', txt=text, border=0)

    def insert_charts(self, plot, number):
        self.set_xy(30.0, 20.0 + 90.0 * number)
        self.image(plot,  link='', type='', w=150, h=160)

# fenetre pour la première ouverture de l'appli
layout_first_time = [
    [sg.Text("Mise en place de l'application")],
    [sg.Text("Base de donnée"),sg.Input(), sg.FileBrowse(key="database", file_types=(("Base de donnée", "database.csv"),))],
    [sg.Text("Dossier contenant les chromatogrammes"),sg.Input(), sg.FolderBrowse(key="data_path")],
    [sg.Text("Dossier de fichiers temporaires"),sg.Input(), sg.FolderBrowse(key="temp_path")],
    [sg.Button('Ok'), sg.Button('Quitter')]
]

# structure de la fenetre
layout = [
    [sg.Text("Interpretation des chromatogrammes")],
    [sg.Text("Chromatogramme"),sg.Input(), sg.FileBrowse(key="chromatogram", file_types=(("Fichiers chromatogram", "*chromatogram.csv"),))],
    [sg.Button('Ok'), sg.Button('Quitter')],
    [sg.Canvas(key='figCanvas', size=(500,500))],
    [sg.Text("Commentaire"), sg.Input(key='label'), sg.Text('Categorie'), sg.Input(key='label_num'), sg.Button('Valider')]
]

keys_to_clear = ["label", "label_num"]

# integration de matplotlib tirée de : https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib.py

if __name__ == '__main__':

    cwd = os.getcwd()
    print('Chemin courant : ',cwd)

    # Vérifiaction si on déjà configuré l'app pour l'initialisation
    if exists("./initialisation"):
        f = open("./initialisation", 'r')
        DATABASE_PATH = f.readline()[:-1]  # on retire le \n
        DATA_PATH = f.readline()[:-1]
        TEMP_PATH = f.readline()[:-1]
        f.close()

    else :
        # dans le cas de la première ouverture de l'appli on stock tous les chemins utiles
        window = sg.Window('Mise en place', layout_first_time, element_justification='center', font='16', resizable=True)
        # boucle de la fenetre d'initialisation
        while True:
            event, values = window.read()
            # quitter l'application
            if event in (sg.WIN_CLOSED, 'Quitter'):
                window.close()
                sys.exit()
            # affichage des courbes
            if event == 'Ok':
                DATABASE_PATH = values['database']
                DATA_PATH = values['data_path'] + '/'
                TEMP_PATH = values['temp_path'] + '/' 
                print(DATABASE_PATH)
                break
        f = open("./initialisation", 'w')
        f.writelines([DATABASE_PATH + '\n',DATA_PATH + '\n',TEMP_PATH + '\n'])
        f.close()
        window.close()

    # mise en place de la fenetre principale
    ia = tools.MachineLearningTechnique()
    print(DATA_PATH)
    ia.load(DATA_PATH)

    window = sg.Window('Application d\'IA', layout, element_justification='center', font='16', resizable=True)
    fig_canvas_agg = None
    name = None
    file = open(DATABASE_PATH, 'a')
    # boucle principale
    while True:
        event, values = window.read()
        # quitter l'application
        if event in (sg.WIN_CLOSED, 'Quitter'):
            break
        # affichage des courbes
        if event == 'Ok':
            # récupération du nom des fichiers
            path,name = split(values['chromatogram'])
            print(path)
            name = name[:-len(tools.CHROM_EXT)]
            if fig_canvas_agg is not None:
                # si on a déjà un graphique d'affiché on le retire
                fig_canvas_agg.get_tk_widget().pack_forget()
            df,data = read(path, name)
            res = predict(data.df)
            # affichage des graphiques et creation du pdf
            if df is not None and data is not None:
                fig_canvas_agg = plot(df,data)
                pdf_name = join(path,name)+'.pdf'
                createPdf(pdf_name,name,df,data, data.problemsDescription() if data.problemsDescription() != '' else "aucun problème détecté", res)
                # ouverture du pdf
                os.startfile(pdf_name)
        # ajout à la base de donnée
        if event == 'Valider':
            line = checkEntries(values)
            if line is None:
                continue
            # trouver le numéro du nouveau fichier
            # ajout à la base de donnée et enregistrement des fichier traitées
            file.write(line)
            data.df['values'].to_csv(DATA_PATH + name + '.csv')
            # on vide tous les champs pour montrer que le changement est pris en compte
            for key in keys_to_clear:
                window[key]('')
    # on libère le descripteur de fichier de la base de sonnée
    file.close()
    # fermeture de la fenetre
    window.close()