import tools
# gestion de fichiers
from os.path import split, join
from shutil import copyfile
# gestion de fenetres
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_figure(canvas, figure):
    """Matplotlib dans la fenetre de PySimpleGUI"""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas_agg.get_tk_widget().size = (400,400)
    return figure_canvas_agg

def readAndPlot(path, name):
    """Affiche les courbes pour comparer avant / après traitement directement dans la fenetre"""
    fig = plt.figure()
    # lecture des données brutes :
    file_path = join(path, name + tools.CHROM_EXT)
    print(file_path)
    try:
        df = tools.readCSV(file_path)
    except FileNotFoundError:
        sg.popup('Erreur', 'Problème de lecture, le fichier -chromatogram.csv semble ne pas être présent')
        return None
    df = df.drop(df[df.index > tools.INTERVAL[1]].index)
    df = df.drop(df[df.index < tools.INTERVAL[0]].index)
    # lecture des données traités
    try:
        data = tools.readAndAdaptDataFromCSV(path, name)
    except tools.ReadDataException:
        sg.popup('Erreur', 'Problème de lecture, le fichier -ms.csv est-il bien présent ?')
        return None
    # affichage
    plt.subplot(2,1,1)
    df['values'].plot()
    plt.title('Avant traitement')
    plt.subplot(2,1,2)
    data.df['values'].plot()
    plt.title('Après traitement')
    fig.tight_layout(pad=1.0)
    fig_canvas_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
    problems = data.problemsDescription()
    if problems != '':
        sg.popup('Information - molécules détectés', problems)
    return fig_canvas_agg

database = 'data/example.csv'
DATA_PATH = 'data/'

# Aspect de la fenetre
layout = [
    [sg.Text("Interpretation des chromatogrammes")],
    [sg.Text("Chromatogramme"),sg.Input(), sg.FileBrowse(key="chromatogram", file_types=(("Fichiers chromatogram", "*chromatogram.csv"),))],
    [sg.Button('Ok'), sg.Button('Cancel')],
    [sg.Canvas(key='figCanvas', size=(500,500))],
    [sg.Text("Commentaire"), sg.Input(key='label'), sg.Text('Categorie'), sg.Input(key='label_num'), sg.Button('Valider')]
]

# integration de matplotlib tirée de : https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Matplotlib.py

if __name__ == '__main__':
    window = sg.Window('Application d\'IA', layout, element_justification='center', font='16', resizable=True)
    fig_canvas_agg = None
    name = None
    file = open(database, 'a')
    # gestion des input dans la fenetre
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        if event == 'Ok':
            # vérification des noms des deux fichiers
            path,name = split(values['chromatogram'])
            name = name[:-len(tools.CHROM_EXT)]
            if fig_canvas_agg != None:
                fig_canvas_agg.get_tk_widget().pack_forget()
            fig_canvas_agg = readAndPlot(path, name)
        if event == 'Valider':
            label = values['label']
            try :
                label_num = int(values['label_num'])
            except ValueError:
                sg.popup('Erreur', 'La valeur entrée dans le champ "Interpretation" doit être un entier compris entre 0 et 5')
                continue
            if label_num > 5 or label_num < 0:
                sg.popup('Erreur', 'La valeur entrée dans le champ "Interpretation" doit être comprise entre 0 et 5')
                continue
            if name == None:
                sg.popup('Erreur', 'Aucun fichier sélectionné')
                continue
            status = 1 if label == 0 else 0
            line = f"{name},{status},{label},{label_num}\n"
            file.write(line)
            copyfile(join(path,name+tools.CHROM_EXT),join(DATA_PATH, name+tools.CHROM_EXT))
            copyfile(join(path,name+tools.MOL_EXT),join(DATA_PATH, name+tools.MOL_EXT))

    file.close()
    window.close()