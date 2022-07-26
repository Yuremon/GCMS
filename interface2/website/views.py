from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
import random
import os, calendar, time
import numpy as np
from fpdf import FPDF
from os.path import join, exists, split
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from src import load_and_predict as lp
from src import tools

ALLOWED_EXTENSIONS = {'csv'}

layout = [
    [sg.Text("Interpretation des chromatogrammes")],
    [sg.Text("Chromatogramme"),sg.Input(), sg.FileBrowse(key="chromatogram", file_types=(("Fichiers chromatogram", "*chromatogram.csv"),))],
    [sg.Button('Ok'), sg.Button('Quitter')],
    [sg.Canvas(key='figCanvas', size=(500,500))],
    [sg.Text("Commentaire"), sg.Input(key='label'), sg.Text('Categorie'), sg.Input(key='label_num'), sg.Button('Valider')]
]
window = sg.Window('Application d\'IA', layout, element_justification='center', font='16', resizable=True)
fig_canvas_agg = None

views = Blueprint('views', __name__)

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

def createPdf(pdf_name,name, df, data, ruleBasedText, IaText):
    """Creation du pdf"""
    basePath = os.path.dirname(__file__)
    upload_path = os.path.join(basePath, 'static/assets/img')
    TEMP_PATH = os.path.abspath(upload_path)
    # cration remplissage du pdf
    pdf = PDF()
    pdf.add_page()
    pdf.insert_title(name)
    pdf.insert_charts(TEMP_PATH+'/plotdf.png',0)
    pdf.insert_text("Résultat de l'intelligence artificielle : \n" + IaText,0)
    pdf.insert_text("Détection de pics : \n" + ruleBasedText,1)
    # export du pdf
    pdf.output(pdf_name,'F').encode('latin-1')

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
    basePath = os.path.dirname(__file__)
    upload_path = os.path.join(basePath, 'static/assets/img')
    TEMP_PATH = os.path.abspath(upload_path)
    """Creation du graphique pour affichage dans la fenetre et enregistrement pour l'utiliser dans le pdf"""
    fig = plt.figure()
    plt.subplot(2,1,1)
    df['values'].plot()
    plt.title('Avant traitement')
    plt.subplot(2,1,2)
    data.df['values'].plot()
    plt.title('Après traitement')
    fig.tight_layout(pad=1.0)
    plt.savefig(fname=TEMP_PATH+'/plotdf.png')
    problems = data.problemsDescription()
    #if problems != '':
        #sg.popup('Information - molécules détectés', problems)
    return 


def draw_figure(canvas, figure):
    """Integration de matplotlib dans la fenetre de PySimpleGUI"""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.show()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas_agg.get_tk_widget().size = (400,400)
    return figure_canvas_agg


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS




@views.route('/')
def home():
    return render_template("home.html")

@views.route('/analyse', methods=['GET','POST'])
def analyse():
    fig_canvas_agg = None
    res2=0
    rep="Exemple de ce que vous devriez voir après avoir effectuer l'analyse"
    res1=0
    if request.method == 'POST':
        
        #Récupération du fichier -chromatogram.csv
        file = request.files['file']
        if file is None :
            res2 = "Le fichier est de type None"
            return render_template("analyse.html", user=rep, res=res)
        if file.filename == '':
            flash('Pas de fichier sélectionné')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Le fichier sélectionné n'est pas au format csv")
        if file and allowed_file(file.filename):
            nowtime = calendar.timegm(time.gmtime())
            basePath = os.path.dirname(__file__)
            upload_path = os.path.join(basePath, 'savings/')
            upload_path = os.path.abspath(upload_path)
            file.save(upload_path+"/"+str(nowtime)+ "-chromatogram.csv")
            #C'était la parite enregistrement du fichier déposé sur le serveur dans un dossier spécialement conçu pour ça
            #Maintenant on cherche à connecter le script load_and_predict
            res2 = file.filename

        #Récupération du fichier -ms.csv
        file1 = request.files['file1']
        if file1 is None :
            res1 = "Le fichier est de type None"
            return render_template("analyse.html", rep=rep)
        if file1.filename == '':
            flash('Pas de fichier sélectionné')
            return redirect(request.url)
        if not allowed_file(file1.filename):
            flash("Le fichier sélectionné n'est pas au format csv")
        if file1 and allowed_file(file1.filename):
            nowtime = calendar.timegm(time.gmtime())
            basePath = os.path.dirname(__file__)
            upload_path = os.path.join(basePath, 'savings/')
            upload_path = os.path.abspath(upload_path)
            file1.save(upload_path+"/"+str(nowtime)+ "-ms.csv")
            #C'était la parite enregistrement du fichier déposé sur le serveur dans un dossier 
            #Spécialement conçu pour cela

            #Maintenant on cherche à connecter le script load_and_predict
            res1 = file1.filename  
            
            
            
        rep = lp.calcul(upload_path, str(nowtime))


        #Boucle pour les graphiques + pdf
        if file and file1 and allowed_file(file1.filename) and allowed_file(file.filename):
            path = upload_path
            name = str(nowtime)
            df,data = read(path, name)
            if df is not None and data is not None:
                fig_canvas_agg = plot(df,data)
                pdf_name = join(path,name)+'.pdf'
                createPdf(pdf_name,name,df,data, data.problemsDescription() if data.problemsDescription() != '' else "aucun problème détecté", str(rep))
                # ouverture du pdf
                #os.startfile(pdf_name)
                return render_template("analyse.html", rep=rep)
            window.close()    

    return render_template("analyse.html", rep=rep)

@views.route('/guide')
def guide():
    return render_template("guide.html")

@views.route('/stat')
def stat():
    return render_template("stat.html")

