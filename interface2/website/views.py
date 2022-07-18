from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from .models import Note, VI, Expressions, Delivery, Invoicing, Ordering, Remuneration, Sales, WorkEvents
import random
import numpy as np

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template("home.html")

@views.route('/analyse')
def analyse():
    return render_template("analyse.html")

@views.route('/datas')
def datas():
    return render_template("datas.html")

@views.route('/guide')
def guide():
    return render_template("guide.html")

@views.route('/stat')
def stat():
    return render_template("stat.html")

@views.route('/ajout', methods=['GET','POST'])
def ajout():
    return render_template("ajout.html")
