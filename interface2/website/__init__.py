from gzip import READ
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
import csv


db = SQLAlchemy()
DB_NAME = "database.db"


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User, Note, VI, Expressions, Delivery, Invoicing, Ordering, Remuneration, Sales, WorkEvents

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app


def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        add_vi(app)
        add_voc(app)
        print('Created Database!')
          
          
def add_vi(app):
    with open('data/IrregularVerbs.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import VI
        for row in csv_reader :
            new_vi = VI(base_verbal = row["Base Verbale"], preterit = row["Preterit"], past_participle = row["Past participle"], trad = row["Traduction"])
            with app.app_context():
                db.session.add(new_vi)
                db.session.commit()


def add_voc(app):
    with open('data/VocaExpressions.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Expressions
        for row in csv_reader :
            new_voc = Expressions(anglais = row["Version Anglaise"], francais = row["Expression Française"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()

    with open('data/Delivery.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Delivery
        for row in csv_reader :
            new_voc = Delivery(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()
    
    with open('data/Invoicing.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Invoicing
        for row in csv_reader :
            new_voc = Invoicing(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()
    
    with open('data/Ordering.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Ordering
        for row in csv_reader :
            new_voc = Ordering(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()
    
    with open('data/RemunerationContracts.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Remuneration
        for row in csv_reader :
            new_voc = Remuneration(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()
    
    with open('data/Sales.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import Sales
        for row in csv_reader :
            new_voc = Sales(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()
    
    with open('data/WorkEvents.csv', mode='r', encoding='utf-8') as csv_file :
        csv_reader = csv.DictReader(csv_file)
        
        from .models import WorkEvents
        for row in csv_reader :
            new_voc = WorkEvents(anglais = row["Anglais"], francais = row["Français"])
            with app.app_context():
                db.session.add(new_voc)
                db.session.commit()