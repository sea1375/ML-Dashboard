# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

"""
$ (Unix/Mac) export FLASK_APP=run.py
$ (Windows) set FLASK_APP=run.py
			set FLASK_ENV=development
$ (Powershell) $env:FLASK_APP = ".\run.py"

flask run --host=0.0.0.0 --port=5000
"""
import os
from flask import Flask, url_for, json
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from importlib import import_module
from logging import basicConfig, DEBUG, getLogger, StreamHandler

db = SQLAlchemy()
login_manager = LoginManager()
# graph_json_data  = []

def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)

def register_blueprints(app):
    for module_name in ('base', 'home'):
        module = import_module('app.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)

def configure_database(app):

    @app.before_first_request
    def initialize_database():
        db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()

# def read_chunks():
#     SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
#     NUMBER_OF_GRAPH = 1000
#     PADDING_SIZE = 3
#     try:
#         for index in range(NUMBER_OF_GRAPH):
#             url_string = str(index)
#             url_string = 'chunk' + '0'*(PADDING_SIZE - len(url_string)) + url_string
#             json_url = os.path.join(SITE_ROOT, 'graphs', url_string, 'adpcicd-G.json')

#             if os.path.exists(json_url) == False:
#                 break

#             data = json.load(open(json_url))
#             graph_json_data.append(data)
#     except:
#         print('except')

def create_app(config):
    app = Flask(__name__, static_folder='base/static')
    app.config.from_object(config)
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)
    # read_chunks()
    return app
