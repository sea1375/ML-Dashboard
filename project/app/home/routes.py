# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from app.home import blueprint
from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
import time

@blueprint.route('/index')
@login_required
def index():
    return render_template('index.html', segment='index')

@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if template == 'train':
            file_name = request.args.get('file_name')
            train_prefix = request.args.get('train_prefix')
            data_prefix = request.args.get('data_prefix')
            model = request.args.get('model')
            max_degree = request.args.get('max_degree')
            epochs = request.args.get('epochs')
            train_chunks = request.args.get('train_chunks')
            train_percentage = request.args.get('train_percentage')
            validate_batch_size = request.args.get('validate_batch_size')
            nodes_max = request.args.get('nodes_max')

            command = 'python -m'
            command = command + ' ' + file_name
            command = command + ' --train_prefix ' + train_prefix
            command = command + ' --data_prefix ' + data_prefix
            command = command + ' --model ' + model
            command = command + ' --max_degree ' + max_degree
            command = command + ' --epochs ' + epochs
            command = command + ' --train_chunks ' + train_chunks
            command = command + ' --train_percentage ' + train_percentage
            command = command + ' --validate_batch_size ' + validate_batch_size
            command = command + ' --nodes_max ' + nodes_max

            print(command)
            os.system(command)
            return 'success'
            
        if not template.endswith( '.html' ):
            template += '.html'
        # Detect the current page
        segment = get_segment( request )
        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( template, segment=segment )

        # if segment == 'new':
            # data={'projects': 'C:\\Users\\eaminbo\\Documents\\production\\ETS\\code\\Visualizer\\project\\app\\projects'}			

            # return render_template( template, data=data, segment=segment)
        # else:
            # return render_template( template, segment=segment )		

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

# Helper - Extract current page name from request 
def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'
            # segment = 'new'
        return segment    

    except:
        return None  
