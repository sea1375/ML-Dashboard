# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os
from app.home import blueprint
from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app import login_manager, graph_json_data
from jinja2 import TemplateNotFound
import time
import test

@blueprint.route('/index')
@login_required
def index():
    return render_template('index.html', segment='index')

@blueprint.route('/<template>')
@login_required
def route_template(template):
    print(template)
    try:
        if template == 'test':
            os.system("python app/home/test.py")
            for index in range(10):
                print('waiting...')
                time.sleep(3)
            return 'ok'
        if not template.endswith( '.html' ):
            template += '.html'
        # Detect the current page
        segment = get_segment( request )
        # Serve the file (if exists) from app/templates/FILE.html
        if template == 'open.html':
            return render_template( template, segment=segment, graph_json_data = graph_json_data , number_of_graphs = len(graph_json_data))
        else:
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
