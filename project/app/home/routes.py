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
    print(template)
    try:
        if template == 'train':
            os.system("python -m app.graphsage.supervised_train --train_prefix app/base/static/graphs --data_prefix adpcicd --model gcn --max_degree 10 --epochs 30 --train_chunks True --train_percentage 0.8 --validate_batch_size -1 --nodes_max 1000")
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
