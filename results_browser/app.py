import sys
sys.path.append('../')

import click

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import flask
from flask_caching import Cache
import glob
import os

from src.io import load_json

##########
# Get Data
##########
@click.command()
@click.argument('output_dir')
def main(output_dir):
    def get_motif_dict(output_dir):
        motif_dirs = [d for d in os.listdir(output_dir) if 'motif' in d]

        motif_dict = {}
        every_fn = {}
        for md in motif_dirs:
            all_files = [x for x in os.listdir(os.path.join(output_dir, md)) if '.png' in x or '.wav' in x]
            all_occ = {f.replace('.png', '').replace('.wav', '') for f in all_files}
            this_fn = {o:os.path.join(output_dir, md) for o in all_occ}
            every_fn.update(this_fn)
            this_d = {}
            for o in sorted(all_occ):
                o_files = [f for f in all_files if o in f]
                audio = [f for f in o_files if '.wav' in f][0]
                img = [f for f in o_files if '.png' in f][0]
                this_d[o] = {'audio': audio, 'plot': img}
            motif_dict[md] = this_d
        return motif_dict, every_fn


    ##############
    # Get run data
    ##############
    motif_dict, every_fn = get_motif_dict(output_dir)
    plot_name_path = {}
    component_options = sorted(motif_dict.keys(), key=lambda y: int(y.split('_')[1]))
    static_image_route = '/static_/'

    try:
        metadata = load_json(os.path.join(output_dir, 'metadata.json'))
    except:
        metadata = None
    
    try:
        importances = load_json(os.path.join(output_dir, 'importances.json'))
    except:
        importances = None

    num_plots = max([len(m) for m in motif_dict.values()])

    ################
    # Initialize App
    ################
    app = dash.Dash(external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'simple'
    })

    cache.clear()
    app.title = 'Melodic Exploration - Results Browser'
    graph_layout = go.Layout(
        title="Euclidean Distance to Parent Pattern, Example 1",
        xaxis=dict(
            title="Example Number"
        ),
        yaxis=dict(
            title="Euclidean Distance to Example 1"
        ))
    figs = [{
        "data": [
            go.Bar(
                x=list(range(1,len(importances['0'])+1)),
                y=list(importances['0'].values()),
                marker_color = ['lightslategray' if x!=i else 'crimson' for x in range(len(importances['0']))]
            )],
        "layout": graph_layout} for i in range(len(importances['0']))]

    if metadata:
        s = {'line-height': '15px'}
        metadata = {k.replace('_', ' ').capitalize():v for k,v in metadata.items()}
        metadata_html = html.Div([html.P(f'{k}: {v}', style=s) for k,v in metadata.items()])
    else:
        metadata_html = html.P('No metadata available')
    components = [
        html.H1(f'Melodic Exploration - Results Browser'),
        html.P(f'Output Directory - {output_dir}'),
        html.H2('Run Metadata'),
        metadata_html,
        html.H2('Select Motif from Dropdown Menu'),
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in component_options],
            value=min(motif_dict.keys()))
    ]
    images = [
        (html.H4(id=f'caption{i}'), 
         html.Div([
            html.Div([html.Img(id=f'image{i}')], className='six columns'), 
            html.Div([dcc.Graph(
                        id=f"graph{i}",
                        figure=figs[i]
                    )], className='six columns'),
        ], className="row"),
         html.Div([html.Audio(id=f'audio{i}', controls=True)]),
         html.Hr())\
          for i in range(num_plots)]
    images = [x for y in images for x in y]

    app.layout = html.Div(components + images)

    ############
    ## Callbacks
    ############
    @app.callback(
        [dash.dependencies.Output(f'image{i}', 'src') for i in range(num_plots)]\
        ,[dash.dependencies.Input('image-dropdown', 'value')])
    def update_image_src(value):
        plots = [x['plot'] for x in motif_dict[value].values()]
        return [static_image_route + p for p in plots]


    @app.callback(
        [dash.dependencies.Output(f'caption{i}', 'children') for i in range(num_plots)]\
        ,[dash.dependencies.Input('image-dropdown', 'value')])
    def update_image_text(value):
        to_return = []
        for i,x in enumerate(motif_dict[value].keys()):
            this = x.split('_')[1]
            to_return.append(f'Example {i+1}, {this}')
        return to_return


    @app.callback(
        [dash.dependencies.Output(f'audio{i}', 'src') for i in range(num_plots)]\
        ,[dash.dependencies.Input('image-dropdown', 'value')])
    def update_audio(value):
        audio = [x['audio'] for x in motif_dict[value].values()]
        return [static_image_route + a for a in audio]


    @app.callback(
        [dash.dependencies.Output(f'graph{i}', 'figure') for i in range(num_plots)]\
        ,[dash.dependencies.Input('image-dropdown', 'value')])
    def update_graph(value):
        this_motif = str(value.split('_')[1])
        these_importances = importances[this_motif]
        figs = [{
            "data": [
                go.Bar(
                    x=list(range(1,len(these_importances)+1)),
                    y=list(these_importances.values()),
                    marker_color = ['lightslategray' if x!=i else 'crimson' for x in range(len(these_importances))]
                )],
            "layout": graph_layout} for i in range(len(these_importances))]
        return figs

        
    # Add a static image route that serves images from desktop
    # Be *very* careful here - you don't want to serve arbitrary files
    # from your computer or server
    @app.server.route(f'{static_image_route}<image_path>.png')
    def serve_image(image_path):
        image_name = '{}.png'.format(image_path)
        image_dir = every_fn[image_path]
        return flask.send_from_directory(image_dir, image_name)

    @app.server.route(f'{static_image_route}<audio_path>.wav')
    def serve_audio(audio_path):
        audio_name = '{}.wav'.format(audio_path)
        audio_dir = every_fn[audio_path]
        return flask.send_from_directory(audio_dir, audio_name)

    app.run_server(debug=True)

if __name__ == '__main__':
    main()