import click

import dash
import dash_core_components as dcc
import dash_html_components as html

import flask
import glob
import os

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
            for o in all_occ:
                o_files = [f for f in all_files if o in f]
                audio = [f for f in o_files if '.wav' in f][0]
                img = [f for f in o_files if '.png' in f][0]
                this_d[o] = {'audio': audio, 'plot': img}
            motif_dict[md] = this_d
        return motif_dict, every_fn

    motif_dict, every_fn = get_motif_dict(output_dir)
    plot_name_path = {}
    component_options = sorted(motif_dict.keys(), key=lambda y: int(y.split('_')[1]))
    static_image_route = '/static/'

    num_plots = max([len(m) for m in motif_dict.values()])

    ################
    # Initialize App
    ################
    app = dash.Dash()
    app.title = 'Melodic Exploration - Results Browser'

    components = [
        html.H1(f'Motifs Returned at {output_dir}'),
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in component_options],
            value=min(motif_dict.keys()))
    ]
    images = [
        (html.H3(id=f'caption{i}'), 
         html.Img(id=f'image{i}'),
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
        return [f'Example {i+1}, {x}' for i,x in enumerate(motif_dict[value].keys())]


    @app.callback(
        [dash.dependencies.Output(f'audio{i}', 'src') for i in range(num_plots)]\
        ,[dash.dependencies.Input('image-dropdown', 'value')])
    def update_audio(value):
        audio = [x['audio'] for x in motif_dict[value].values()]
        return [static_image_route + a for a in audio]
        
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