from dash import Dash, html, dcc, callback, Output, Input
from ultralyticsplus import YOLO, render_result
import dash_mantine_components as dmc
import numpy as np
import json

app = Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.Button('update', id='do-update'),
        dmc.RangeSlider(value=[0, 200],
                        step = 1,id = 'act-range',style={'width': '25vw', 
                                                    'margin-left' : '2vw' }),
        dcc.Dropdown(id='highlight-node',
                     placeholder="Выберите вершину", style={'width': '20vw', 'margin-left' : '2vw'}),
    ], style={'margin-top' : '2vh', 'display': 'flex'}),

    dcc.Graph(id='graph-content', style={
                            'height': '80vh', 'width': '80vw', 'margin-top': '0vh', 'margin-left': '5vw'})
])

model = YOLO('keremberke/yolov8m-pothole-segmentation')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

#=====================


@callback(
    Output('graph-content', 'figure'),
    Input('do-update', 'value'),
    Input('act-range', 'value'),
    Input('highlight-node', 'value'),
    Input('center-subgr', 'value'),
    Input('depth', 'value'),
    Input('display-mode', 'value'),
    Input('links', 'value')
)
def update_graph(_, time_range, highlight_node, center_node, depth, edges_display, link_top_p):

    print("construction attempt")
    if edges_display == "links":
        # links_ = display_links
        display_edges_ = False
    else:
        links_ = []
        display_edges_ = True

    if center_node is None:
        #center_node = list(my_graph.G.nodes)[0]
        disp_gr = my_graph.G
    else:
        disp_gr = gran.select_subgraph([center_node], depth)

    print("subgraph:\t done")
    pos = nx.kamada_kawai_layout(nx.Graph(disp_gr))
    #pos = my_graph.pos
    print("layout:\t done")

    fig = draw_graph(disp_gr, pos,
                     links = links_,
                     link_color_key = {"freq_link" : "black", "dist_link" : "orange", "sem_link" : "magenta"},
                     display_edges = display_edges_,
                     color_key = "color", 
                     edge_limit_key_name = None, #'locations', 
                     edge_limit_key_values = time_range, 
                     highlight_around= [highlight_node] if highlight_node != None else [])
    print("done")
    return fig

if __name__ == '__main__':
    app.run(debug=True)