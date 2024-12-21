from dash import Dash, dcc, html, Input, Output, State, callback
import datetime
from ultralyticsplus import YOLO, render_result
import base64 
import cv2


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
model = YOLO('keremberke/yolov8m-pothole-segmentation')
# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])

def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

def process_image(image):
    print("processing image")
    print(type(image))
    img_dec = base64.decodebytes(image.encode("utf8").split(b";base64,")[1])
    with open("tmp.jpg", "wb") as fp:
        fp.write(img_dec)
    image_good = cv2.imread('tmp.jpg')
    print(type(image_good))
    results = model.predict(image_good)
    # print(results[0].boxes)
    # print(results[0].masks)
    print("inferece complete")
    render = render_result(model=model, image=image_good, result=results[0])
    print(type(render))
    # cv2.imwrite('tmp.jpg', image_good)
    # with open("tmp.jpg", "rb") as fp:
    #     render_dec = fp.read()
    #     print(type(render_dec))
    # encoded = base64.b64encode(image).decode('utf-8')
    # encoded = 'data:image/png;base64,' + encoded
    # print(type(encoded))
    return render

@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            process_image(c) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        print(children)
        return html.Img(src = children[0])

if __name__ == '__main__':
    app.run(debug=True)