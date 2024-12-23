from dash import Dash, dcc, html, Input, Output, callback
from transformers import  AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
import torch


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
## задавание модели
ckpt = 'facebook/detr-resnet-50'
image_processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)
# пустая картинка до первого инпута
blank = Image.new('RGB', (1000,500), color = (255, 255, 255) ) 
# список всех лейблов модели
all_labels = list(model.config.label2id.keys())
print(f"all model labels:\t{all_labels}")

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
        
        multiple=False
    ),
    html.Div(
        dcc.Dropdown(
            options = all_labels,
            value = all_labels,
            multi=True,
            id = "all-label-values"
        )
    ),
    html.Div(
        html.Button('Select all', id='submit-val', n_clicks=0,
                    style = {'width': '100%'}),
        style = {'width': '100%', 'textAlign': 'center'}
    ),
    dcc.Loading(html.Div(id='output-image-upload')),
])

@callback(
    Output('all-label-values', 'value'),
    Input('submit-val', 'n_clicks')
)
def button_select(_):
    return all_labels

def resize_image(image, base_width = 300):
    wpercent = (base_width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
    return image

def process_image(image, labels_to_highlight):
    found_labels = []
    print("processing image")
    print(type(image))
    # преобразование изображения к нужному формату
    img_dec = base64.decodebytes(image.encode("utf8").split(b";base64,")[1])
    with open("tmp.jpg", "wb") as fp:
        fp.write(img_dec)
    image_good = Image.open('tmp.jpg').convert('RGB')
    print(type(image_good))

    with torch.no_grad():
        inputs = image_processor(images=[image_good], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image_good.size[1], image_good.size[0]]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.40, target_sizes=target_sizes)[0]
        print(results)
        print("inferece complete")
        items = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = score.item()
            label = label.item()
            box = [i.item() for i in box]
            print(f"{model.config.id2label[label]}: {round(score, 3)} at {box}")
            found_labels.append(model.config.id2label[label]) # добавляем найденные метки в список всех найденных на изобрадении меток
            items.append((score, label, box))
            
        print("result constructed")
    # рисуем рамки
    draw = ImageDraw.Draw(image_good)
    for i in range(len(items)):
        box = items[i][2]
        label = items[i][1]
        score = round(items[i][0], 2)
        x1, y1, x2, y2 = tuple(box)
        if model.config.id2label[label] not in labels_to_highlight: # пропускаем метки, которые не требуется отображать
            print(f"label {model.config.id2label[label]} is not to highlight, skipping")
            continue
        
        draw.rectangle((x1, y1, x2, y2), outline="red", width=10)
        # добавление текста
        text = model.config.id2label[label] + "  (" + str(score) + ")"
        font = ImageFont.truetype("Arial.ttf", int(image_good.size[0] / 50))
        bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)
    print("image modified")
    return image_good, np.unique(found_labels)

@callback(
        Output('output-image-upload', 'children'),
        Output('all-label-values', 'options'),
        Input('upload-image', 'contents'),
        Input('all-label-values', 'value'))
def update_output(list_of_contents, labels_to_highlight):
    print("---------------------------------------------------------------")
    print(f"labels to highlight: {labels_to_highlight}")
    if list_of_contents is not None:
        result, labels_on_image = process_image(list_of_contents, labels_to_highlight)
        print(f"labels found: {labels_on_image}")
        return html.Div(html.Img(src = resize_image(result, base_width = 1000)), style = {'width': '100%', 'textAlign': 'center'}), labels_on_image
    else:
        return html.Div(html.Img(src = blank), style = {'width': '100%', 'textAlign': 'center'}), all_labels

if __name__ == '__main__':
    app.run(debug=True, port=8000)