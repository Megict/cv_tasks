FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./app_main.py /code/app_main.py
COPY ./Arial.ttf /code/Arial.ttf

CMD gunicorn --bind 0.0.0.0:8001 app_main:server
