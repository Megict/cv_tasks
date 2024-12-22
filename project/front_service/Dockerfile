FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
RUN pip install fastapi uvicorn
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY ./app_main.py /code/app_main.py

CMD ["uvicorn", "app_main.py:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]