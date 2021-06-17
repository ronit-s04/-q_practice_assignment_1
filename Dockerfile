FROM python:3.9

WORKDIR /q-assignment-practice

COPY requirement.txt .

RUN pip install -r requirement.txt

COPY ./Api ./app

CMD ["python", "./Api/main.py"]

