import uvicorn
from fastapi import FastAPI, Form
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import json
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
new_model = load_model('final_nlp.model')
nltk.download('stopwords')
from text_input import  text_input
# import contractions


def text_tokenizer(text):
    with open('tokenizer_details.json', 'r') as json_file:
        data = json.load(json_file)
    tokenize = tokenizer_from_json(data)
    sqe = np.asarray(tokenize.texts_to_sequences(text))
    #new_list = []
    #for i in range(len(sqe)):
        #new_list.append(sqe[i][0])
    #sqe.append(new_list)
    sqe = pad_sequences(sqe.reshape(-1,sqe.shape[0]), maxlen=50, padding="post", truncating="post")
    return sqe

def expand_contractions(text):
    # text = [contractions.fix(word) for word in text.split()]
    text = ' '.join(text)
    return text

def pre_process_data(text):
    ps = PorterStemmer()
    # text = expand_contractions(text)
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    #text = ' '.join(text)
    return text


app = FastAPI()


@app.get('/')
def index():
    return {'Meassage':'Hello World'}


@app.post('/predict')
def predict_paraphrase(data : text_input):
    data = data.dict()
    data_1 = pre_process_data(data['text_1'])
    input_1 = text_tokenizer(data_1)
    data_2 = pre_process_data(data['text_2'])
    input_2 = text_tokenizer(data_2)
    pre = new_model.predict([input_1.reshape(-1, 50), input_2.reshape(-1, 50)])
    print(pre)
    class_value = np.argmax(pre, axis=1)
    if pre[0] < 0.5:
        return {'Msg': 'Not a paraphrase'}
    else:
        return {'Msg': 'A paraphrase'}





if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

