import joblib
import os
import nltk
import spacy
import re
import spacy.cli
import string
spacy.cli.download("pt_core_news_sm")
import pt_core_news_sm
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY']='whmnnmnyps'

bootstrap = Bootstrap(app)

class InputForm(FlaskForm):
    comment = TextField('Comentário: ', validators=[DataRequired()])


def limpa_cmentario(comentario):
    spacy_pt = pt_core_news_sm.load()
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words("portuguese")
    stopwords.remove('não')
    stopwords.remove('nem')
    comentario = re.sub(r'\s+',' ',comentario)
    comentario = comentario.lower()
    comentario = [word for word in comentario.split() if word not in stopwords and word not in string.punctuation]
    novo_comentario = spacy_pt(" ".join(comentario))
    tokens = [word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in novo_comentario]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form)
    classificacao = ""
    if form.validate_on_submit():
        comentario = form.comment
        classificacao = classifica_comentario(comentario)
    return render_template('index.html', form=form, classificacao=classificacao)

def classifica_comentario(comentario):
    limpa_cmentario(comentario)
    modelfile = os.path.join('app','model','finalized_model.sav')
    vetorizadorfile = os.path.join('app', 'model', 'vetorizador.joblib')
    model = joblib.load(modelname)
    vetorizador = joblib.load(vetorizadorfile)
    vetorizador.transform([comentario])
    return model.predict(x)[0]