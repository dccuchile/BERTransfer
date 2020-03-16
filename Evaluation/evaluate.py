from gensim.models import Word2Vec
import logging
from optparse import OptionParser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models.keyedvectors import KeyedVectors

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-m", "--model", dest="model",
                  help="Model of the embedding, possible values are: w2v, glove, bert and xlnet.",
                  default=None)

parser.add_option("-i", "--initial", dest="initial",
                  help="Initial domain, possible values are: word or sentence.",
                  default=None)

parser.add_option("-e", "--end", dest="end",
                  help="End domain, possible values are: word or sentence.",
                  default=None)

parser.add_option("-o", "--output", dest="output",
                  help="Path where to save results.",
                  default="output.csv")

#  Path of the datasets
AFINN = '/home/ctamblay/Desktop/BERTransfer/Evaluation/data/AFINN.txt'
IMDB = 'imdb.txt' # TODO Check the path


def evaluate(X1, y1, X2, y2):
    X_train, _, y_train, _ = train_test_split(X1, y1, shuffle=True, test_size=0.30, random_state=42)  # Should I use validation data? I am tunning the classifier?
    _, X_test, _, y_test = train_test_split(X2, y2, shuffle=True, test_size=0.30, random_state=42)  # Should I concatenate the test corpus if the domains are different?
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_test = y_test > 0
    y_pred = y_pred > 0
    acc = accuracy_score(y_test, y_pred)  # Sentiment analysis only positives or how much positive?
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("The accuracy is {0}".format(acc))
    print("The precision is {0}".format(prec))
    print("The recall is {0}".format(rec))
    print("The f1 score is {0}".format(f1))


def try_apply(m, x, dim):
    try:
        return m[x]
    except KeyError:
        return np.zeros(dim)


def apply_then_avg(m, x, dim):
    result = 0
    i = 0
    for word in x:
        i += 1
        try:
            result += m[word]
        except KeyError:
            pass  # Avg of known words? Replace unknown by 0?
    return np.divide(result, i)


if __name__ == "__main__":
    (options, args) = parser.parse_args()

    # Load embeddings
    fname = options.filename
    mname = options.model
    iname = options.initial
    ename = options.end

    if not fname:
        raise Exception("Please input a file with the embedding.")
    else:
        if not mname:
            raise Exception("Please input the model to evaluate.")

        if mname == 'w2v':
            if (not iname) or (not ename):
                raise Exception("Please input initial domain and end domain")
            model = Word2Vec.load(fname)
            kv = model.wv

            if iname == 'word' and ename == 'word':
                df = pd.read_csv(AFINN, sep='\t', header=None)
                df[0] = df[0].apply(lambda x: try_apply(model, x, kv.vector_size))  # If it don't know a word what I do? n zeroes by now.
                X = np.stack(df[0].values)
                y = df[1].values
                evaluate(X, y, X, y)  # Domains are the same

            elif iname == 'word' and ename == 'sentence':
                df1 = pd.read_csv(AFINN, sep='\t', header=None)
                df1[0] = df1[0].apply(lambda x: try_apply(model, x, kv.vector_size))  # If it don't know a word what I do? n zeroes by now.
                df2 = pd.read_csv(IMDB, sep='\t', header=None)
                df2[0] = df2[0].apply(lambda x: x) # TODO Calcular la aplicacion del modelo sobre una oracion
                df2[1] = 0 # TODO Pasar pos y neg a -1 y 1
                X1 = np.stack(df1[0].values)
                y1 = df1[1].values
                X2 = np.stack(df1[0].values)
                y2 = df1[1].values
                evaluate(X1, y1, X2, y2)

        elif mname == 'glove':
            if (not iname) or (not ename):
                raise Exception("Please input initial domain and end domain")
            model = KeyedVectors.load_word2vec_format(fname, binary=False)
            if iname == 'word' and ename == 'word':
                df = pd.read_csv(AFINN, sep='\t', header=None)
                df[0] = df[0].apply(lambda x: try_apply(model, x, model.vector_size))  # If it don't know a word what I do? n zeroes by now.
                X = np.stack(df[0].values)
                y = df[1].values
                evaluate(X, y, X, y)  # Domains are the same