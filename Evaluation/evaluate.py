from gensim.models import Word2Vec
import logging
from optparse import OptionParser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np

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


def evaluate(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30, random_state=42)  # Should I use validation data? I am tunning the classifier?
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_test = y_test > 0
    y_pred = y_pred > 0
    acc = accuracy_score(y_test, y_pred)  # Sentiment analysis only positives or how much positive?
    print("The accuracy is {0}".format(acc))
    return df

2
def try_apply(m, x, dim):
    try:
        return m[x]
    except KeyError:
        return np.zeros(dim)


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
                df = pd.read_csv('AFINN.txt', sep='\t', header=None)
                df[0] = df[0].apply(lambda x: try_apply(model, x, kv.vector_size))  # If it don't know a word what I do? n zeroes by now.
                X = np.stack(df[0].values)
                y = df[1].values
                results = evaluate(X, y)