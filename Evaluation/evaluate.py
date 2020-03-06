from gensim.models import Word2Vec
import logging
from optparse import OptionParser
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="Path to the file with embedding. If relative will load from data directory.",
                  default=None)

parser.add_option("-m", "--model", dest="model",
                  help="Model of the embedding, possible values are: word2vec, glove, bert and xlnet.",
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


def evaluate(df):
    # Scikit
    return df


if __name__ == "__main__":
    (options, args) = parser.parse_args()

    # Load embeddings
    fname = options.filename
    mname = options.model
    iname = options.initial

    if not fname:
        raise Exception("Please input a file with the embedding.")
    else:
        if not mname:
            raise Exception("Please input the model to evaluate.")
        if mname == 'word2vec':
            model = Word2Vec.load(fname)
            if iname == 'word':
                # TODO Traducir AFINN a Vector,Value para luego llamar una funcion standar de evaluacion
                df = pd.read_csv('AFINN.txt', sep='\t', header=None)
                df[0] = df[0].apply(lambda x: model[x])
                results = evaluate(df)
