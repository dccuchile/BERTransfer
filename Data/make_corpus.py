from gensim.corpora import TextCorpus, MmCorpus
from gensim.test.utils import datapath
from gensim import utils
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, handlers=[
    logging.FileHandler("/home/ctamblay/Desktop/BERTransfer/Data/data.log"),
    logging.StreamHandler()
])

training_path = "/home/ctamblay/Desktop/BERTransfer/Data/training_data.txt"


class Corpus(TextCorpus):
    stopwords = set('for a of the and to in on'.split())

    def get_texts(self):
        for i, doc in self.getstream():
            if i % 10000 == 0:
                logging.info("Read {0} articles".format(i))
            yield [word for word in utils.to_unicode(doc).split() if word not in self.stopwords]

    def __len__(self):
        self.length = sum(1 for _ in self.get_texts())
        return self.length


corpus = Corpus(datapath(training_path))
# MmCorpus.serialize('test.mm', corpus)
