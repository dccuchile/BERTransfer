import gensim 
import logging
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file="../Data/training_data_test.txt"

def read_input(input_file):
    logging.info("Reading file {0}...".format(input_file))

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                logging.info("Read {0} articles".format(i))
            yield gensim.utils.simple_preprocess(line)

documents = list(read_input(data_file))
logging.info("Done reading data file")
model = gensim.models.Word2Vec(documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)

with open('word2vec_model.bin', 'wb') as w2v_file:
    pickle.dump(model, w2v_file)