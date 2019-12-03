import pickle

# Step 2
with open('word2vec_model.bin', 'rb') as w2v_file:
    model = pickle.load(w2v_file)


