from Mediumrare import gensim_nlp, predictor_model
import pickle

embed_fname = '/home/jdechery/doc2vec.model'
clf_fname = '/home/jdechery/forest_classifier.pkl'
def re_train_models():
    gensim_nlp.main(fname=fname)
    predictor_model.main(clf_fname=clf_fname)

def load_models():
    embedder = gensim_nlp.DocEmbedder()
    embedder.load_model()

    with open(clf_fname, 'rb') as f:
        clf, labelencoder, channeldf = pickle.load(f)

    return embedder, clf, labelencoder, channeldf
