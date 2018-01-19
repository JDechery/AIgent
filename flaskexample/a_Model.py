import pickle
from gensim import models
from Mediumrare import gensim_nlp, db_tools
import numpy as np
import pudb

def ModelIt(blogtext):
    # pudb.set_trace()
    clf_fname = '/home/jdechery/forest_classifier.pkl'
    with open(clf_fname,'rb') as f:
        clf, labelencoder, channeldf = pickle.load(f)

    embed_fname = '/home/jdechery/doc2vec.model'
    embedder = gensim_nlp.DocEmbedder()
    embedder.load_model(fname=embed_fname)

    blog_tokens = db_tools.clean_document(blogtext)
    blog_vec = embedder.model.infer_vector(blog_tokens).reshape(1,-1)

    class_probs = clf.predict_proba(blog_vec)[0].tolist()
    class_names = labelencoder.classes_
    doc_classes = sorted(zip(class_names, class_probs), key=lambda x: x[1], reverse=True)
    return doc_classes
