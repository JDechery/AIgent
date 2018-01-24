import pickle
from gensim import models
from Mediumrare import gensim_nlp, db_tools, predictor_model, channel_metadata
import numpy as np

def ModelIt(blogtext, topN=8):
    embed_fname = '/home/jdechery/doc2vec.model'
    embedder = gensim_nlp.DocEmbedder()
    embedder.load_model(fname=embed_fname)

    clf_fname = '/home/jdechery/forest_classifier.pkl'
    clf,_,_ = predictor_model.load_classifier()
    _,_,labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=15)

    blog_tokens = db_tools.clean_document(blogtext)
    blog_vec = embedder.model.infer_vector(blog_tokens).reshape(1,-1)

    class_probs = clf.predict_proba(blog_vec)[0].tolist()
    top_idx = np.argsort(class_probs)[:topN]
    top_channels = labelencoder.classes_[top_idx]

    channel_data_funs = (channel_metadata.most_recent_pubs,
                         channel_metadata.most_common_tags,
                         channel_metadata.mean_channel_claps)
    top_channel_data = [[data_fun(channeldf, chan) for data_fun in channel_data_funs] for chan in top_channels]
    # top_tags = [channel_metadata]

    # doc_classes = sorted(zip(class_names, class_probs), key=lambda x: x[1], reverse=True)
    return top_channel_data
