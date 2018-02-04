from Mediumrare import db_tools, gensim_nlp, predictor_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
clf, *_ = predictor_model.load_classifier()
X, y, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=25)

# %% compute accuracy
def predicted_in_topN(y_hat, y_test, N):
    argmax_labels = np.argsort(y_hat, axis=1)
    y_test = np.reshape(y_test, (-1, 1))
    labels = np.reshape(argmax_labels[:,-N:], (-1, N))
    truepositive = np.any(np.equal(labels, np.repeat(y_test, N, axis=1)), axis=1)
    return truepositive

kfold = 5
topN = 5#range(1,21)
cv = StratifiedKFold(n_splits=kfold)

accuracy = np.zeros((kfold, len(labelencoder.classes_)))
confmat = []
for splitid, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_test)
    confmat.append(confusion_matrix(y_test, np.argsort(y_hat,axis=1)[:,0]))
    truepositive = predicted_in_topN(y_hat, y_test, topN)
    for lbl in np.unique(y_test):
        accuracy[splitid, lbl] = np.mean(truepositive[(y_test==lbl).flatten()])

mu_acc = accuracy.mean(axis=0)
mu_conf = np.asarray(confmat).mean(axis=0)
# %% get tag df
remove_bad_chars = lambda word: re.sub('[{}"]', '', word)
tags = channeldf.tags
tags = [remove_bad_chars(row) for row in tags]

countvectorizer = CountVectorizer(input='content', strip_accents='unicode', min_df=0)
tag_counts = countvectorizer.fit_transform(tags)
voc = countvectorizer.vocabulary_

tagdf = pd.DataFrame(data=tag_counts.todense(), columns=voc)
tagdf['channel'] = channeldf['channel'].tolist()

# %% get tag distance between channels
channel_tagvec = tagdf.groupby('channel').mean()
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances, cosine_distances
tagdist = manhattan_distances(channel_tagvec)
# tagdist = cosine_distances(channel_tagvec)

matplotlib.rcParams.update({'font.size': 18})
# %%
# fig, ax = plt.subplots()
# plt.scatter(tagdist.flatten(),mu_conf.flatten(),4)
# plt.show()
tagdf['channel'].value_counts().plot(kind='bar', color='k',xticks=None, figsize=(10,6))
plt.xticks([])
plt.xlabel('channels')
plt.ylabel('n articles')
plt.show()

# %% acc vs tag similarity
yy = mu_acc
# xx = np.mean(tagdist,axis=0)
xx = np.percentile(tagdist,.5,axis=0)
r = np.corrcoef(xx,yy)
fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(xx,yy,100)
plt.title('r='+str(r[1,0]))
plt.xlabel('median distance')
plt.ylabel('mean accuracy')
plt.show()

# %% acc vs n sample
yy = mu_acc
xx = tagdf['channel'].value_counts()
r = np.corrcoef(xx,yy)
fig, ax = plt.subplots(figsize=(10,6))
plt.scatter(xx,yy, 100)
plt.xlabel('n articles')
plt.ylabel('mean accuracy')
plt.title('r='+str(r[1,0]))
plt.show()
