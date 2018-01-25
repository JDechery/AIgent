from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from Mediumrare import db_tools, gensim_nlp, predictor_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pickle
from itertools import combinations
from collections import OrderedDict
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.stats import entropy

# %% oob error over n_trees
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
min_blogs = 15
X, y, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=min_blogs)
n_docs = X.shape[0]
word_cols = ['dim%d'%x for x in range(X.shape[1])]
n_channel = len(labelencoder.classes_)
# gridclf, labelencoder, channeldf = predictor_model.load_classifier()
# n_docs = channeldf.shape[0]
# X = channeldf[word_cols].as_matrix()
# y = labelencoder.transform(channeldf['channel'])
# %% oob_score over n_estimators
# clf = RandomForestClassifier(warm_start=True, oob_score=True,
                              # n_jobs=-1, max_features='sqrt')
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               n_jobs=-1)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               n_jobs=-1)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               n_jobs=-1))
]
n_trees = range(50, 751, 100)
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
# error_rate = []
for label, clf in ensemble_clfs:
    for trees in n_trees:
        clf.set_params(n_estimators=trees)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((trees, oob_error))
# %% plot
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label.split(',')[1])

plt.xlim(n_trees[0], n_trees[-1])
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(frameon=False, fontsize=13)
plt.title('Forest hyperparameter errors')
plt.tight_layout()
# plt.savefig('/home/jdechery/code/insight/Mediumrare/crossval_forest_hyperparam.png', dpi=300)
plt.show()
# %% get accuracy of top N labels (not only top 1 label)
def predicted_in_topN(clf, Xtest, ytest, N):
    y_hat = clf.predict_proba(X_test)
    argmax_labels = np.argsort(y_hat, axis=1)
    # for N in topN:
    ytest = np.reshape(y_test, (-1, 1))
    labels = np.reshape(argmax_labels[:,-N:], (-1, N))
    truepositive = np.any(np.equal(labels, np.repeat(ytest, N, axis=1)), axis=1)
    return truepositive

kfold = 3
topN = range(1,21)
cv = StratifiedKFold(n_splits=kfold)
n_trees = 500
# clf = gridclf.best_estimator_
# clf = KNeighborsClassifier(p=2, n_jobs=-1)
clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, max_features='sqrt')
accuracy = np.zeros((kfold, len(topN)))
for splitid, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_test)
    argmax_labels = np.argsort(y_hat, axis=1)
    for N in topN:
        ytest = np.reshape(y_test, (-1, 1))
        labels = np.reshape(argmax_labels[:,-N:], (-1, N))
        truepositive = np.any(np.equal(labels, np.repeat(ytest, N, axis=1)), axis=1)
        accuracy[splitid, N-1] = np.mean(truepositive)
# %% plot acc
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8,6))
plt.errorbar(x=topN, y=np.mean(accuracy,axis=0), yerr=np.std(accuracy,axis=0), linewidth=2.)

psuccess = np.asarray(topN)/n_channel
pdiff = np.mean(accuracy,axis=0)-psuccess
bestN = np.argmax(pdiff)
plt.plot([topN[bestN], topN[bestN]], [0, pdiff[bestN]], linestyle='--',
         color='r', linewidth=2., label='_nolegend_')
plt.scatter([topN[bestN]], [pdiff[bestN]], s=200, marker='*', c='r', label='_nolegend_')

plt.plot(topN, psuccess, color='k', linewidth=2.)
plt.plot(topN, pdiff, linestyle='--', linewidth=2, color=[.6, .6, .6])
plt.xlabel('including top N choices')
plt.ylabel('mean accuracy')
plt.ylim((0, 1))
plt.xlim((1, topN[-1]))
plt.xticks(range(2,22,2))
plt.legend(['random', 'difference', 'classifier accuracy'], frameon=False)
plt.title('Publisher predictions from\npublishers with over {:d} posts (n={:d})'.format(min_blogs, n_channel))
plt.tight_layout()
plt.savefig('/home/jdechery/code/insight/Mediumrare/crossval_topNaccuracy.png', dpi=300)
plt.show()
# %% confusion matrix
kfold = 3
cv = StratifiedKFold(n_splits=kfold)
n_trees = 500
clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_trees, max_features='sqrt')
conf = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    cmat = confusion_matrix(y_test, y_hat)
    _, n_test_labels = np.unique(y_test, return_counts=True)
    cmat = cmat / np.reshape(np.asarray(n_test_labels), (n_channel, -1))
    conf.append(cmat)
# %%
sns.heatmap(conf[0], vmin=0, vmax=1)
plt.show()
# %% plot error rate per category vs label diversity
# no effect
kfold = 3
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
cv = StratifiedKFold(n_splits=kfold)
plotdata = np.zeros((len(np.unique(y)), kfold, 3))
for splitid, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]
    tag_test = channeldf.loc[channeldf.index[test_idx], 'tags']
    # X_train, X_test, y_train, y_test, _, tag_test = X, y, channeldf['tags'])
    clf.fit(X_train, y_train)
    # y_hat = clf.predict(X_test)
    truepos = predicted_in_topN(clf, X_test, y_test, 1)
    remove_bad_chars = lambda word: re.sub('[{}"]', '', word)

    tags = [remove_bad_chars(row) for row in tag_test]

    countvectorizer = CountVectorizer(input='content', strip_accents='unicode', min_df=0)
    tag_counts = countvectorizer.fit_transform(tags)
    voc = countvectorizer.vocabulary_

    tagdf = pd.DataFrame(data=tag_counts.todense(), columns=voc)
    tagdf['label'] = y_test
    tagdf['result'] = truepos
# tagdf['ntags'] = tag_counts.sum(axis=1)
#
# def entropy_from_values(pdseries):
    # return entropy(pdseries.value_counts()/len(pdseries))
    aggfun = voc.copy()
    for key, _ in aggfun.items():
        aggfun[key] = 'sum'
    aggfun['result'] = 'mean'
    # aggfun['count'] = lambda x: len(x)
    bylabel = tagdf.groupby('label').agg(aggfun)
    bylabel['tagent'] = bylabel[list(voc.keys())].apply(lambda x: entropy(x / x.sum()), axis=1)
    # bylabel[list(voc.keys())] = TfidfTransformer().fit_transform(bylabel[list(voc.keys())]).todense()
    # bylabel['tagtfidf'] = bylabel[list(voc.keys())].apply(lambda x: x[x!=0].mean(), axis=1)
    bylabel['count'] = tagdf['label'].value_counts(sort=False)

    plotdata[:, splitid, 0] = bylabel['result']
    # plotdata[:, splitid, 1] = bylabel['tagtfidf']
    plotdata[:, splitid, 1] = bylabel['tagent']
    plotdata[:, splitid, 2] = bylabel['count']
# bylabel['tagent'] = bylabel[list(voc.keys())].mean(axis=1)

# bylabel.plot.scatter(y='tagent', x='result')
# plt.show()
# %%
plt.errorbar(y=plotdata[:,:,0].mean(axis=1).flatten(),
             x=plotdata[:,:,1].mean(axis=1).flatten(),
             yerr=plotdata[:,:,0].std(axis=1).flatten(),
             xerr=plotdata[:,:,1].std(axis=1).flatten(),
             linestyle='', marker='o')
plt.show()


# %% publisher doc within/between similarity
# dvs = embedder.model.docvecs
doc_similarity = np.zeros((n_channel, n_channel))
ndoc_pairs = np.zeros((n_channel, n_channel))
for doc_ii, doc_jj in combinations(range(n_docs), 2):
    # if channeldf['channel'].iloc[doc_ii] == channeldf['channel'].iloc[doc_jj]:
    #     key = 'within'
    # else:
    #     key = 'between'
    pairids = labelencoder.transform(channeldf['channel'].iloc[[doc_ii, doc_jj]])
    sim = dvs.similarity(doc_ii, doc_jj)
    doc_similarity[pairids[0], pairids[1]] += sim
    ndoc_pairs[pairids[0], pairids[1]] += 1

sim_norm = (doc_similarity+doc_similarity.T) / (ndoc_pairs+ndoc_pairs.T)
# %%
# sim_norm[np.where(np.isnan(sim_norm))] = np.nanmean(sim_norm)
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(sim_norm, ax=ax)#, vmin=0, vmax=1)
# plt.sca(axs[0])
plt.axis('equal')
# sns.heatmap(sim_norm/np.diag(sim_norm), ax=axs[1])
# plt.sca(axs[1])
# plt.axis('equal')
plt.xticks(np.arange(0.5, n_channel+0.5), labelencoder.classes_, rotation='vertical')
plt.show()

# %%
# mean_mag = np.mean(np.abs(X), axis=1)
# sns.kdeplot(mean_mag, gridsize=250)
sns.heatmap(X, vmax=2, vmin=-2)
plt.show()
