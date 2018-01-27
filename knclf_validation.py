import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.utils.class_weight import compute_class_weight
from Mediumrare import gensim_nlp, predictor_model, db_tools
from imblearn.over_sampling import RandomOverSampler
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
min_blogs = 25
X, y, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=min_blogs)
n_docs = X.shape[0]
n_channel = len(labelencoder.classes_)
# %%
clf = KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree', weights='uniform')
params = {
    'n_neighbors': [3, 5, 8, 10, 15],
    'p': list(np.arange(1,2.1,.2))
}
#     'weights': ['uniform', 'distance'],
scorer = make_scorer(accuracy_score)
cv = RandomizedSearchCV(clf, cv=3, n_jobs=-1, return_train_score=True,
                        scoring=scorer, param_distributions=params, n_iter=10)
#
cv.fit(X, y)
# %%
# classw = compute_class_weight('balanced', np.unique(y), y)
kfold = 3
topN = range(1,21)
kcv = StratifiedKFold(n_splits=kfold)
clf = cv.best_estimator_
Xover, yover = RandomOverSampler().fit_sample(X, y)
accuracy = np.zeros((kfold, len(topN)))
accuracy_weighted = np.zeros((kfold, len(topN)))
for splitid, (train_idx, test_idx) in enumerate(kcv.split(Xover, yover)):
    X_train, X_test = Xover[train_idx, :], Xover[test_idx, :]
    y_train, y_test = yover[train_idx], yover[test_idx]
    clf.fit(X_train, y_train)
    y_hat = clf.predict_proba(X_test)
    # y_hat_weighted = y_hat * classw
    argmax_labels = np.argsort(y_hat, axis=1)
    # argmax_wlabel = np.argsort(y_hat_weighted, axis=1)
    ytest = np.reshape(y_test, (-1, 1))
    for N in topN:
        labels = np.reshape(argmax_labels[:,-N:], (-1, N))
        truepositive = np.any(np.equal(labels, np.repeat(ytest, N, axis=1)), axis=1)
        accuracy[splitid, N-1] = np.mean(truepositive)

        # labels = np.reshape(argmax_wlabel[:,-N:], (-1, N))
        # truepositive = np.any(np.equal(labels, np.repeat(ytest, N, axis=1)), axis=1)
        # accuracy_weighted[splitid, N-1] = np.mean(truepositive)
# %% plot
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8,6))
plt.errorbar(x=topN, y=np.mean(accuracy,axis=0), yerr=np.std(accuracy,axis=0), linewidth=2.)
# plt.errorbar(x=topN, y=np.mean(accuracy_weighted,axis=0), yerr=np.std(accuracy_weighted,axis=0), color='g')

randacc = np.sort(np.bincount(y))[::-1]
randacc = np.cumsum(randacc/randacc.sum())
plt.plot(range(1,1+len(randacc)), randacc)
diff = np.mean(accuracy,axis=0)-randacc[:accuracy.shape[1]]
plt.plot(range(1,1+len(diff)), diff, color='k')
# plt.xlim((1, 10))
# plt.ylim((0, .7))
plt.show()
