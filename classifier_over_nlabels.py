from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, confusion_matrix
from Mediumrare import db_tools, gensim_nlp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pickle
import pprint
conn = db_tools.get_conn()
query = 'SELECT id, claps, blog_url FROM mediumclean ORDER BY id'
# %% get blogdata df
rows = conn.execute(query).fetchall()
blogdf = pd.DataFrame(rows, columns=['id','claps','url'])
blogdf['channel'] = blogdf['url'].map(lambda x: x.split('/')[3])
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()

# %%
doc_vectors = embedder.model.docvecs
colnames = ['dim%d'%x for x in range(len(doc_vectors[0]))]
embedded_vectors = pd.DataFrame(np.asarray(doc_vectors), columns=colnames)
blogdf = pd.concat([blogdf, embedded_vectors], axis=1)

# %% classification on channels
#remove rows with less than 600 data points
n_blogs = blogdf['channel'].value_counts()
acc = []
nlabels = range(6,20)
for nlabel in nlabels:
    keep_channels = n_blogs[:nlabel].index
    channeldf = blogdf.loc[blogdf['channel'].isin(keep_channels), :]

    X = channeldf[colnames].as_matrix()
    labelencoder = LabelEncoder().fit(channeldf['channel'])
    y = labelencoder.transform(channeldf['channel'])
    # %
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=75)
    clf_fname = '/home/jdechery/forest_classifier.pkl'

    cv = StratifiedKFold(n_splits=5)
    metric_names = ('confusion', 'accuracy')
    metrics = dict(zip(metric_names, (confusion_matrix, accuracy_score)))
    scores = dict(zip(metric_names, ([],[])))
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        for metric in metric_names:
            scores[metric].append(metrics[metric](y_test, y_hat))
    print(scores['accuracy'])
    acc.append(scores['accuracy'])

# %%
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(12,8))

plotdf = pd.DataFrame(np.asarray(acc).T, columns=[str(x) for x in nlabels])
ax.errorbar(nlabels, plotdf.mean(), yerr=plotdf.std(), fmt='b')
ax.plot(nlabels, 1/np.asarray(nlabels), 'k')
plt.xlabel('number of labels')
plt.ylabel('mean accuracy')
plt.legend(['random','classifier'])
