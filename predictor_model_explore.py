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

conn = db_tools.get_conn()
query = 'SELECT id, claps, blog_url from mediumclean'
# %% get blogdata df
rows = conn.execute(query).fetchall()
blogdf = pd.DataFrame(rows, columns=['id','claps','url'])
blogdf.set_index('id', inplace=True)
blogdf['channel'] = blogdf['url'].map(lambda x: x.split('/')[3])
# labels = LabelEncoder.fit(blogdf['channel'])
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
doc_vectors = embedder.model.docvecs
tags = list(doc_vectors.doctags.keys())
tag2id = list(map(lambda x: int(x.split('_')[1])+1, tags))
embedded_vectors = [doc_vectors[key] for key in tags]
colnames = ['dim%d'%x for x in range(len(embedded_vectors[0]))]
embedded_vectors = pd.DataFrame(embedded_vectors, columns=colnames)
embedded_vectors['id'] = tag2id

blogdf = embedded_vectors.join(blogdf, how='left', on='id')

# %% regression on claps
# sns.heatmap(embedded_vectors.iloc[:,:-1])
# sns.heatmap(X.cov())
# X = blogdf[colnames]
# y = np.log(blogdf['claps'])
# rgr = LinearRegression()
# rgr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
# %%
# scores = cross_val_score(rgr, X, y, n_jobs=-1, cv=10, scoring=make_scorer(median_absolute_error))
# print(scores)

# %% classification on channels
#remove rows with less than 600 data points
n_blogs = blogdf['channel'].value_counts()
keep_channels = n_blogs[n_blogs>=300].index
channeldf = blogdf.loc[blogdf['channel'].isin(keep_channels), :]

X = channeldf[colnames].as_matrix()
labelencoder = LabelEncoder().fit(channeldf['channel'])
y = labelencoder.transform(channeldf['channel'])
# %%
# clf = LinearSVC(class_weight='balanced')
# clf_fname = '/home/jdechery/svc_classifier.pkl'
clf = RandomForestClassifier(n_jobs=-1, n_estimators=50)
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
# %% save
with open(clf_fname, 'wb') as f:
    pickle.dump((clf, labelencoder, channeldf), f)
# %% plot mean confusion matrix
conf_mat_norm = list(map(lambda x: np.asarray(x), scores['confusion']))
conf_mat_norm = list(map(lambda x: x/x.sum(axis=1)[:, np.newaxis], conf_mat_norm))
conf_mat_norm = np.mean(np.stack(conf_mat_norm, axis=2), axis=2)
# %%
matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_mat_norm, vmin=1/6, vmax=.35)
ax.set_title('accuracy = %.2f+/-%.2f'%(np.mean(scores['accuracy']), np.std(scores['accuracy'])))
plt.xticks(np.arange(.5,6.5,1), labelencoder.classes_, rotation='vertical')
plt.yticks(np.arange(.5,6.5,1), labelencoder.classes_, rotation='horizontal')
plt.show()
