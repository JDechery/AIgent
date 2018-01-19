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
# blogdf.set_index('id', inplace=True)
blogdf['channel'] = blogdf['url'].map(lambda x: x.split('/')[3])
# labels = LabelEncoder.fit(blogdf['channel'])
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
# %%
word = 'trump'
print('most similar word to', word)
pp = pprint.PrettyPrinter(indent=2)
sim = embedder.model.wv.most_similar(word)
sim = [(x, round(100*y)/100) for x, y in sim]
pp.pprint(sim)
# %%
doc_vectors = embedder.model.docvecs
# tags = list(doc_vectors.doctags.keys())
# tag2id = list(map(lambda x: int(x.split('_')[1])+1, tags))
# embedded_vectors = [doc_vectors[key] for key in tags]
colnames = ['dim%d'%x for x in range(len(doc_vectors[0]))]
embedded_vectors = pd.DataFrame(np.asarray(doc_vectors), columns=colnames)
blogdf = pd.concat([blogdf, embedded_vectors], axis=1)
# %% plot embedding
matplotlib.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(embedded_vectors, vmin=-2, vmax=2)
plt.xlabel('embedded dimension')
plt.ylabel('blog post')
plt.yticks(range(0,6500,500), [str(x) for x in range(0,6500,500)])
plt.xticks(range(0,101,20), [str(x) for x in range(0,100,20)])
plt.title('doc2vec embedding of medium blogposts')
plt.show()
# %%
fig, axs = plt.subplots(nrows=2, figsize=(8, 8))
sns.distplot(embedded_vectors.as_matrix().flatten(), bins=400, kde_kws={'gridsize':1000}, ax=axs[0])
axs[0].set_xlim([-.5, .5])
axs[0].set_xlabel('magnitude')
axs[0].set_ylabel('probability density')

sns.heatmap(np.cov(embedded_vectors.as_matrix().T))
plt.show()
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
keep_channels = n_blogs[n_blogs>=350].index
channeldf = blogdf.loc[blogdf['channel'].isin(keep_channels), :]

X = channeldf[colnames].as_matrix()
labelencoder = LabelEncoder().fit(channeldf['channel'])
y = labelencoder.transform(channeldf['channel'])
# %%
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(channeldf['channel'])
plt.xticks(rotation='vertical')
plt.xlabel('publisher tag')
plt.ylabel('# posts')
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
sns.heatmap(conf_mat_norm, vmin=0, vmax=.5)
ax.set_title('accuracy = %.2f+/-%.2f'%(np.mean(scores['accuracy']), np.std(scores['accuracy'])))
plt.xticks(np.arange(.5,6.5,1), labelencoder.classes_, rotation='vertical')
plt.yticks(np.arange(.5,6.5,1), labelencoder.classes_, rotation='horizontal')
plt.xlabel('predicted channel')
plt.ylabel('actual channel')
plt.show()
fig.savefig('/home/jdechery/code/insight/confusionmat.png',set_resolution=300)
