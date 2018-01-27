import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from imblearn import over_sampling as os
from imblearn import pipeline as pl

from Mediumrare import gensim_nlp, predictor_model
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
# mostsim = embedder.model.wv.most_similar()
# d2vacc = embedder.model.accuracy('/home/jdechery/code/insight/Mediumrare/questions-words.txt')
min_blogs = 25
X, y, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=min_blogs)
n_docs = X.shape[0]
n_channel = len(labelencoder.classes_)
# %%
smote = os.SMOTE(n_jobs=-1)
clf = RandomForestClassifier(n_jobs=-1, n_estimators=500)
pipeline = pl.make_pipeline(smote, clf)
scorer = make_scorer(accuracy_score)
param_range = range(1, 16, 5)

train_scores, test_scores = validation_curve(
    pipeline, X, y, param_name="smote__k_neighbors", param_range=param_range,
    cv=3, scoring=scorer, n_jobs=1)
# %%
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

plt.plot(param_range, test_scores_mean, label='SMOTE')
ax.fill_between(param_range, test_scores_mean + test_scores_std,
                test_scores_mean - test_scores_std, alpha=0.2)
idx_max = np.argmax(test_scores_mean)
plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
            label=r'Cohen Kappa: ${0:.2f}\pm{1:.2f}$'.format(
                test_scores_mean[idx_max], test_scores_std[idx_max]))

plt.title("Validation Curve with SMOTE-CART")
plt.xlabel("k_neighbors")
plt.ylabel("acc")
plt.show()
