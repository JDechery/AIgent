import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from Mediumrare import gensim_nlp, predictor_model
# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
min_blogs = 25
X, y, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=min_blogs)
n_docs = X.shape[0]
n_channel = len(labelencoder.classes_)
# %%
clf = GradientBoostingClassifier(n_estimators=50, warm_start=True)
pipeline = make_pipeline(clf)
scorer = make_scorer(cohen_kappa_score)
param_range = [None, 'sqrt', 'log2']

train_scores, test_scores = validation_curve(
    pipeline, X, y, param_name="clf__max_features", param_range=param_range,
    cv=3, scoring=scorer, n_jobs=-1)
# %
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots()
plt.plot()

plt.show()
