from Mediumrare import gensim_nlp, predictor_model, db_tools
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import re
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import matplotlib
# %% load tags
conn = db_tools.get_conn()
tag_query = 'SELECT id, tags, claps, cleantext from mediumcleanfull ORDER BY id'

blogrows = conn.execute(tag_query).fetchall()
remove_bad_chars = lambda word: re.sub('[{}"]', '', word)

tags = [remove_bad_chars(row[1]) for row in blogrows]
claps = [row[2] for row in blogrows]
ids = [row[0] for row in blogrows]

countvectorizer = CountVectorizer(input='content', strip_accents='unicode', min_df=2)
tag_counts = countvectorizer.fit_transform(tags)
# tag_counts = TfidfTransformer().fit_transform(tag_counts)
voc = countvectorizer.vocabulary_

tagdf = pd.DataFrame(data=tag_counts.todense(), columns=voc)
tagdf['claps'] = claps
tagdf['id'] = ids
tagdf['tags'] = tags
# %% get training examples
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
min_blogs = 25
channeldf, wordcols = predictor_model.data_base2frame(embedder.model, min_blogs=min_blogs)
clf, labelencoder, _ = predictor_model.load_classifier()
channeldf.drop(['id','tags','claps'], axis=1, inplace=True)
fulldf = tagdf.merge(channeldf, left_index=True, right_index=True, how='left')
fulldf['untested'] = fulldf['dim0'].isnull()
# %%
# X_untested = fulldf.loc[fulldf.untested, wordcols].as_matrix()
tag_pop_by_chan = fulldf[list(voc.keys())+['channel']].groupby('channel').mean()
# tag_pop_by_chan = tag_pop_by_chan.apply(lambda x: x/sum(x), axis=1)
# mean tag vector for each channels
# tag_pop_by_chan = tag_pop_by_chan.mean(axis=1)
# %%
plotdata = []
untested_idx = np.where(fulldf.untested)[0]
for idx in untested_idx[:500]:

    words = blogrows[idx][3]
    wordvec = embedder.model.infer_vector(words).reshape(1,-1)
    # X = fulldf.loc[idx, wordcols].as_matrix()
    classprob = clf.predict_proba(np.atleast_2d(wordvec))
    # rankorder = pd.Series(index=labelencoder.classes_, data=classprob[0])
    class_order = np.argsort(classprob[0])[::-1]
    # rankorder = pd.Series(index=labelencoder.classes_[class_order], data=range(len(class_order)))

    tags = fulldf.loc[idx, voc].loc[fulldf.loc[idx,voc]!=0].index
    for tag in tags:
        plotdata.append(tag_pop_by_chan[tag].values[class_order])
    # tag_overlap = tag_pop_by_chan[tags].sum(axis=1).sort_values()
    # tag_overlap = fulldf.loc[:, list(tags)+['channel']].groupby('channel').mean().mean(axis=1)

    # rankproj = []
    # for label in class_order:
        # chan = labelencoder.classes_[label]
        # proj = np.dot(fulldf.loc[idx,voc], tag_pop_by_chan.loc[chan,:])/np.dot(fulldf.loc[idx,voc],fulldf.loc[idx,voc])
        # rankproj.append(proj)

    # plotdata.append(rankproj)
    # ovdf = pd.concat([rankorder, tag_overlap], axis=1)
    # ovdf.columns=['rank','overlap']

    # plotdata += list(ovdf.itertuples(index=False, name=None))
    # fulldf.loc[~fulldf.index.isin([idx]), list(tags)+['channel']].groupby('channel').sum()
#     n_shared_tags = dict.fromkeys(labelencoder.classes_, 0)
#
# for tag in tags:
#     n_shared = fulldf.loc[fulldf[tags[0]]!=0, 'channel'].value_counts()
#     for key in n_shared.index:
#         n_shared_tags[key] += n_shared[key]
plotdata = np.asarray(plotdata)
# %%
# x, y = zip(*plotdata)
matplotlib.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(8,6))
# for r in plotdata:
#     plt.scatter(x=range(1,1+len(class_order)), y=r, s=3, c='k')
dta = plotdata.copy()
dta[dta==0] = np.nan
dta = np.nanpercentile(dta,.75,axis=0)[::-1]
plt.plot([5,5], [-.1,.1], color=[.2,.2,.2])
plt.plot(pd.rolling_apply(dta,3,np.mean), linewidth=3)
plt.xlabel('publisher rank')
plt.ylabel('tag similarity')
plt.text(5.75, .0042, 'top 5 publishers', color=[.2,.2,.2])
plt.yticks([])
plt.xticks([])
plt.ylim(.0022,.0045)
plt.tight_layout()
plt.savefig('/home/jdechery/code/insight/Mediumrare/tag_similarity.png',dpi=300)
# plt.show()
