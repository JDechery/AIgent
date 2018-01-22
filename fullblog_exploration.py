from Mediumrare import db_tools
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

conn = db_tools.get_conn()
tag_query = 'SELECT id, tags, claps from mediumblogfull ORDER BY id'

blogrows = conn.execute(tag_query).fetchall()
remove_bad_chars = lambda word: re.sub('[{}"]', '', word)

tags = [remove_bad_chars(row[1]) for row in blogrows]
claps = [row[2] for row in blogrows]

countvectorizer = CountVectorizer(input='content', strip_accents='unicode', min_df=2)
tag_counts = countvectorizer.fit_transform(tags)
voc = countvectorizer.vocabulary_

tagdf = pd.DataFrame(data=tag_counts.todense(), columns=voc)
tagdf['claps'] = claps
# %%
# tagdf.sum().hist(bins=range(26), grid=False)
tagdf['claps'].hist(bins=range(0,2000,100))
plt.xlim((0,2000))
# %%
common_tags_bool = tagdf.sum() > 30
common_tags_bool['claps'] = False
mean_claps = tagdf.loc[:,common_tags_bool].apply(lambda x: tagdf.loc[x>0, 'claps'].mean())
mean_claps.sort_values(ascending=False)
