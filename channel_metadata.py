from Mediumrare import predictor_model, gensim_nlp
import pandas as pd
import numpy as np
import re
from itertools import chain
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
_, _, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=15)
bad_idx = channeldf['pub_date'].map(lambda x: len(x))<10
channeldf = channeldf.drop(channeldf.index[bad_idx])
channeldf['pub_date'] = pd.to_datetime(channeldf['pub_date'], format='%b %d, %Y')

remove_bad_chars = lambda word: re.sub('[{}"]', '', word)
channeldf['tags'] = channeldf['tags'].map(remove_bad_chars)

# %%
def most_recent_pubs(channeldf, channel, npub=3):
    channel_idx = channeldf['channel'] == channel
    meaningful_cols = ['id','claps','tags','author','url', 'title', 'pub_date','channel']
    most_recent = channeldf.loc[channel_idx, 'pub_date'].sort_values(ascending=False)[:npub].index
    return channeldf.loc[most_recent, meaningful_cols]

def mean_channel_claps(channeldf, channel):
    channel_idx = channeldf['channel'] == channel
    avg_claps = channeldf.loc[channel_idx, 'claps'].mean()
    return round(avg_claps*10)/10

def most_common_tags(channeldf, channel, ntags=3):
    channel_idx = channeldf['channel'] == channel
    tags = list(chain(*channeldf.loc[channel_idx, 'tags'].map(lambda x: x.split(',')).values))
    common_tags = pd.Series(tags).value_counts()[:ntags]
    return common_tags.index.tolist()
