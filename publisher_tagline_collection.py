from Mediumrare import db_tools, gensim_nlp, predictor_model
from sqlalchemy import text
conn = db_tools.get_conn()
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
_, _, labelencoder, channeldf = predictor_model.reorg_for_training(embedder.model, min_blogs=25)
channels = channeldf.channel.unique()
# %%

tagline = {}
for chan in channels:
    tagline[chan] = input(chan + ':')
# %% make table
print(tagline)
creation_query = '''CREATE TABLE channeltags (
                    id serial PRIMARY KEY,
                    channel TEXT,
                    tagline TEXT)'''

conn.execute(creation_query)
# %% insert rows
insert_query = text("""INSERT into channeltags
                  (channel, tagline)
                  VALUES (:channel, :tagline)""")
for key, val in tagline.items():
    conn.execute(insert_query, channel=key, tagline=val)
