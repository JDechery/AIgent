from sklearn.linear_model import LinearRegression
from Mediumrare import db_tools, gensim_nlp
import pandas as pd

conn = db_tools.get_conn()
query = 'SELECT id, claps, blog_url from mediumclean'
# %% get blogdata df
rows = conn.execute(query).fetchall()
blogdf = pd.DataFrame(rows, columns=['id','claps','url'])
blogdf.set_index('id', inplace=True)
blogdf['channel'] = blogdf['url'].map(lambda x: x.split('/')[3])

# %%
embedder = gensim_nlp.DocEmbedder()
embedder.load_model()
doc_vectors = embedder.model.docvecs
