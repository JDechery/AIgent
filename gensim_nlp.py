from gensim import corpora, models
import gensim
# import multiprocessing
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
import string
from Mediumrare import db_tools
from itertools import chain
import pudb
# cores = multiprocessing.cpu_count()
# assert gensim.models.doc2vec.FAST_VERSION > -1 # "This will be painfully slow otherwise"

conn = db_tools.get_conn()
query = 'SELECT textcontent from mediumblog'
# %%
class RawDocToCleanDoc(object):
    def __init__(self, conn, query):
        self.query = query
        self.conn = conn
        self.rows = self.conn.execute(self.query)
        self._nrows = self.count_rows()

    def clean_document(self,doc):
        doc = gensim.utils.simple_preprocess(doc)
        # doc = gensim.utils.lemmatize(doc)
        return doc

    def count_rows(self):
        return self.conn.execute("SELECT COUNT(*) FROM mediumblog").fetchone()[0]

    def __len__(self):
        return self._nrows

    def __iter__(self):
        self.rows = self.conn.execute(self.query)
        return self

    def __next__(self):
        # rows = self.conn.execute(self.query)
        doc = self.rows.fetchone()
        if doc is None:
            raise StopIteration
        else:
            return self.clean_document(doc[0])
# %%
def get_clean_docs():
    return RawDocToCleanDoc(conn, query)

class TaggedDoc(object):
    def __init__(self):
        self.docs = get_clean_docs()
        self.ii = 0
    def __iter__(self):
        self.ii = 0
        return self
    def __next__(self):
        nextdoc = next(self.docs)
        if nextdoc is None:
            raise StopIteration
        else:
            TDoc = models.doc2vec.TaggedDocument(words=next(self.docs), tags=['SENT_%s' % self.ii])
            self.ii = self.ii+1
            return TDoc
# %%
class DocEmbedder(object):
    def __init__(self):
        self.docs = get_clean_docs()
        self.model = None

    def train_model(self, **modelargs):
        if self.model is None:
            self.model = models.Doc2Vec(**modelargs)
        self.model.build_vocab(TaggedDoc())
        # self.model.finalize_vocab()
        for epoch in range(10):
            self.model.train(TaggedDoc(), total_examples=len(self.docs), epochs=1)
            self.model.alpha -= 0.002 # decrease the learning rate
            self.model.min_alpha = self.model.alpha # fix the learning rate, no decay
            self.model.train(TaggedDoc(), total_examples=len(self.docs), epochs=1)

    def save_model(self, fname='/tmp/model.model'):
        self.model.save(fname)

# %%
class DatabaseToMM():
    def __init__(self, conn, query):
        self.conn = conn
        self.query = query
        self.dictionary = corpora.Dictionary()

    def get_raw_docs(self):
        return RawDocToCleanDoc(self.conn, self.query)

    def load_dictionary(self, fname):
        self.dictionary.load(fname)

    def make_dictionary(self, tosave=False, fname='/tmp/dict'):
        docs = self.get_raw_docs()
        for doc in docs:
            # doc = gensim.utils.simple_preprocess(doc)
            self.dictionary.add_documents([doc])
        self.dictionary.compactify()
        if tosave:
            self.dictionary.save()

    def get_clean_bow(self):
        docs = self.get_raw_docs()
        for doc in docs:
            # doc = gensim.utils.simple_preprocess(doc)
            yield self.dictionary.doc2bow(doc)

    def save_serialized_docs(self, fname):
        corpora.MmCorpus.serialize(fname, self.get_clean_bow())

    def load_serialized_docs(self, fname):
        corpus = corpora.MmCorpus(fname=fname)

if __name__=='__main__':
    conn = db_tools.get_conn()
    query = 'SELECT textcontent from mediumblog'
    default_fname = '/tmp/corpus.mm'
    serializer = DatabaseToMM(conn, query)
    serializer.make_dictionary()
    serializer.save_serialized_docs(default_fname)
