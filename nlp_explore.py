from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Mediumrare import db_tools
import string
from itertools import chain
from sklearn.decomposition import PCA, TruncatedSVD

conn = db_tools.get_conn()
query = 'SELECT textcontent from mediumblog'
documents = conn.execute(query).fetchall()
documents = [doc[0].replace('\n','').lower() for doc in documents]
documents = [doc.replace(r"http\S+", "") for doc in documents]
documents = [doc.replace(r"http", "") for doc in documents]
documents = [doc.replace(r"@\S+", "") for doc in documents]
documents = [doc.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ") for doc in documents]
documents = [doc.replace(r"@", "at") for doc in documents]

stop_words = stopwords.words('english')
ltzr = WordNetLemmatizer()

vectorizer = CountVectorizer(decode_error='ignore', stop_words=stop_words)
doc_counts = vectorizer.fit_transform(documents)
lsa = TruncatedSVD(n_components=3)
lsa.fit(doc_counts)
# word_counts = vectorizer.fit_transform(all_tokens)


# doc_tokens = []
# for doc in documents:
#     # doc = documents[0]
#     # doc = doc[0].replace('\n','').lower()
#     tokens = word_tokenize(doc)
#     tokens = [tok for tok in tokens if not (tok in stop_words or tok in string.punctuation)]
#     tokens = [ltzr.lemmatize(ltzr.lemmatize(tok, 'n'),'v') for tok in tokens]
#     doc_tokens.append(tokens)
#
# all_tokens = list(chain.from_iterable(doc_tokens))
# vectorizer = CountVectorizer(decode_error='ignore')
# word_counts = vectorizer.fit_transform(all_tokens)

class NLProcessor():

    def __init__(self, raw_text=[], swords=None, quick_input=False):
        self.raw_text = raw_text
        self.processed_text = None
        self.flat_text = None
        self.removed_articles = None
        self.word_count = None
        self.ddiv_count = None
        if swords == None:
            self.swords = stopwords.words('english')
        else:
            self.swords = swords
        if quick_input:
            self.process_text(break_on=['.'],init_split_on=None)
            self.flatten_text()

    def set_text(self, in_text, ttype='raw', single_doc=False):
        '''takes in text (a list of text per doc) and stores it'''
        if single_doc:
            in_text = [in_text]

        if ttype=='raw':
            self.raw_text = in_text

        elif ttype=='tokenized':
            self.processed_text = in_text

    def get_text(self, ttype='tokenized', output_type='list', index=None, columns=None):
        '''returns text as a list or dataframe'''
        # set return text
        if ttype=='raw':
            return_text = self.raw_text
        elif ttype=='tokenized':
            return_text = self.processed_text
        elif ttype=='flat':
            return_text = self.flat_text

        # return in correct format
        if output_type=='list':
            return return_text  # stored as string
        elif output_type=='dataframe':
            df = pd.DataFrame(return_text,index=index,columns=columns)
            if ttype=='raw' or ttype=='flat':
                df.transpose()
            return df

    def process_text(self, in_text=None, break_on=['.','?','!'], stopwords='default', to_stem=True,
                        init_split_on='database', origdb=None, ddiv_max_length=250,keep_raw=False):
        '''does split, lower, break into ddivs, tokenize, remove swords, and stem if desired'''

        if in_text is None:
            in_text = self.raw_text

        if stopwords=='default':
            swords = self.swords

        # this is specifically for parsing the storage of my scraper
        if init_split_on=='database':
            if origdb is not None:
                if '.' in break_on:
                    p_output = process_text_sentences(in_text,origdb,swords,ddiv_max_length,to_stem,keep_raw)
                else:
                    p_output = process_text_paragraphs(in_text,origdb,swords,ddiv_max_length,to_stem)
                atext = p_output[0]
                self.removed_articles = p_output[1]

        # this is for parsing other storages
        else:
            if '.' in break_on:
                atext = [plist_to_slist([a]) for a in in_text]
                atext = [[process_paragraph(s,swords,to_stem) for s in art] for art in atext]
            else:
                atext = [a.split(break_on) for a in in_text]
                atext = [[process_paragraph(s,swords,to_stem) for s in art] for art in atext]

        self.processed_text = atext
        self.word_count = sum([len(s) for s in atext])

    def flatten_text(self,text='default'):
        '''flattens text to include simply a list of all words in each corpus/article'''
        if text=='default':
            text = self.processed_text
        self.flat_text = [flatten_list_of_lists(art) for art in text]

    def make_ddiv_count(self,text='default'):
        '''finds ddiv count per article'''
        if text=='default':
            text = self.processed_text
        self.ddiv_count = [len(a) for a in text]

    def get_ddiv_count(self):
        '''returns ddiv count'''
        return self.ddiv_count

    def get_word_count(self):
        '''returns word count'''
        return self.word_count

    def get_removed_articles(self):
        '''returns removed articles'''
        return self.removed_articles
