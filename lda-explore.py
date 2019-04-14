import gensim
import pickle
import pyLDAvis
import pyLDAvis.gensim

lda = gensim.models.ldamulticore.LdaMulticore.load("models/test.model")

# might be better to recompute this from the saved corpus
dictionary = pickle.load(open("models/lda-dict.pkl","rb"))
corpus = pickle.load(open("models/lda-corpus.pkl","rb"))
print(len(corpus))

vis = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
html = pyLDAvis.prepared_data_to_html(vis)
with open("lda.html","w") as f:
    f.write(html)

