import gensim 
import pickle

def main():
    texts = pickle.load(open("corpus/basic-full.pkl","rb"))

    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,id2word=dictionary,num_topics=8,workers=31)
    
    lda.save("models/test.model")
    pickle.dump(dictionary,open("models/lda-dict.pkl","wb"))
    pickle.dump(corpus,open("models/lda-corpus.pkl","wb"))

if __name__ == "__main__":
    main()
