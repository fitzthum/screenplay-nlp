import sys

from gensim.models import Word2Vec
import progressbar
import pickle


def main():
    print("Loading...")
    corpus = pickle.load(open(sys.argv[1],"rb"))
    print("Training,..")
    model = Word2Vec(corpus, min_count=3,workers=30)    
    model.save("models/w2v.model")
if __name__ == "__main__":
    main()

