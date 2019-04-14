from gensim.models import Word2Vec
import pickle
import sys 



def main():
    words = ['love','money','america']
    print("Loading")
    model = Word2Vec.load(sys.argv[1])

    for word in words:
        print("Nearest Neighbors for {}".format(word))
        print(model.wv.most_similar(word))

if __name__ == "__main__":
    main()
