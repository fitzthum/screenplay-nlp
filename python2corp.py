import pickle
import sys

corpus = pickle.load(open(sys.argv[1],"rb"))
print(corpus)

with open("im-corp,txt","w") as f:
    f.write(corpus)
