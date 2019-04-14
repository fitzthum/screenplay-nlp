# Builds a preprocessed corpus from the database. 
#
# Resulting corpus should be perfect for LDA or Word2Vec
#

import os
import logging

import progressbar 
import pandas 
import numpy
import re 
import pickle

import itertools 
import multiprocessing 


import MySQLdb
from db import connect

import spacy
import gensim
from nltk.corpus import stopwords 


# some global variables. necessary for multiprocessing :(
BASE_DIR = "scripts"

# TODO: add some screenplay specific words
stop_words = set(stopwords.words('english'))
spc = spacy.load('en', disable=['parser', 'ner'])


    
def process_doc(path):
    HEAD_SKIP = 10 
    doc = []

    with open(os.path.join(BASE_DIR,path[0]),"r") as f:
        for line in itertools.islice(f,HEAD_SKIP,None):
            # some simple cleanup 
            line = re.sub('\s+', ' ', line)
            line = re.sub("\'", "", line)
            
            # stopwords, lemmatize, remove uppercase
            for tkn in gensim.utils.simple_preprocess(line,deacc=True):
                if tkn not in stop_words and not tkn.isupper():
                    tkn = spc(tkn)[0].lemma_
                    doc.append(tkn)
    
    return doc


def get_corpus_paths(n = None):
    cursor = connect()

    if n:
        query = "SELECT filename FROM script_basics LIMIT {}".format(n)
    else:
        query = "SELECT filename FROM script_basics"
    
    cursor.execute(query)
    result = cursor.fetchall()
   
    cursor.close()
    return result

def main():
    N_THREADS = 20 

    print("Loading Paths")
    paths = get_corpus_paths()
    print("Paths Loaded")

    data = []


    #for path in progressbar.progressbar(paths):
    #        data.append(process_doc(path))

    print("Processing Documents")
    with multiprocessing.Pool(N_THREADS) as p: 
        data = p.map(process_doc,paths)


    print("Generating Bigrams")
    bigrams = gensim.models.Phrases(data)
    bigram_mod = gensim.models.phrases.Phraser(bigrams)
    data = bigram_mod[data]

    print("Generating Trigrams")
    trigrams = gensim.models.Phrases(data)
    trigram_mod = gensim.models.phrases.Phraser(trigrams)
    data = trigram_mod[data]    

    print("Dumping Data")
    pickle.dump(data,open("corpus/basic-full.pkl","wb"))

if __name__ == "__main__":
    main()
