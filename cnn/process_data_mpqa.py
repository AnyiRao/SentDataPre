#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:54:56 2017

@author: anyi
"""

import os
import cPickle
import nltk
import csv
import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
import os.path
from process_data_common import load_bin_vec, add_unknown_words, get_W

train_fn = 'data/mpqa/mpqa.csv'

def read_csv(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        cr = csv.reader(f, delimiter=',')
        for l in cr:
            x.append(l[0][2:].replace('\\', ' ').lower())
            y.append(int(l[0][0]))   # [1, n] in raw file
    return (x, y)

def build_data_cv(cv=10,clean_string=True):
    """
    Loads data 
    """
    revs = []
    vocab = defaultdict(float)
    train_x,train_y = read_csv(train_fn)
    for i in range(len(train_x)):
        rev = []
        rev.append(train_x[i].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {"y":train_y[i],
                  "text": orig_rev,
                  "num_words": len(orig_rev.split()),
                  "split": np.random.randint(0,cv)
                }
        revs.append(datum)
    return revs, vocab

def clean_str(string, TREC=False):
    re.compile('(http|https):\/\/[^\/"]+[^ |"]*')
    string = re.sub('(http|https):\/\/[^\/"]+[^ |"]*', "http", string)
    match = re.search("(#\w+)", string)
    if match is not None:
        for i in match.groups():
            string += " " + i + " "
#    string += (" " + "chr200" + " ") * (len([chr(ord(input_x)) for input_x in string if ord(input_x) > 130]) / 4)
    string = ''.join([chr(ord(input_x)) for input_x in string if ord(input_x) < 130]). \
                replace("\\\\", "").replace("&gt;", ">"). \
                replace("&lt;", "<").replace("&amp;", "&")
#    string += (" " + "chr201" + " ") * (len(string.split()))
#    string += (" " + "chr202" + " ") * string.count(":)")
#    string += (" " + "chr203" + " ") * string.count("@")
#    string += (" " + "chr204" + " ") * string.count("http")
#    string += (" " + "chr205" + " ") * string.count(":(")
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def process_data(file_name):
    if os.path.isfile(file_name):
        print "file {} already exists".format(file_name)
        return

    print "creating dataset..."

    # load data
    print "loading data...",
    revs, vocab = build_data_cv(cv=10,clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)

    # load word2vec
    print "loading word2vec vectors...",
    w2v_file = 'data/GoogleNews-vectors-negative300.bin'
    w2v = load_bin_vec(w2v_file, vocab)
    print "num words already in word2vec: " + str(len(w2v))
    print "word2vec loaded!"

    #Addind random vectors for all unknown words
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)

    # dump to pickle file
    cPickle.dump([revs, W, W2, word_idx_map, vocab, max_l], open(file_name, "wb"))

    print "dataset created!"

def get_idx_from_sent(sent, word_idx_map, max_l, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]    

if __name__ == '__main__':
    process_data("data/processed/mpqa.p")
    x = cPickle.load(open("data/processed/mpqa.p","rb"))
    revs, W, W2, word_idx_map, vocab ,max_l = x[0], x[1], x[2], x[3], x[4], x[5]
    datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l,k=300, filter_h=5)