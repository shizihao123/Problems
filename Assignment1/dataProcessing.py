#!/usr/bin/env python
import math
import nltk
import os
import re
from os import listdir
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem.porter import *

basePath = "/home/jun/Desktop/ICML/"

def calculateDocTFVal(filePath):                    #function
    f = open(filePath, "r")                     #read from files
    context = f.read()
    f.close()

    rawRes = nltk.word_tokenize(context)        #word tokenization

    stops = set(stopwords.words('english'))     #remove stopWords
    filterRes = []
    for word in rawRes :
        if(word.lower() not in stops and word.__len__() != 1 and word.__len__() <= 20 ):
            if(re.match("^[A-Za-z]+$", word)):                 #remove the word which is not in an alphabet or digital  (my way to get rid of messy code)
                filterRes.append(word)

    stemmer = PorterStemmer()  # word stemming
    filterRes = [stemmer.stem(word.lower()) for word in filterRes]

    totalWords = filterRes.__len__()            #which is the final wordList to be handled
    res = set(filterRes)                        #turn list to set to avoid duplication

    tf = {}                                     #calculate tf value in a document
    for word in filterRes:
        if word not in tf.keys():
            tf[word] = 1
        else:
            tf[word] = tf[word] + 1

    for word in tf.keys():
        tf[word] = tf[word] / totalWords

    return [res, tf]                                   #return wordList and tf value in a document


subdirs = [(basePath + f) for f in listdir(basePath)]  #calculate  subdirs under base path
dict = {}
wordList = set()
res =[]
dir = {}
for subdir in subdirs:                                 #calculate tf value of each document under 15 classes
    dir[subdir] = [subdir + "/" + filePath for filePath in listdir(subdir)]
    for filePath in dir[subdir]:
        res =  calculateDocTFVal(filePath)
        wordList = wordList | res[0]                   #aggregate all the words to the wordList
        dict[filePath] = res[1]                        #save all the file's TF value

totalDoc = dict.keys().__len__()                       #total number of documents
IDF = {}
for word in wordList:
    count = 0
    for name in dict.keys():                           #calculte each word's IDF value
        if (word in dict[name].keys()):
            count = count + 1
    IDF[word] =  math.log2(totalDoc / count)

TFIDF = {}
for file in dict.keys():                               #calculete eachword's TFIDF value and them to the dic to classify
    TFIDF[file] = {}
    for word in dict[file]:
       TFIDF[file][word] = dict[file][word] * IDF[word]

wordListInOrder = sorted(list(wordList))                 #sort the wordList by lexicographical order

outBaseDir = "/home/jun/Desktop/Results/"              #output base directory
output = [f for f in listdir(basePath)]
for subdir in output:                                  #output by 15 classes
    if not os.path.exists(outBaseDir + subdir):
        os.makedirs(outBaseDir + subdir)
    f = open(outBaseDir + subdir + "/results.txt", "w")
    for file in dir[basePath + subdir]:
        i = 0
        for word in wordListInOrder:
            if word in TFIDF[file].keys() :          #if the tfidf value equals 0 , ignore
                # print (i, ":", TFIDF[file][word], end =" ", file = f)
                print (i, ":", TFIDF[file][word], " ")
            i = i + 1
        print ("\n",  file)
    f.close()

f = open("/home/jun/Desktop/wordList", "w")            #output wordList
for word in wordListInOrder:  # output wordList
    # print(word, end=" ", file=f)
    print(word, " ")
# print("\n", file=f)
f.close()
