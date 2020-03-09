from dataLoader import data_loader
import numpy as np
import pandas as pd
import re
import nltk
from nltk import word_tokenize
import string
from nltk.stem import WordNetLemmatizer


__author__ = "Zhao Chi"

# stopwords from https://countwordsfree.com/stopwords/english/txt



class data_cleaner(object):
    def __init__(self,stopwords_path,glove_voc_path):
        self.glove_voc = self.load_txt(glove_voc_path)
        self.stopwords_path = stopwords_path
    
    def load_txt(self,txtpath):
        f = open(txtpath,"r")
        words = []
        for line in f.readlines():
            words.append(str(line.strip()))
        return words

    def stopwords_generation(self):

        stopwords = self.load_txt(self.stopwords_path)
        wanted_POS = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW'] 
        stopwords_from_text = []
        for word in self.processed_POS_tag:
            if word[1] not in wanted_POS:
                stopwords_from_text.append(word[0])
        stopwords_from_punctuation= list(str(string.punctuation))
        self.stopwords=list(set(stopwords+stopwords_from_text+stopwords_from_punctuation))

    def remove_non_printable_characters(self,sents_list):
        """
        sents_list: list of sentences string format need to be cleaned
        """
        sents_str = " ".join(sents_list)
        sents_str = sents_str.lower()
        printable = set(string.printable)
        sents_list = list(filter(lambda x: x in printable,sents_str))
        sents_str = "".join(sents_list)
        return sents_str
    
    def word_Lemmatization(self,sents_list,txtpath):
        glove_voc = self.glove_voc# loading glove_voc 
        sents_str = self.remove_non_printable_characters(sents_list)
        words = word_tokenize(sents_str)
        self.words = words
        POS_tag = nltk.pos_tag(words)
        self.origin_POS_tag = POS_tag
        wordnet_lemmatizer = WordNetLemmatizer()
        adjective_tags = ['JJ','JJR','JJS']
        lemmatized_words = []
        for word in POS_tag:
            if word[1] in adjective_tags:
                lemmatized_words.append(str(wordnet_lemmatizer.lemmatize(word[0],pos="a"))) 
                # a denotes adjective in "pos" 
            else:
                lemmatized_words.append(str(wordnet_lemmatizer.lemmatize(word[0]))) #default POS = noun
        self.lemmatized_words=lemmatized_words
        POS_tag = nltk.pos_tag(lemmatized_words)
        self.processed_POS_tag = POS_tag
        # print(len(lemmatized_words))
        # print(len(set(lemmatized_words)))

    def remove_stopwords(self):
        processed_data = []
        for word in self.lemmatized_words:
            if word not in self.stopwords:
                processed_data.append(word)
        self.processed_data = processed_data
    
    def remove_non_word_vocabulary(self,txtpath):
        """
        remove word not in glove vocabulary and repeated words
        """
        glove_voc = self.glove_voc
        datas=list(set(self.processed_data))
        words_data=[]
        for i in datas:
            if i in glove_voc and i not in words_data:
                words_data.append(i)
        self.processed_data=words_data
        return words_data

if __name__ == "__main__":
    stopwords_path = "./stopwords_en"
    glove_voc_path = "./data/glove_vocabulary.txt"
    data_path = './data/Ferguson-en.xlsx'
    d = data_loader(data_path,0)
    d.load_data()
    d.get_all_text()
    all_texts = d.all_texts
    cleaner = data_cleaner(stopwords_path=stopwords_path,glove_voc_path=glove_voc_path)
    # sents_str = cleaner.remove_non_printable_characters()
    cleaner.word_Lemmatization(sents_list=all_texts,txtpath=glove_voc_path)
    print("---"*20)
    print(cleaner.lemmatized_words)
    print(len(cleaner.lemmatized_words))
    print("stopblackgenocide" in cleaner.lemmatized_words)
    cleaner.stopwords_generation()
    cleaner.remove_stopwords()
    print(len(cleaner.processed_data))
    print(len(list(set(cleaner.processed_data))))
    print("*"*20)
    cleaner.remove_non_word_vocabulary(txtpath=glove_voc_path)
    print("stopblackgenocide" in cleaner.lemmatized_words)
    print(len(list(set(cleaner.processed_data))))
    print(len(cleaner.processed_data))
    # print(cleaner.processed_data)