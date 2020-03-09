import numpy as np
import pandas as pd
from dataLoader import data_loader
import os
import sys
import re

# data_path = './data/Ferguson-en.xlsx'
# d = data_loader(data_path,0)
# d.load_data()
# # print(d.data_frame)
# # print(list(d.data_frame["Username"]))
# # data = d.data_frame_to_array()
# # print(data.shape)
# # d.build_data_dic()
# # print(d.username_tup)
# d.build_data_dic()
# # print(d.username_dict)
# data = d.data_frame_to_array()
# print(data)

# a=[1,2,3]
# b=[4,5,6]
# print(dict(zip(a,b)))
# c={}
# d=np.concatenate((a,b),axis=1)
# c["p"]=d
# print(d[:,1])

# a= range(-5,5)
# b= list(filter(lambda x: x<0,a))
# print(b)
# a=[["1",2],[3,4]]
# print("1" in a)

# b=[1,1,2,3,2]
# c=[]
# [c.append(x) for x in b if x not in c]
# print(c)

# a=["123213 http://asd","123 http[://asdsaddas]"]
# b=pd.Series(a).str.replace("http.*", "", regex=True).tolist()
# print(b)

def extract_word_vectors():
    # Extract word vectors
    all_vocabulary_from_glove = []
    f = open('./data/glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        all_vocabulary_from_glove.append(values[0])
    f.close()
    print(len(all_vocabulary_from_glove))
    return all_vocabulary_from_glove

# all_vocabulary_from_glove = extract_word_vectors()

# np.savetxt("glove_vocabulary.text",all_vocabulary_from_glove,fmt='%s')

# f = open()


a=np.array(['aaa','xxx','aaa',"xxx"])
d=a.reshape(-1,1)
# b=np.array(['aaa','xxx','aaa',"xxx"]).reshape(-1,1)
# c=np.concatenate((a,b),axis=1)
print(a.shape)
# def load_txt(txtpath):
#     f = open(txtpath,"r")
#     words = []
#     for line in f.readlines():
#         words.append(str(line.strip()))
#     return words
# glove_voc = load_txt('./data/glove_vocabulary.txt')
# # print(glove_voc)


# print(len(a))
# for i in a:
#     if i not in glove_voc:
#         # print(i)
#         a.remove(i)
# print(len(a))

