# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re

class data_loader(object):
    def __init__(self, data_path,flag=1):
        '''flag = 1, key=Username, flag = 0 key=Date
        '''
        # self.subsents_list = None
        self.data_path = data_path
        self.flag = flag
        if flag:
            self.field_id = "Username"
            self.alfield_id = "Date"
        else:
            self.field_id = "Date"
            self.alfield_id = "Username"
    def load_data(self):
        data_frame = pd.read_excel(self.data_path)
        t1 = data_frame['Date'].str.replace("^[0-9]*:[0-9]*\s-\s","",regex=True)
        t2 = t1.str.replace("\sавг.\s","-8-",regex=True)
        t3 = t2.str.replace("\sг.$","",regex=True)
        # ^[0-9]*:[0-9]*\s-\s remove hour:minutes
        # \sавг.\s relpace “ авг. ” by “-8-”
        # \sг.$     remove  г.
        data_frame["Date"]=t3
        self.data_frame = data_frame
    def build_data_dic(self):
        field_set = set(self.data_frame[self.field_id])
        self.field_set=field_set
        # field_list=[]
        # usersname_list=[]
        field_dict={}
        for i in field_set:
            # print (i)
            # loc Function is used to select data
            user = self.data_frame.loc[self.data_frame[self.field_id]==i]
            # print(data_list)
            # print(data_list)
            alfield_id = np.array(list(user[self.alfield_id])).reshape(-1,1)
            text = np.array(list(user["Text"])).reshape(-1,1)
            field_array = np.concatenate((alfield_id,text),axis=1)
            # usersname_list.append(field_list)
            field_dict[i]= field_array
        self.field_dict=field_dict

    def change_flag(self):
        if self.flag:
            self.field_id = "Date"
            self.alfield_id = "Username"
        else:
            self.field_id = "Username"
            self.alfield_id = "Date"

    def get_all_text(self):
        # remove url in text
        all_texts = pd.Series(self.data_frame['Text']).str.replace("http.*", "", regex=True)
        all_texts = all_texts.str.replace("pic.twitter.com.*", "", regex=True)
        all_texts = all_texts.tolist()
        self.all_texts = all_texts
    # def data_frame_to_array(self):
    #     data = self.data_frame.values
    #     return data
    
    # def clean_data(self,text):
        
    #     # clean_sentences = pd.Series(self.data_frame["Text"]).str.replace("http.*", "", regex=True)
    #     # clean_sentences = clean_sentences.str.replace("[^a-zA-Z]", " ")
    #     # self.clean_sentences = [s.lower() for s in clean_sentences]
    #     # self.clean_sentences = clean_sentences
    #     clean_sentences = re.sub(r"http.*","",text)
    #     # clean_sentences = re.sub(r"[^a-zA-Z]"," ",clean_sentences)
    #     return clean_sentences
    # def seq_joint(self):
    #     for i in self.field_dict.values():
    #         sens_list = i[:,1]
    #         print(sens_list)
    #         sens = [self.clean_data(x) for x in sens_list]
    #         text_i = " ".join(sens)
    #         print(text_i)
    #         print("**"*20)
    #         clean_text = self.clean_data(text_i)
    #         print(clean_text)
    #         # print(clean_text)
    #         tr4w = TextRank4Keyword()
    #         tr4w.analyze(clean_text, candidate_pos = ['NOUN', 'PROPN'], window_size=2, lower=True)
    #         print("*"*20)
    #         tr4w.get_keywords(10)
    #         break




if __name__ == '__main__':
    data_path = './data/Ferguson-en.xlsx'
    d = data_loader(data_path,1)
    d.load_data()
    d.build_data_dic()
    # print("*"*20)
    # print(d.field_dict['gprince1110'][0][0])
    d.change_flag()
    d.load_data()
    d.build_data_dic()
    # print("*"*20)
    # print(d.field_dict['25-8-2014'][1][1])
    # d.clean_data()
    # print(d.clean_sentences[0])
    # print(len(d.clean_sentences))
    # a=[]
    # print("-"*50)
    # for k,v in d.field_dict:
    #     print(len(k),len(v))
    #     break
    # d.get_all_text()
    # a=pd.Series(d.data_frame["Text"]).str.replace("http.*", "", regex=True)
    # b=np.array(a)
    # print(b.shape)
    d.get_all_text()
    print(type(d.all_texts))